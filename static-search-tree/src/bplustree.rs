use std::{fmt::Debug, iter::zip, simd::Simd};

use itertools::Itertools;

use crate::node::{BTreeNode, MAX};
use crate::{prefetch_index, prefetch_ptr, SearchIndex, SearchScheme};

// N total elements in a node.
// B branching factor.
// B-1 actual elements in a node.
// REV: when true, nodes contain the largest elements of their first B (of B+1) children,
//      rather than the smallest elements of the last B children.
#[derive(Debug)]
pub struct BpTree<const B: usize, const N: usize, const REV: bool> {
    tree: Vec<BTreeNode<B, N>>,
    pub cnt: usize,
    offsets: Vec<usize>,
}

pub type BpTree16 = BpTree<16, 16, false>;
pub type BpTree15 = BpTree<15, 16, false>;
pub type BpTree16R = BpTree<16, 16, true>;

/// Helper functions for constructing searcher objects with the same generics.
impl<const B: usize, const N: usize, const REV: bool> BpTree<B, N, REV> {
    pub const fn search() -> BpTreeSearch<B, N, REV> {
        BpTreeSearch
    }
    pub const fn search_split() -> BpTreeSearchSplit<B, N, REV> {
        BpTreeSearchSplit
    }
    pub const fn search_batch<const P: usize>() -> BpTreeSearchBatch<B, N, REV, P> {
        BpTreeSearchBatch
    }
    pub const fn search_batch_prefetch<const P: usize>() -> BpTreeSearchBatchPrefetch<B, N, REV, P>
    {
        BpTreeSearchBatchPrefetch
    }
    pub const fn search_batch_ptr<const P: usize>() -> BpTreeSearchBatchPtr<B, N, REV, P> {
        BpTreeSearchBatchPtr
    }
    pub const fn search_batch_ptr2<const P: usize>() -> BpTreeSearchBatchPtr2<B, N, REV, P> {
        BpTreeSearchBatchPtr2
    }
    pub const fn search_batch_ptr3<const P: usize, const LAST: bool>(
    ) -> BpTreeSearchBatchPtr3<B, N, REV, P, LAST> {
        BpTreeSearchBatchPtr3
    }
    pub const fn search_batch_ptr3_full<const P: usize, const LAST: bool>(
    ) -> BpTreeSearchBatchPtr3Full<B, N, REV, P, LAST> {
        BpTreeSearchBatchPtr3Full
    }
    pub const fn search_batch_no_prefetch<const P: usize, const LAST: bool, const SKIP: usize>(
    ) -> BpTreeSearchBatchNoPrefetch<B, N, REV, P, LAST, SKIP> {
        BpTreeSearchBatchNoPrefetch
    }
    pub const fn search_interleave<const P: usize, const LAST: bool>(
    ) -> BpTreeSearchInterleave<B, N, REV, P, LAST> {
        BpTreeSearchInterleave
    }
}

impl<const B: usize, const N: usize, const REV: bool> BpTree<B, N, REV> {
    fn blocks(n: usize) -> usize {
        n.div_ceil(B)
    }
    fn prev_keys(n: usize) -> usize {
        Self::blocks(n).div_ceil(B + 1) * B
    }
    fn height(n: usize) -> usize {
        if n <= B {
            1
        } else {
            Self::height(Self::prev_keys(n)) + 1
        }
    }
    fn offset(mut n: usize, h: usize) -> usize {
        let mut k = 0;
        for _ in 0..h {
            k += Self::blocks(n);
            n = Self::prev_keys(n);
        }
        k
    }

    fn go_to(k: usize, j: usize) -> usize {
        k * (B + 1) + j + 1
    }

    fn node(&self, b: usize) -> &BTreeNode<B, N> {
        unsafe { &*self.tree.get_unchecked(b) }
    }

    fn get(&self, b: usize, i: usize) -> u32 {
        unsafe { *self.tree.get_unchecked(b).data.get_unchecked(i) }
    }

    pub fn new_fwd(vals: &[u32], full_array: bool) -> Self {
        let n = vals.len();
        let height = Self::height(n);
        let n_blocks = if full_array {
            let mut idx = 0;
            for _ in 0..height {
                idx = Self::go_to(idx, 0);
            }
            idx
        } else {
            Self::offset(n, height)
        };

        eprintln!("Allocating tree of {} blocks ..", n_blocks);
        let tree = vec![BTreeNode { data: [0; N] }; n_blocks];
        eprintln!("Allocating DONE");
        let mut bptree = Self {
            tree,
            cnt: 0,
            offsets: if full_array {
                let mut idx = 0;
                let mut v = (0..height)
                    .map(|_| {
                        let t = idx;
                        idx = Self::go_to(idx, 0);
                        t
                    })
                    .collect_vec();
                v.reverse();
                v
            } else {
                (1..=height)
                    .map(|h| n_blocks - Self::offset(n, h))
                    .collect()
            },
        };
        // eprintln!("Height: {}, n_blocks: {}", height, n_blocks);
        // eprintln!("FWD {:?}", bptree.offsets);

        for &v in vals {
            assert!(v <= MAX);
        }

        // offset of the layer containing original data.
        let o = bptree.offsets[0];
        // eprintln!("o: {}", o);

        // Copy the input values to their layer.
        for (i, &val) in vals.iter().enumerate() {
            bptree.tree[o + i / B].data[i % B] = val;
        }

        // Initialize layers; copied from Algorithmica.
        // https://en.algorithmica.org/hpc/data-structures/s-tree/#construction-1
        for h in 1..height {
            eprintln!("Starting layer {h} at offset {}", bptree.offsets[h]);
            for i in 0..B * (bptree.offsets[h - 1] - bptree.offsets[h]) {
                let mut k = i / B;
                let j = i % B;
                k = k * (B + 1) + j + 1;
                // compare to right of key
                // and then to the left
                for _l in 1..h {
                    k *= B + 1;
                }
                // eprintln!("Writing layer {h} offset {} + {}", bptree.offsets[h], i / B);
                let t = bptree.offsets[h] + i / B;
                if !REV {
                    if k * B < n {
                        bptree.tree[t].data[i % B] = bptree.tree[o + k].data[0];
                    } else {
                        for j in i % B..B {
                            bptree.tree[t].data[j] = MAX;
                        }
                        eprintln!("Breaking in layer {h} at index B*{t}+{}", i % B);
                        break;
                    }
                } else {
                    if k * B < n {
                        bptree.tree[t].data[i % B] = bptree.tree[o + k - 1].data[B - 1];
                    } else {
                        for j in i % B..B {
                            bptree.tree[t].data[j] = MAX;
                        }
                        eprintln!("Breaking in layer {h} at index B*{t}+{}", i % B);
                        break;
                    }
                }
            }
        }
        // eprintln!("FWD Tree after initialization:");
        // for (i, node) in bptree.tree.iter().enumerate() {
        //     eprintln!("{i:>2}: {node:?}");
        // }
        bptree
    }
}

impl<const B: usize, const N: usize, const REV: bool> SearchIndex for BpTree<B, N, REV> {
    fn new(vals: &[u32]) -> Self {
        let n = vals.len();
        let height = Self::height(n);
        let n_blocks = Self::offset(n, height);
        let mut bptree = Self {
            tree: vec![BTreeNode { data: [MAX; N] }; n_blocks],
            cnt: 0,
            offsets: (0..=height).map(|h| Self::offset(n, h)).collect(),
        };
        // eprintln!("REV {:?}", bptree.offsets);

        for &v in vals {
            assert!(v <= MAX);
        }

        // Copy the input values to the start.
        for (i, &val) in vals.iter().enumerate() {
            bptree.tree[i / B].data[i % B] = val;
        }
        // Initialize layers; copied from Algorithmica.
        // https://en.algorithmica.org/hpc/data-structures/s-tree/#construction-1
        for h in 1..height {
            for i in 0..B * (bptree.offsets[h + 1] - bptree.offsets[h]) {
                let mut k = i / B;
                let j = i % B;
                k = k * (B + 1) + j + 1;
                // compare to right of key
                // and then to the left
                for _l in 0..h - 1 {
                    k *= B + 1;
                }
                if !REV {
                    bptree.tree[bptree.offsets[h] + i / B].data[i % B] = if k * B < n {
                        bptree.tree[k].data[0]
                    } else {
                        MAX
                    };
                } else {
                    bptree.tree[bptree.offsets[h] + i / B].data[i % B] = if k * B < n {
                        bptree.tree[k - 1].data[B - 1]
                    } else {
                        MAX
                    };
                }
            }
        }
        // eprintln!("REV Tree after initialization:");
        // for (i, node) in bptree.tree.iter().enumerate() {
        //     eprintln!("{i:>2}: {node:?}");
        // }
        bptree.offsets.pop();
        bptree
    }
}

fn batched<const P: usize>(qs: &[u32], f: impl Fn(&[u32; P]) -> [u32; P]) -> Vec<u32> {
    let it = qs.array_chunks();
    assert!(
        it.remainder().is_empty(),
        "For now, batched queries cannot handle leftovers"
    );
    it.flat_map(f).collect()
}

pub struct BpTreeSearch<const B: usize, const N: usize, const REV: bool>;
impl<const B: usize, const N: usize, const REV: bool> SearchScheme for BpTreeSearch<B, N, REV> {
    type INDEX = BpTree<B, N, REV>;

    fn query_one(&self, index: &Self::INDEX, q: u32) -> u32 {
        assert!(!REV);
        let mut k = 0;
        for o in index.offsets[1..index.offsets.len()].into_iter().rev() {
            let jump_to = index.node(o + k).find(q);
            k = k * (B + 1) + jump_to;
        }

        let idx = index.node(k).find(q);
        index.get(k + idx / B, idx % B)
    }
}

pub struct BpTreeSearchSplit<const B: usize, const N: usize, const REV: bool>;
impl<const B: usize, const N: usize, const REV: bool> SearchScheme
    for BpTreeSearchSplit<B, N, REV>
{
    type INDEX = BpTree<B, N, REV>;

    fn query_one(&self, index: &Self::INDEX, q: u32) -> u32 {
        assert!(!REV);
        let mut k = 0;
        for o in index.offsets[1..index.offsets.len()].into_iter().rev() {
            let jump_to = index.node(o + k).find_split(q);
            k = k * (B + 1) + jump_to;
        }

        let idx = index.node(k).find(q);
        index.get(k + idx / B, idx % B)
    }
}

pub struct BpTreeSearchBatch<const B: usize, const N: usize, const REV: bool, const P: usize>;
impl<const B: usize, const N: usize, const REV: bool, const P: usize> SearchScheme
    for BpTreeSearchBatch<B, N, REV, P>
{
    type INDEX = BpTree<B, N, REV>;

    fn query(&self, index: &Self::INDEX, qs: &[u32]) -> Vec<u32> {
        batched(qs, |qb: &[u32; P]| {
            assert!(!REV);
            let mut k = [0; P];
            for o in index.offsets[1..index.offsets.len()].into_iter().rev() {
                for i in 0..P {
                    let jump_to = index.node(o + k[i]).find(qb[i]);
                    k[i] = k[i] * (B + 1) + jump_to;
                }
            }

            std::array::from_fn::<_, P, _>(|i| {
                let idx = index.node(k[i]).find(qb[i]);
                index.get(k[i] + idx / B, idx % B)
            })
        })
    }
}

pub struct BpTreeSearchBatchPrefetch<
    const B: usize,
    const N: usize,
    const REV: bool,
    const P: usize,
>;
impl<const B: usize, const N: usize, const REV: bool, const P: usize> SearchScheme
    for BpTreeSearchBatchPrefetch<B, N, REV, P>
{
    type INDEX = BpTree<B, N, REV>;

    fn query(&self, index: &Self::INDEX, qs: &[u32]) -> Vec<u32> {
        batched(qs, |qb: &[u32; P]| {
            assert!(!REV);
            let mut k = [0; P];
            let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));
            for h in (1..index.offsets.len()).rev() {
                let o = unsafe { index.offsets.get_unchecked(h) };
                let o2 = unsafe { index.offsets.get_unchecked(h - 1) };
                for i in 0..P {
                    let jump_to = index.node(o + k[i]).find_splat(q_simd[i]);
                    k[i] = k[i] * (B + 1) + jump_to;
                    prefetch_index(&index.tree, o2 + k[i]);
                }
            }

            std::array::from_fn(|i| {
                let idx = index.node(k[i]).find_splat(q_simd[i]);
                index.get(k[i] + idx / B, idx % B)
            })
        })
    }
}

pub struct BpTreeSearchBatchPtr<const B: usize, const N: usize, const REV: bool, const P: usize>;
impl<const B: usize, const N: usize, const REV: bool, const P: usize> SearchScheme
    for BpTreeSearchBatchPtr<B, N, REV, P>
{
    type INDEX = BpTree<B, N, REV>;

    fn query(&self, index: &Self::INDEX, qs: &[u32]) -> Vec<u32> {
        batched(qs, |qb: &[u32; P]| {
            assert!(!REV);
            let mut k = [0; P];
            let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

            let offsets = index
                .offsets
                .iter()
                .map(|o| unsafe { index.tree.as_ptr().add(*o) })
                .collect_vec();

            for h in (1..offsets.len()).rev() {
                let o = unsafe { *offsets.get_unchecked(h) };
                let o2 = unsafe { *offsets.get_unchecked(h - 1) };
                for i in 0..P {
                    let jump_to = unsafe { *o.add(k[i]) }.find_splat(q_simd[i]);
                    k[i] = k[i] * (B + 1) + jump_to;
                    prefetch_ptr(unsafe { o2.add(k[i]) });
                }
            }

            std::array::from_fn(|i| {
                let idx = index.node(k[i]).find_splat(q_simd[i]);
                index.get(k[i] + idx / B, idx % B)
            })
        })
    }
}

pub struct BpTreeSearchBatchPtr2<const B: usize, const N: usize, const REV: bool, const P: usize>;
impl<const B: usize, const N: usize, const REV: bool, const P: usize> SearchScheme
    for BpTreeSearchBatchPtr2<B, N, REV, P>
{
    type INDEX = BpTree<B, N, REV>;

    fn query(&self, index: &Self::INDEX, qs: &[u32]) -> Vec<u32> {
        batched(qs, |qb: &[u32; P]| {
            assert!(!REV);
            let mut k = [0; P];
            let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

            let offsets = index
                .offsets
                .iter()
                .map(|o| unsafe { index.tree.as_ptr().add(*o) })
                .collect_vec();

            for h in (1..offsets.len()).rev() {
                let o = unsafe { *offsets.get_unchecked(h) };
                let o2 = unsafe { *offsets.get_unchecked(h - 1) };
                for i in 0..P {
                    let jump_to = unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i]);
                    k[i] = k[i] * (B + 1) + jump_to * 64;
                    prefetch_ptr(unsafe { o2.byte_add(k[i]) });
                }
            }

            let o = unsafe { *offsets.get_unchecked(0) };
            std::array::from_fn(|i| {
                let mut idx = unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i]);
                if !REV && idx == B {
                    idx = N;
                }
                unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
            })
        })
    }
}

pub struct BpTreeSearchBatchPtr3<
    const B: usize,
    const N: usize,
    const REV: bool,
    const P: usize,
    const LAST: bool,
>;
impl<const B: usize, const N: usize, const REV: bool, const P: usize, const LAST: bool> SearchScheme
    for BpTreeSearchBatchPtr3<B, N, REV, P, LAST>
{
    type INDEX = BpTree<B, N, REV>;

    fn query(&self, index: &Self::INDEX, qs: &[u32]) -> Vec<u32> {
        batched(qs, |qb: &[u32; P]| {
            let mut k = [0; P];
            let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

            let offsets = index
                .offsets
                .iter()
                .map(|o| unsafe { index.tree.as_ptr().add(*o) })
                .collect_vec();

            for h in (1..offsets.len()).rev() {
                let o = unsafe { *offsets.get_unchecked(h) };
                let o2 = unsafe { *offsets.get_unchecked(h - 1) };
                for i in 0..P {
                    let jump_to = if !LAST {
                        unsafe { *o.byte_add(k[i]) }.find_splat64(q_simd[i])
                    } else {
                        unsafe { *o.byte_add(k[i]) }.find_splat64_last(q_simd[i])
                    };
                    k[i] = k[i] * (B + 1) + jump_to;
                    prefetch_ptr(unsafe { o2.byte_add(k[i]) });
                }
            }

            let o = unsafe { *offsets.get_unchecked(0) };
            std::array::from_fn(|i| {
                let mut idx = if !LAST {
                    unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i])
                } else {
                    unsafe { *o.byte_add(k[i]) }.find_splat_last(q_simd[i])
                };
                if !REV && idx == B {
                    idx = N;
                }
                unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
            })
        })
    }
}

pub struct BpTreeSearchBatchPtr3Full<
    const B: usize,
    const N: usize,
    const REV: bool,
    const P: usize,
    const LAST: bool,
>;
impl<const B: usize, const N: usize, const REV: bool, const P: usize, const LAST: bool> SearchScheme
    for BpTreeSearchBatchPtr3Full<B, N, REV, P, LAST>
{
    type INDEX = BpTree<B, N, REV>;

    fn query(&self, index: &Self::INDEX, qs: &[u32]) -> Vec<u32> {
        batched(qs, |qb: &[u32; P]| {
            let mut k = [0; P];
            let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

            let o = index.tree.as_ptr();

            for _h in (1..index.offsets.len()).rev() {
                for i in 0..P {
                    let jump_to = if !LAST {
                        unsafe { *o.byte_add(k[i]) }.find_splat64(q_simd[i])
                    } else {
                        unsafe { *o.byte_add(k[i]) }.find_splat64_last(q_simd[i])
                    };
                    k[i] = k[i] * (B + 1) + jump_to + 64;
                    prefetch_ptr(unsafe { o.byte_add(k[i]) });
                }
            }

            std::array::from_fn(|i| {
                let mut idx = if !LAST {
                    unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i])
                } else {
                    unsafe { *o.byte_add(k[i]) }.find_splat_last(q_simd[i])
                };
                if !REV && idx == B {
                    idx = N;
                }
                unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
            })
        })
    }
}

pub struct BpTreeSearchBatchNoPrefetch<
    const B: usize,
    const N: usize,
    const REV: bool,
    const P: usize,
    const LAST: bool,
    const SKIP: usize,
>;
impl<
        const B: usize,
        const N: usize,
        const REV: bool,
        const P: usize,
        const LAST: bool,
        const SKIP: usize,
    > SearchScheme for BpTreeSearchBatchNoPrefetch<B, N, REV, P, LAST, SKIP>
{
    type INDEX = BpTree<B, N, REV>;

    fn query(&self, index: &Self::INDEX, qs: &[u32]) -> Vec<u32> {
        batched(qs, |qb: &[u32; P]| {
            let mut k = [0; P];
            let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

            let offsets = index
                .offsets
                .iter()
                .map(|o| unsafe { index.tree.as_ptr().add(*o) })
                .collect_vec();

            let lim = offsets.len().saturating_sub(SKIP).max(1);

            for h in (lim..offsets.len()).rev() {
                let o = unsafe { *offsets.get_unchecked(h) };
                // let o2 = unsafe { *offsets.get_unchecked(h - 1) };
                for i in 0..P {
                    let jump_to = if !LAST {
                        unsafe { *o.byte_add(k[i]) }.find_splat64(q_simd[i])
                    } else {
                        unsafe { *o.byte_add(k[i]) }.find_splat64_last(q_simd[i])
                    };
                    k[i] = k[i] * (B + 1) + jump_to;
                    // prefetch_ptr(unsafe { o2.byte_add(k[i]) });
                }
            }

            for h in (1..lim).rev() {
                let o = unsafe { *offsets.get_unchecked(h) };
                let o2 = unsafe { *offsets.get_unchecked(h - 1) };
                for i in 0..P {
                    let jump_to = if !LAST {
                        unsafe { *o.byte_add(k[i]) }.find_splat64(q_simd[i])
                    } else {
                        unsafe { *o.byte_add(k[i]) }.find_splat64_last(q_simd[i])
                    };
                    k[i] = k[i] * (B + 1) + jump_to;
                    prefetch_ptr(unsafe { o2.byte_add(k[i]) });
                }
            }

            let o = unsafe { *offsets.get_unchecked(0) };
            std::array::from_fn(|i| {
                let mut idx = if !LAST {
                    unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i])
                } else {
                    unsafe { *o.byte_add(k[i]) }.find_splat_last(q_simd[i])
                };
                if !REV && idx == B {
                    idx = N;
                }
                unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
            })
        })
    }
}

pub struct BpTreeSearchInterleave<
    const B: usize,
    const N: usize,
    const REV: bool,
    const P: usize,
    const LAST: bool,
>;
impl<const B: usize, const N: usize, const REV: bool, const P: usize, const LAST: bool> SearchScheme
    for BpTreeSearchInterleave<B, N, REV, P, LAST>
{
    type INDEX = BpTree<B, N, REV>;

    fn query(&self, index: &Self::INDEX, qs: &[u32]) -> Vec<u32> {
        if index.offsets.len() % 2 != 0 {
            return vec![];
        }
        assert!(index.offsets.len() % 2 == 0);

        let offsets = index
            .offsets
            .iter()
            .map(|o| unsafe { index.tree.as_ptr().add(*o) })
            .collect_vec();

        let mut k1 = &mut [0; P];
        let mut k2 = &mut [0; P];

        // temporary: assert h is even

        // 1. setup first P queries
        // 2. do 2nd half of first P queries, and first half of next; repeat
        // 3. last half of last P queries.

        let hh = index.offsets.len() / 2;

        let mut chunks = qs.array_chunks::<P>();
        assert!(
            chunks.remainder().is_empty(),
            "Interleave does not process trailing bits"
        );
        let mut out = Vec::with_capacity(qs.len());

        let c1 = chunks.next().unwrap();
        let mut q_simd1 = &mut [Simd::<u32, 8>::splat(0); P];
        let mut q_simd2 = &mut [Simd::<u32, 8>::splat(0); P];
        *q_simd1 = c1.map(|q| Simd::<u32, 8>::splat(q));

        let hs_first = (hh..offsets.len()).rev();
        let hs_second = (1..hh).rev();

        // 1
        for h1 in hs_first.clone() {
            // eprintln!("h1: {}", h1);
            let o = unsafe { *offsets.get_unchecked(h1) };
            let o2 = unsafe { *offsets.get_unchecked(h1 - 1) };
            for i in 0..P {
                let jump_to = if !LAST {
                    unsafe { *o.byte_add(k1[i]) }.find_splat64(q_simd1[i])
                } else {
                    unsafe { *o.byte_add(k1[i]) }.find_splat64_last(q_simd1[i])
                };
                k1[i] = k1[i] * (B + 1) + jump_to;
                prefetch_ptr(unsafe { o2.byte_add(k1[i]) });
            }
        }

        // 2
        for c1 in chunks {
            // swap
            (q_simd1, q_simd2) = (q_simd2, q_simd1);
            (k1, k2) = (k2, k1);
            *q_simd1 = c1.map(|q| Simd::<u32, 8>::splat(q));
            k1.fill(0);

            // First hh levels of c1.
            // Last hh levels of c2, with the last level special.

            for (h1, h2) in zip(hs_first.clone(), hs_second.clone()) {
                // eprintln!("h1: {}, h2: {}", h1, h2);
                let o1 = unsafe { *offsets.get_unchecked(h1) };
                let o12 = unsafe { *offsets.get_unchecked(h1 - 1) };
                let o2 = unsafe { *offsets.get_unchecked(h2) };
                let o22 = unsafe { *offsets.get_unchecked(h2 - 1) };
                for i in 0..P {
                    // 1
                    let jump_to = if !LAST {
                        unsafe { *o1.byte_add(k1[i]) }.find_splat64(q_simd1[i])
                    } else {
                        unsafe { *o1.byte_add(k1[i]) }.find_splat64_last(q_simd1[i])
                    };
                    k1[i] = k1[i] * (B + 1) + jump_to;
                    prefetch_ptr(unsafe { o12.byte_add(k1[i]) });

                    // 2
                    let jump_to = if !LAST {
                        unsafe { *o2.byte_add(k2[i]) }.find_splat64(q_simd2[i])
                    } else {
                        unsafe { *o2.byte_add(k2[i]) }.find_splat64_last(q_simd2[i])
                    };
                    k2[i] = k2[i] * (B + 1) + jump_to;
                    prefetch_ptr(unsafe { o22.byte_add(k2[i]) });
                }
            }

            // eprintln!("h1: {}, h2: {}", hh, 0);

            let o1 = unsafe { *offsets.get_unchecked(hh) };
            let o12 = unsafe { *offsets.get_unchecked(hh - 1) };

            // last iteration is special, where h2 = 0.
            let o = unsafe { *offsets.get_unchecked(0) };
            let ans: [u32; P] = std::array::from_fn(|i| {
                let jump_to = if !LAST {
                    unsafe { *o1.byte_add(k1[i]) }.find_splat64(q_simd1[i])
                } else {
                    unsafe { *o1.byte_add(k1[i]) }.find_splat64_last(q_simd1[i])
                };
                k1[i] = k1[i] * (B + 1) + jump_to;
                prefetch_ptr(unsafe { o12.byte_add(k1[i]) });

                let mut idx = if !LAST {
                    unsafe { *o.byte_add(k2[i]) }.find_splat(q_simd2[i])
                } else {
                    unsafe { *o.byte_add(k2[i]) }.find_splat_last(q_simd2[i])
                };
                if !REV && idx == B {
                    idx = N;
                }
                unsafe { (o.byte_add(k2[i]) as *const u32).add(idx).read() }
            });
            out.extend_from_slice(&ans);
        }

        // 3
        for h2 in hs_second {
            // eprintln!("h2: {}", h2);
            let o = unsafe { *offsets.get_unchecked(h2) };
            let o2 = unsafe { *offsets.get_unchecked(h2 - 1) };
            for i in 0..P {
                let jump_to = if !LAST {
                    unsafe { *o.byte_add(k1[i]) }.find_splat64(q_simd1[i])
                } else {
                    unsafe { *o.byte_add(k1[i]) }.find_splat64_last(q_simd1[i])
                };
                k1[i] = k1[i] * (B + 1) + jump_to;
                prefetch_ptr(unsafe { o2.byte_add(k1[i]) });
            }
        }
        // h=0
        let o = unsafe { *offsets.get_unchecked(0) };
        let ans: [u32; P] = std::array::from_fn(|i| {
            let mut idx = if !LAST {
                unsafe { *o.byte_add(k1[i]) }.find_splat(q_simd1[i])
            } else {
                unsafe { *o.byte_add(k1[i]) }.find_splat_last(q_simd1[i])
            };
            if !REV && idx == B {
                idx = N;
            }
            unsafe { (o.byte_add(k1[i]) as *const u32).add(idx).read() }
        });
        out.extend_from_slice(&ans);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        binary_search::{BinarySearch, SortedVec},
        SearchIndex,
    };

    #[test]
    fn test_b_tree_k_2() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let bptree = BpTree::<2, 2, false>::new(&vals);
        println!("{:?}", bptree);
    }

    #[test]
    fn test_b_tree_k_3() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = BpTree::<3, 3, false>::new(&vals);
        println!("{:?}", computed_out);
    }

    #[test]
    fn test_bptree_search_bottom_layer() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 452;
        let bptree = BpTree::<16, 16, false>::new(&vals);
        let bptree_res = bptree.query_one(q, &BpTreeSearch);

        let bin_res = SortedVec::new(&vals).query_one(q, &BinarySearch);
        println!("{bptree_res}, {bin_res}");
        assert!(bptree_res == bin_res);
    }

    #[test]
    fn test_bptree_search_top_node() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 289;
        let bptree = BpTree::<16, 16, false>::new(&vals);
        let bptree_res = bptree.query_one(q, &BpTreeSearch);

        let bin_res = SortedVec::new(&vals).query_one(q, &BinarySearch);
        println!("{bptree_res}, {bin_res}");
        assert!(bptree_res == bin_res);
    }

    #[test]
    fn test_simd_cmp() {
        let mut vals: Vec<u32> = (1..16).collect();
        vals.push(MAX);
        let bptree = BpTree::<16, 16, false>::new(&vals);
        let idx = bptree.tree[0].find(1);
        println!("{}", idx);
        assert_eq!(idx, 0);
    }
}
