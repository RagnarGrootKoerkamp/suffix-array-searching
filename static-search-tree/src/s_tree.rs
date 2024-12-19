use std::{fmt::Debug, iter::zip, simd::Simd};

use itertools::Itertools;

use crate::node::{BTreeNode, MAX};
use crate::{prefetch_index, prefetch_ptr, SearchIndex};

/// N total elements in a node.
/// B branching factor.
/// B-1 actual elements in a node.
#[derive(Debug)]
pub struct STree<const B: usize, const N: usize> {
    tree: Vec<BTreeNode<N>>,
    offsets: Vec<usize>,
}

impl<const B: usize, const N: usize> SearchIndex for STree<B, N> {
    fn new(vals: &[u32]) -> Self {
        Self::new_params(vals, false, false, false)
    }
}

pub type STree16 = STree<16, 16>;
pub type STree15 = STree<15, 16>;

impl<const B: usize, const N: usize> STree<B, N> {
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

    fn node(&self, b: usize) -> &BTreeNode<N> {
        unsafe { &*self.tree.get_unchecked(b) }
    }

    fn get(&self, b: usize, i: usize) -> u32 {
        unsafe { *self.tree.get_unchecked(b).data.get_unchecked(i) }
    }

    pub fn new_params(vals: &[u32], fwd: bool, rev: bool, full_array: bool) -> Self {
        if full_array {
            assert!(fwd, "Full array only makes sense in forward layout.");
        }
        if fwd {
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
                    if !rev {
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
            // for o in &bptree.offsets {
            //     eprintln!("Offset: {}", o);
            // }
            // for node in &bptree.tree {
            //     eprintln!("{:?}", node);
            // }
            bptree
        } else {
            let n = vals.len();
            let height = Self::height(n);
            let n_blocks = Self::offset(n, height);
            let mut bptree = Self {
                tree: vec![BTreeNode { data: [MAX; N] }; n_blocks],
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
                    if !rev {
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
            bptree.offsets.pop();
            bptree
        }
    }

    fn search_with_find_impl(&self, q: u32, find: impl Fn(&BTreeNode<N>, u32) -> usize) -> u32 {
        let mut k = 0;
        for o in self.offsets[1..self.offsets.len()].into_iter().rev() {
            let jump_to = find(self.node(o + k), q);
            k = k * (B + 1) + jump_to;
        }

        let o = self.offsets[0];
        let mut idx = find(self.node(o + k), q);
        if idx == B {
            idx = N;
        }
        self.get(o + k + idx / N, idx % N)
    }

    pub const fn search_with_find(
        find: impl Fn(&BTreeNode<N>, u32) -> usize + Copy,
    ) -> impl Fn(&STree<B, N>, u32) -> u32 {
        move |index, q| index.search_with_find_impl(q, find)
    }

    pub fn search(&self, q: u32) -> u32 {
        let mut k = 0;
        for o in self.offsets[1..self.offsets.len()].into_iter().rev() {
            let jump_to = self.node(o + k).find(q);
            k = k * (B + 1) + jump_to;
        }

        let o = self.offsets[0];
        let mut idx = self.node(o + k).find(q);
        if idx == B {
            idx = N;
        }
        self.get(o + k + idx / N, idx % N)
    }

    pub fn batch<'a, const P: usize>(&self, qb: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        for o in self.offsets[1..self.offsets.len()].into_iter().rev() {
            for i in 0..P {
                let jump_to = self.node(o + k[i]).find(qb[i]);
                k[i] = k[i] * (B + 1) + jump_to;
            }
        }

        let o = self.offsets[0];
        std::array::from_fn::<_, P, _>(|i| {
            let mut idx = self.node(o + k[i]).find(qb[i]);
            if idx == B {
                idx = N;
            }
            self.get(o + k[i] + idx / N, idx % N)
        })
    }

    pub fn batch_prefetch<const P: usize>(&self, qb: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));
        for h in (1..self.offsets.len()).rev() {
            let o = unsafe { self.offsets.get_unchecked(h) };
            let o2 = unsafe { self.offsets.get_unchecked(h - 1) };
            for i in 0..P {
                let jump_to = self.node(o + k[i]).find_splat(q_simd[i]);
                k[i] = k[i] * (B + 1) + jump_to;
                prefetch_index(&self.tree, o2 + k[i]);
            }
        }

        let o = self.offsets[0];
        std::array::from_fn(|i| {
            let mut idx = self.node(o + k[i]).find(qb[i]);
            if idx == B {
                idx = N;
            }
            self.get(o + k[i] + idx / N, idx % N)
        })
    }

    pub fn batch_ptr<const P: usize>(&self, qb: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
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

        let o = self.offsets[0];
        std::array::from_fn(|i| {
            let mut idx = self.node(o + k[i]).find(qb[i]);
            if idx == B {
                idx = N;
            }
            self.get(o + k[i] + idx / N, idx % N)
        })
    }

    pub fn batch_ptr2<const P: usize>(&self, qb: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
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
            if idx == B {
                idx = N;
            }
            unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
        })
    }

    pub fn batch_ptr3<const P: usize, const LAST: bool>(&self, qb: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
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
            if idx == B {
                idx = N;
            }
            unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
        })
    }

    pub fn batch_ptr3_full<const P: usize, const LAST: bool>(&self, qb: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

        let o = self.tree.as_ptr();

        for _h in (1..self.offsets.len()).rev() {
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
            if idx == B {
                idx = N;
            }
            unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
        })
    }

    pub fn batch_no_prefetch<const P: usize, const LAST: bool, const SKIP: usize>(
        &self,
        qb: &[u32; P],
    ) -> [u32; P] {
        let mut k = [0; P];
        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
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
            if idx == B {
                idx = N;
            }
            unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
        })
    }

    pub fn batch_interleave<const P: usize, const LAST: bool>(&self, qs: &[u32]) -> Vec<u32> {
        if self.offsets.len() % 2 != 0 {
            return vec![];
        }
        assert!(self.offsets.len() % 2 == 0);

        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
            .collect_vec();

        let mut k1 = &mut [0; P];
        let mut k2 = &mut [0; P];

        // temporary: assert h is even

        // 1. setup first P queries
        // 2. do 2nd half of first P queries, and first half of next; repeat
        // 3. last half of last P queries.

        let hh = self.offsets.len() / 2;

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
                if idx == B {
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
            if idx == B {
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
    use crate::{binary_search::SortedVec, SearchIndex};

    #[test]
    fn test_b_tree_k_2() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let bptree = STree::<2, 2>::new_params(&vals, false, false, false);
        println!("{:?}", bptree);
    }

    #[test]
    fn test_b_tree_k_3() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = STree::<3, 3>::new_params(&vals, false, false, false);
        println!("{:?}", computed_out);
    }

    #[test]
    fn test_bptree_search_bottom_layer() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 452;
        let bptree = STree::<16, 16>::new_params(&vals, false, false, false);
        let bptree_res = bptree.search(q);

        let bin_res = SortedVec::new(&vals).binary_search(q);
        println!("{bptree_res}, {bin_res}");
        assert!(bptree_res == bin_res);
    }

    #[test]
    fn test_bptree_search_top_node() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 289;
        let bptree = STree::<16, 16>::new_params(&vals, false, false, false);
        let bptree_res = bptree.search(q);

        let bin_res = SortedVec::new(&vals).binary_search(q);
        println!("{bptree_res}, {bin_res}");
        assert!(bptree_res == bin_res);
    }

    #[test]
    fn test_simd_cmp() {
        let mut vals: Vec<u32> = (1..16).collect();
        vals.push(MAX);
        let bptree = STree::<16, 16>::new_params(&vals, false, false, false);
        let idx = bptree.tree[0].find(1);
        println!("{}", idx);
        assert_eq!(idx, 0);
    }
}
