use std::{fmt::Debug, simd::Simd};

use itertools::Itertools;

use crate::{
    btree::{BTreeNode, MAX},
    prefetch_index, prefetch_ptr,
};

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

    pub fn new_fwd(vals: Vec<u32>, full_array: bool) -> Self {
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

        for &v in &vals {
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

    pub fn new(vals: Vec<u32>) -> Self {
        let n = vals.len();
        let height = Self::height(n);
        let n_blocks = Self::offset(n, height);
        let mut bptree = Self {
            tree: vec![BTreeNode { data: [MAX; N] }; n_blocks],
            cnt: 0,
            offsets: (0..=height).map(|h| Self::offset(n, h)).collect(),
        };
        // eprintln!("REV {:?}", bptree.offsets);

        for &v in &vals {
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

    fn node(&self, b: usize) -> &BTreeNode<B, N> {
        unsafe { &*self.tree.get_unchecked(b) }
    }

    fn get(&self, b: usize, i: usize) -> u32 {
        unsafe { *self.tree.get_unchecked(b).data.get_unchecked(i) }
    }

    pub fn search(&mut self, q: u32) -> u32 {
        assert!(!REV);
        let mut k = 0;
        for o in self.offsets[1..self.offsets.len()].into_iter().rev() {
            let jump_to = self.node(o + k).find(q);
            k = k * (B + 1) + jump_to;
        }

        let index = self.node(k).find(q);
        self.get(k + index / B, index % B)
    }

    pub fn search_split(&mut self, q: u32) -> u32 {
        assert!(!REV);
        let mut k = 0;
        for o in self.offsets[1..self.offsets.len()].into_iter().rev() {
            let jump_to = self.node(o + k).find_split(q);
            k = k * (B + 1) + jump_to;
        }

        let index = self.node(k).find(q);
        self.get(k + index / B, index % B)
    }

    pub fn batch<const P: usize>(&mut self, q: &[u32; P]) -> [u32; P] {
        assert!(!REV);
        let mut k = [0; P];
        for o in self.offsets[1..self.offsets.len()].into_iter().rev() {
            for i in 0..P {
                let jump_to = self.node(o + k[i]).find(q[i]);
                k[i] = k[i] * (B + 1) + jump_to;
            }
        }

        std::array::from_fn(|i| {
            let index = self.node(k[i]).find(q[i]);
            self.get(k[i] + index / B, index % B)
        })
    }

    pub fn batch_prefetch<const P: usize>(&mut self, q: &[u32; P]) -> [u32; P] {
        assert!(!REV);
        let mut k = [0; P];
        let q_simd = q.map(|q| Simd::<u32, 8>::splat(q));
        for h in (1..self.offsets.len()).rev() {
            let o = unsafe { self.offsets.get_unchecked(h) };
            let o2 = unsafe { self.offsets.get_unchecked(h - 1) };
            for i in 0..P {
                let jump_to = self.node(o + k[i]).find_splat(q_simd[i]);
                k[i] = k[i] * (B + 1) + jump_to;
                prefetch_index(&self.tree, o2 + k[i]);
            }
        }

        std::array::from_fn(|i| {
            let index = self.node(k[i]).find_splat(q_simd[i]);
            self.get(k[i] + index / B, index % B)
        })
    }

    pub fn batch_ptr<const P: usize>(&mut self, q: &[u32; P]) -> [u32; P] {
        assert!(!REV);
        let mut k = [0; P];
        let q_simd = q.map(|q| Simd::<u32, 8>::splat(q));

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

        std::array::from_fn(|i| {
            let index = self.node(k[i]).find_splat(q_simd[i]);
            self.get(k[i] + index / B, index % B)
        })
    }

    pub fn batch_ptr2<const P: usize>(&mut self, q: &[u32; P]) -> [u32; P] {
        assert!(!REV);
        let mut k = [0; P];
        let q_simd = q.map(|q| Simd::<u32, 8>::splat(q));

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
            let mut index = unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i]);
            if !REV && index == B {
                index = N;
            }
            unsafe { (o.byte_add(k[i]) as *const u32).add(index).read() }
        })
    }

    pub fn batch_ptr3<const P: usize, const LAST: bool>(&mut self, q: &[u32; P]) -> [u32; P] {
        self.batch_ptr3_par::<P, LAST>(q)
    }

    pub fn batch_ptr3_par<const P: usize, const LAST: bool>(&self, q: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        let q_simd = q.map(|q| Simd::<u32, 8>::splat(q));

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
            let mut index = if !LAST {
                unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i])
            } else {
                unsafe { *o.byte_add(k[i]) }.find_splat_last(q_simd[i])
            };
            if !REV && index == B {
                index = N;
            }
            unsafe { (o.byte_add(k[i]) as *const u32).add(index).read() }
        })
    }

    pub fn batch_ptr3_full<const P: usize, const LAST: bool>(&mut self, q: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        let q_simd = q.map(|q| Simd::<u32, 8>::splat(q));

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
            let mut index = if !LAST {
                unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i])
            } else {
                unsafe { *o.byte_add(k[i]) }.find_splat_last(q_simd[i])
            };
            if !REV && index == B {
                index = N;
            }
            unsafe { (o.byte_add(k[i]) as *const u32).add(index).read() }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments_sorted_arrays::BinarySearch;

    #[test]
    fn test_b_tree_k_2() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let bptree = BpTree::<2, 2, false>::new(vals);
        println!("{:?}", bptree);
    }

    #[test]
    fn test_b_tree_k_3() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = BpTree::<3, 3, false>::new(vals);
        println!("{:?}", computed_out);
    }

    #[test]
    fn test_bptree_search_bottom_layer() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 452;
        let mut bptree = BpTree::<16, 16, false>::new(vals.clone());
        let bptree_res = bptree.search(q);

        let binsearch_res = BinarySearch::new(vals).search(q);
        println!("{bptree_res}, {binsearch_res}");
        assert!(bptree_res == binsearch_res);
    }

    #[test]
    fn test_bptree_search_top_node() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 289;
        let mut bptree = BpTree::<16, 16, false>::new(vals.clone());
        let bptree_res = bptree.search(q);

        let binsearch_res = BinarySearch::new(vals).search(q);
        println!("{bptree_res}, {binsearch_res}");
        assert!(bptree_res == binsearch_res);
    }

    #[test]
    fn test_simd_cmp() {
        let mut vals: Vec<u32> = (1..16).collect();
        vals.push(MAX);
        let bptree = BpTree::<16, 16, false>::new(vals);
        let idx = bptree.tree[0].find(1);
        println!("{}", idx);
        assert_eq!(idx, 0);
    }
}
