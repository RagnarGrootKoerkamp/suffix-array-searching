use std::{fmt::Debug, simd::Simd};

use itertools::Itertools;

use crate::{
    btree::{BTreeNode, MAX},
    prefetch_index, prefetch_ptr,
};

// N total elements in a node.
// B branching factor.
// B-1 actual elements in a node.
#[derive(Debug)]
pub struct BpTree<const B: usize, const N: usize> {
    tree: Vec<BTreeNode<B, N>>,
    pub cnt: usize,
    offsets: Vec<usize>,
}

pub type BpTree16 = BpTree<16, 16>;
pub type BpTree15 = BpTree<15, 16>;

impl<const B: usize, const N: usize> BpTree<B, N> {
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

    pub fn new(vals: Vec<u32>) -> Self {
        let n = vals.len();
        let height = Self::height(n);
        let n_blocks = Self::offset(n, height);
        let mut bptree = Self {
            tree: vec![BTreeNode { data: [MAX; N] }; n_blocks],
            cnt: 0,
            offsets: (0..=height).map(|h| Self::offset(n, h)).collect(),
        };

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
                k = k * (B + 1) + j + 1; // compare to right of key
                                         // and then to the left
                for _l in 0..h - 1 {
                    k *= B + 1;
                }
                bptree.tree[bptree.offsets[h] + i / B].data[i % B] = if k * B < n {
                    bptree.tree[k].data[0]
                } else {
                    MAX
                };
            }
        }
        bptree
    }

    fn node(&self, b: usize) -> &BTreeNode<B, N> {
        unsafe { &*self.tree.get_unchecked(b) }
    }

    fn get(&self, b: usize, i: usize) -> u32 {
        unsafe { *self.tree.get_unchecked(b).data.get_unchecked(i) }
    }

    pub fn search(&mut self, q: u32) -> u32 {
        let mut k = 0;
        for o in self.offsets[1..self.offsets.len() - 1].into_iter().rev() {
            let jump_to = self.node(o + k).find(q);
            k = k * (B + 1) + jump_to;
        }

        let index = self.node(k).find(q);
        self.get(k + index / B, index % B)
    }

    pub fn search_split(&mut self, q: u32) -> u32 {
        let mut k = 0;
        for o in self.offsets[1..self.offsets.len() - 1].into_iter().rev() {
            let jump_to = self.node(o + k).find_split(q);
            k = k * (B + 1) + jump_to;
        }

        let index = self.node(k).find(q);
        self.get(k + index / B, index % B)
    }

    pub fn batch<const P: usize>(&mut self, q: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        for o in self.offsets[1..self.offsets.len() - 1].into_iter().rev() {
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
        let mut k = [0; P];
        let q_simd = q.map(|q| Simd::<u32, 8>::splat(q));
        for h in (1..self.offsets.len() - 1).rev() {
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
        let mut k = [0; P];
        let q_simd = q.map(|q| Simd::<u32, 8>::splat(q));

        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
            .collect_vec();

        for h in (1..offsets.len() - 1).rev() {
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
        let mut k = [0; P];
        let q_simd = q.map(|q| Simd::<u32, 8>::splat(q));

        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
            .collect_vec();

        for h in (1..offsets.len() - 1).rev() {
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
            if index == B {
                index = N;
            }
            unsafe { (o.byte_add(k[i]) as *const u32).add(index).read() }
        })
    }

    pub fn batch_ptr3<const P: usize>(&mut self, q: &[u32; P]) -> [u32; P] {
        let mut k = [0; P];
        let q_simd = q.map(|q| Simd::<u32, 8>::splat(q));

        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
            .collect_vec();

        for h in (1..offsets.len() - 1).rev() {
            let o = unsafe { *offsets.get_unchecked(h) };
            let o2 = unsafe { *offsets.get_unchecked(h - 1) };
            for i in 0..P {
                let jump_to = unsafe { *o.byte_add(k[i]) }.find_splat64(q_simd[i]);
                k[i] = k[i] * (B + 1) + jump_to;
                prefetch_ptr(unsafe { o2.byte_add(k[i]) });
            }
        }

        let o = unsafe { *offsets.get_unchecked(0) };
        std::array::from_fn(|i| {
            let mut index = unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i]);
            if index == B {
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
        let bptree = BpTree::<2, 2>::new(vals);
        println!("{:?}", bptree);
    }

    #[test]
    fn test_b_tree_k_3() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = BpTree::<3, 3>::new(vals);
        println!("{:?}", computed_out);
    }

    #[test]
    fn test_bptree_search_bottom_layer() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 452;
        let mut bptree = BpTree::<16, 16>::new(vals.clone());
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
        let mut bptree = BpTree::<16, 16>::new(vals.clone());
        let bptree_res = bptree.search(q);

        let binsearch_res = BinarySearch::new(vals).search(q);
        println!("{bptree_res}, {binsearch_res}");
        assert!(bptree_res == binsearch_res);
    }

    #[test]
    fn test_simd_cmp() {
        let mut vals: Vec<u32> = (1..16).collect();
        vals.push(MAX);
        let bptree = BpTree::<16, 16>::new(vals);
        let idx = bptree.tree[0].find(1);
        println!("{}", idx);
        assert_eq!(idx, 0);
    }
}
