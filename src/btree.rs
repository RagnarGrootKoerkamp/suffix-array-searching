use std::arch::x86_64::{_mm256_movemask_epi8, _mm256_packs_epi32};
use std::fmt::Debug;
use std::simd::prelude::*;

pub(super) const MAX: u32 = i32::MAX as u32;

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct BTreeNode<const B: usize, const N: usize> {
    pub(super) data: [u32; N],
}

#[derive(Debug)]
pub struct BTree<const B: usize, const N: usize> {
    tree: Vec<BTreeNode<B, N>>,
    pub cnt: usize,
}

pub type BTree16 = BTree<16, 16>;

impl<const B: usize, const N: usize> Default for BTreeNode<B, N> {
    fn default() -> BTreeNode<B, N> {
        BTreeNode { data: [0; N] }
    }
}

impl<const B: usize, const N: usize> BTreeNode<B, N> {
    pub fn find(&self, q: u32) -> usize {
        self.find_popcnt(q)
    }

    pub fn find_ctz(&self, q: u32) -> usize {
        let data_simd: Simd<u32, 16> = Simd::from_slice(&self.data[0..N]);
        let q_simd = Simd::splat(q);
        let mask = q_simd.simd_le(data_simd);
        mask.first_set().unwrap_or(B)
    }

    /// Return the index of the first element >=q.
    /// Assumes that all elements fit in an i32, since SIMD doesn't have
    /// unsigned comparisons.
    pub fn find_popcnt(&self, q: u32) -> usize {
        let low: Simd<u32, 8> = Simd::from_slice(&self.data[0..N / 2]);
        let high: Simd<u32, 8> = Simd::from_slice(&self.data[N / 2..N]);
        let q_simd = Simd::<_, 8>::splat(q);
        // Merge the two masks, and convert to a single shuffled(!) mask.
        // But that's OK since popcount doesn't care about order.
        // TODO: Can we do this using portable SIMD?
        unsafe {
            let q_simd: Simd<i32, 8> = t(q_simd);
            let mask_low = q_simd.simd_gt(t(low));
            let mask_high = q_simd.simd_gt(t(high));
            use std::mem::transmute as t;
            let merged = _mm256_packs_epi32(t(mask_low), t(mask_high));
            let mask = _mm256_movemask_epi8(t(merged));
            mask.count_ones() as usize / 2
        }
    }

    pub fn find_splat(&self, q_simd: Simd<u32, 8>) -> usize {
        let low: Simd<u32, 8> = Simd::from_slice(&self.data[0..N / 2]);
        let high: Simd<u32, 8> = Simd::from_slice(&self.data[N / 2..N]);
        unsafe {
            let q_simd: Simd<i32, 8> = t(q_simd);
            let mask_low = q_simd.simd_gt(t(low));
            let mask_high = q_simd.simd_gt(t(high));
            use std::mem::transmute as t;
            let merged = _mm256_packs_epi32(t(mask_low), t(mask_high));
            let mask = _mm256_movemask_epi8(t(merged));
            mask.count_ones() as usize / 2
        }
    }
    /// Return the index of the first element >=q.
    /// This first does a single comparison to choose the left or right half of the array,
    /// and then uses SIMD on that half.
    /// This may reduce the pressure on SIMD registers.
    pub fn find_split(&self, q: u32) -> usize {
        let idx;
        if q <= self.data[B / 2] {
            idx = 0;
        } else {
            idx = B / 2;
        }
        let half_simd = Simd::<u32, 8>::from_slice(&self.data[idx..idx + B / 2]);
        let q_simd = Simd::splat(q);
        let mask = q_simd.simd_le(half_simd);
        idx + mask.first_set().unwrap_or(8)
    }
}

impl<const B: usize, const N: usize> BTree<B, N> {
    pub fn new(vals: Vec<u32>) -> Self {
        // always have at least one node
        let n_blocks = vals.len().div_ceil(B);
        let mut btree = Self {
            tree: vec![BTreeNode::default(); n_blocks],
            cnt: 0,
        };
        let mut i: usize = 0;
        let k = 0;
        // SIMD operations fail for values outside this range.
        for &v in &vals {
            assert!(v <= MAX);
        }
        btree.to_btree(&vals, &mut i, k);
        btree
    }

    // recursive function to create a btree
    // a is the original sorted array
    // k is the number of the block
    // i is the position in the original array
    fn to_btree(&mut self, a: &[u32], i: &mut usize, k: usize) {
        let num_blocks = (a.len() + B - 1) / B;
        if k < num_blocks {
            for j in 0..B {
                self.to_btree(a, i, Self::go_to(k, j));
                self.tree[k].data[j] = a.get(*i).unwrap_or(&MAX).clone();
                *i += 1;
            }
            self.to_btree(a, i, Self::go_to(k, B));
        }
    }

    fn go_to(k: usize, j: usize) -> usize {
        k * (B + 1) + j + 1
    }

    fn get(&self, b: usize, i: usize) -> u32 {
        unsafe { *self.tree.get_unchecked(b).data.get_unchecked(i) }
    }

    // basic searching with no vectorized magic inside the nodes
    pub fn search(&mut self, q: u32) -> u32 {
        // completely naive
        let mut k = 0;
        let btree_blocks = self.tree.len();
        let mut ans = MAX;
        while k < btree_blocks {
            let mut jump_to = 0;
            for j in 0..B {
                let compare_to = self.get(k, j);
                if q <= compare_to {
                    break;
                }
                jump_to += 1;
            }
            if jump_to < B {
                ans = self.get(k, jump_to);
            }
            k = Self::go_to(k, jump_to);
        }
        ans
    }

    pub fn search_loop(&mut self, q: u32) -> u32 {
        // completely naive
        let mut k = 0;
        let btree_blocks = self.tree.len();
        let mut ans = MAX;
        while k < btree_blocks {
            let mut jump_to = 0;
            for j in 0..B {
                let compare_to = self.get(k, j);
                jump_to += (q > compare_to) as usize;
            }
            if jump_to < B {
                ans = self.get(k, jump_to);
            }
            k = Self::go_to(k, jump_to);
        }
        ans
    }

    pub fn search_simd(&mut self, q: u32) -> u32 {
        // completely naive
        let mut k = 0;
        let btree_blocks = self.tree.len();
        let mut ans = MAX;
        while k < btree_blocks {
            let jump_to = self.tree[k].find(q);
            if jump_to < B {
                ans = self.get(k, jump_to);
            }
            k = Self::go_to(k, jump_to);
        }
        ans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments_sorted_arrays::BinarySearch;

    #[test]
    fn test_b_tree_k_2() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let btree = BTree::<2, 2>::new(vals);
        println!("{:?}", btree);
    }

    #[test]
    fn test_b_tree_k_3() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = BTree::<3, 3>::new(vals);
        println!("{:?}", computed_out);
    }

    #[test]
    fn test_btree_search_bottom_layer() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 452;
        let mut btree = BTree::<16, 16>::new(vals.clone());
        let btree_res = btree.search(q);

        let binsearch_res = BinarySearch::new(vals).search(q);
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }

    #[test]
    fn test_btree_search_top_node() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 289;
        let mut btree = BTree::<16, 16>::new(vals.clone());
        let btree_res = btree.search(q);

        let binsearch_res = BinarySearch::new(vals).search(q);
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }

    #[test]
    fn test_simd_cmp() {
        let mut vals: Vec<u32> = (1..16).collect();
        vals.push(MAX);
        let btree = BTree::<16, 16>::new(vals);
        let idx = btree.tree[0].find(1);
        println!("{}", idx);
        assert!(idx == 0);
    }

    #[test]
    fn test_btree_simd_bottom_layer() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 452;
        let mut btree = BTree::<16, 16>::new(vals.clone());
        let btree_res = btree.search_simd(q);

        let binsearch_res = BinarySearch::new(vals).search(q);
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }

    #[test]
    fn test_btree_simd_top_node() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 289;
        let mut btree = BTree::<16, 16>::new(vals.clone());
        let btree_res = btree.search_simd(q);

        let binsearch_res = BinarySearch::new(vals).search(q);
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }
}
