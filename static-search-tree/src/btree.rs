use crate::{
    node::{BTreeNode, MAX},
    SearchIndex, SearchScheme,
};
use std::fmt::Debug;

#[derive(Debug)]
pub struct BTree<const B: usize, const N: usize> {
    tree: Vec<BTreeNode<N>>,
    pub cnt: usize,
}

pub type BTree16 = BTree<16, 16>;

impl<const B: usize, const N: usize> BTree<B, N> {
    fn go_to(&self, k: usize, j: usize) -> usize {
        k * (B + 1) + j + 1
    }

    fn get(&self, b: usize, i: usize) -> u32 {
        unsafe { *self.tree.get_unchecked(b).data.get_unchecked(i) }
    }
}

impl<const B: usize, const N: usize> SearchIndex for BTree<B, N> {
    fn new(vals: &[u32]) -> Self {
        // always have at least one node
        let n_blocks = vals.len().div_ceil(B);
        let mut btree = Self {
            tree: vec![BTreeNode::default(); n_blocks],
            cnt: 0,
        };
        let mut i: usize = 0;
        let k = 0;
        // SIMD operations fail for values outside this range.
        for &v in vals {
            assert!(v <= MAX);
        }

        // recursive function to create a btree
        // a is the original sorted array
        // k is the number of the block
        // i is the position in the original array
        fn recurse<const B: usize, const N: usize>(
            btree: &mut BTree<B, N>,
            a: &[u32],
            i: &mut usize,
            k: usize,
        ) {
            let num_blocks = (a.len() + B - 1) / B;
            if k < num_blocks {
                for j in 0..B {
                    recurse(btree, a, i, btree.go_to(k, j));
                    btree.tree[k].data[j] = a.get(*i).unwrap_or(&MAX).clone();
                    *i += 1;
                }
                recurse(btree, a, i, btree.go_to(k, B));
            }
        }

        recurse(&mut btree, &vals, &mut i, k);
        btree
    }
}

pub struct BTreeSearch<const B: usize, const N: usize>;
impl<const B: usize, const N: usize> SearchScheme<BTree<B, N>> for BTreeSearch<B, N> {
    // basic searching with no vectorized magic inside the nodes
    fn query_one(&self, index: &BTree<B, N>, q: u32) -> u32 {
        // completely naive
        let mut k = 0;
        let btree_blocks = index.tree.len();
        let mut ans = MAX;
        while k < btree_blocks {
            let mut jump_to = 0;
            for j in 0..B {
                let compare_to = index.get(k, j);
                if q <= compare_to {
                    break;
                }
                jump_to += 1;
            }
            if jump_to < B {
                ans = index.get(k, jump_to);
            }
            k = index.go_to(k, jump_to);
        }
        ans
    }
}

pub struct BTreeSearchLoop<const B: usize, const N: usize>;
impl<const B: usize, const N: usize> SearchScheme<BTree<B, N>> for BTreeSearchLoop<B, N> {
    fn query_one(&self, index: &BTree<B, N>, q: u32) -> u32 {
        // completely naive
        let mut k = 0;
        let btree_blocks = index.tree.len();
        let mut ans = MAX;
        while k < btree_blocks {
            let mut jump_to = 0;
            for j in 0..B {
                let compare_to = index.get(k, j);
                jump_to += (q > compare_to) as usize;
            }
            if jump_to < B {
                ans = index.get(k, jump_to);
            }
            k = index.go_to(k, jump_to);
        }
        ans
    }
}

pub struct BTreeSearchSimd<const B: usize, const N: usize>;
impl<const B: usize, const N: usize> SearchScheme<BTree<B, N>> for BTreeSearchSimd<B, N> {
    fn query_one(&self, index: &BTree<B, N>, q: u32) -> u32 {
        // completely naive
        let mut k = 0;
        let btree_blocks = index.tree.len();
        let mut ans = MAX;
        while k < btree_blocks {
            let jump_to = index.tree[k].find(q);
            if jump_to < B {
                ans = index.get(k, jump_to);
            }
            k = index.go_to(k, jump_to);
        }
        ans
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
        let btree = BTree::<2, 2>::new(&vals);
        println!("{:?}", btree);
    }

    #[test]
    fn test_b_tree_k_3() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = BTree::<3, 3>::new(&vals);
        println!("{:?}", computed_out);
    }

    #[test]
    fn test_btree_search_bottom_layer() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 452;
        let btree = BTree::<16, 16>::new(&vals);
        let btree_res = btree.query_one(q, &BTreeSearch);

        let binsearch_res = SortedVec::new(&vals).query_one(q, &BinarySearch);
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }

    #[test]
    fn test_btree_search_top_node() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 289;
        let btree = BTree::<16, 16>::new(&vals);
        let btree_res = btree.query_one(q, &BTreeSearch);

        let binsearch_res = SortedVec::new(&vals).query_one(q, &BinarySearch);
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }

    #[test]
    fn test_simd_cmp() {
        let mut vals: Vec<u32> = (1..16).collect();
        vals.push(MAX);
        let btree = BTree::<16, 16>::new(&vals);
        let idx = btree.tree[0].find(1);
        println!("{}", idx);
        assert!(idx == 0);
    }

    #[test]
    fn test_btree_simd_bottom_layer() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 452;
        let btree = BTree::<16, 16>::new(&vals);
        let btree_res = btree.query_one(q, &BTreeSearchSimd);

        let binsearch_res = SortedVec::new(&vals).query_one(q, &BinarySearch);
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }

    #[test]
    fn test_btree_simd_top_node() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(MAX);
        let q = 289;
        let btree = BTree::<16, 16>::new(&vals);
        let btree_res = btree.query_one(q, &BTreeSearchSimd);

        let binsearch_res = SortedVec::new(&vals).query_one(q, &BinarySearch);
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }
}
