use std::fmt::Debug;

use crate::btree::BTreeNode;

#[derive(Debug)]
pub struct BpTree<const B: usize, const PAD: usize> {
    tree: Vec<BTreeNode<B, PAD>>,
    pub cnt: usize,
    height: usize,
    offsets: Vec<usize>,
}

pub type BpTree16 = BpTree<16, 0>;

impl<const B: usize, const PAD: usize> BpTree<B, PAD> {
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
            tree: vec![
                BTreeNode {
                    data: [u32::MAX; B],
                    _padding: [0; PAD],
                };
                n_blocks
            ],
            cnt: 0,
            height,
            offsets: (0..=height).map(|h| Self::offset(n, h)).collect(),
        };
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
                    u32::MAX
                };
            }
        }
        bptree
    }

    fn go_to(k: usize, j: usize) -> usize {
        k * (B + 1) + j + 1
    }

    fn node(&self, b: usize) -> &BTreeNode<B, PAD> {
        unsafe { &*self.tree.get_unchecked(b) }
    }

    fn get(&self, b: usize, i: usize) -> u32 {
        unsafe { *self.tree.get_unchecked(b).data.get_unchecked(i) }
    }

    pub fn search(&mut self, q: u32) -> u32 {
        // completely naive
        let mut k = 0;
        for h in (1..=self.height - 1).rev() {
            let jump_to = self.node(self.offsets[h] + k).find(q);
            k = k * (B + 1) + jump_to;
        }

        let index = self.node(k).find(q);
        self.node(k + index / B).data[index % B]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments_sorted_arrays::BinarySearch;

    #[test]
    fn test_b_tree_k_2() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let bptree = BpTree::<2, 0>::new(vals);
        println!("{:?}", bptree);
    }

    #[test]
    fn test_b_tree_k_3() {
        let vals = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        // let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = BpTree::<3, 0>::new(vals);
        println!("{:?}", computed_out);
    }

    #[test]
    fn test_bptree_search_bottom_layer() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(u32::MAX);
        let q = 452;
        let mut bptree = BpTree::<16, 0>::new(vals.clone());
        let bptree_res = bptree.search(q);

        let binsearch_res = BinarySearch::new(vals).search(q);
        println!("{bptree_res}, {binsearch_res}");
        assert!(bptree_res == binsearch_res);
    }

    #[test]
    fn test_bptree_search_top_node() {
        let mut vals: Vec<u32> = (1..2000).collect();
        vals.push(u32::MAX);
        let q = 289;
        let mut bptree = BpTree::<16, 0>::new(vals.clone());
        let bptree_res = bptree.search(q);

        let binsearch_res = BinarySearch::new(vals).search(q);
        println!("{bptree_res}, {binsearch_res}");
        assert!(bptree_res == binsearch_res);
    }

    #[test]
    fn test_simd_cmp() {
        let mut vals: Vec<u32> = (1..16).collect();
        vals.push(u32::MAX);
        let bptree = BpTree::<16, 0>::new(vals);
        let idx = bptree.tree[0].find(1);
        println!("{}", idx);
        assert!(idx == 0);
    }
}
