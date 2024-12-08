use num_traits::bounds::Bounded;
use std::fmt::Debug;
use std::simd::prelude::*;
use std::simd::SimdElement;

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct BTreeNode<
    T: Ord + Copy + Default + Bounded + Debug + SimdElement,
    const B: usize,
    const PAD: usize,
> {
    data: [T; B],
    padding: [u8; PAD],
}

#[derive(Debug)]
pub struct BTree<
    T: Ord + Copy + Default + Bounded + Debug + SimdElement,
    const B: usize,
    const PAD: usize,
> {
    tree: Vec<BTreeNode<T, B, PAD>>,
}

impl<T: Ord + Copy + Default + Bounded + Debug + SimdElement, const B: usize, const PAD: usize>
    BTreeNode<T, B, PAD>
{
    pub fn new() -> BTreeNode<T, B, PAD> {
        BTreeNode {
            data: [T::max_value(); B],
            padding: [0; PAD],
        }
    }
}

impl<T: Ord + Copy + Default + Bounded + Debug + SimdElement, const B: usize, const PAD: usize>
    BTree<T, B, PAD>
where
    Simd<T, 16>: SimdPartialOrd,
    Simd<T, 16>: SimdPartialEq<Mask = Mask<T::Mask, 16>>,
{
    fn go_to(k: usize, j: usize) -> usize {
        k * (B + 1) + j + 1
    }

    // recursive function to create a btree
    // a is the original sorted array
    // i is the current
    // k is the number of the block
    // i is the position in the original array
    fn to_btree(a: &[T], t: &mut Vec<BTreeNode<T, B, PAD>>, i: &mut usize, k: usize) {
        let num_blocks = (a.len() + B - 1) / B;
        if k < num_blocks {
            for j in 0..B {
                BTree::<T, B, PAD>::to_btree(a, t, i, BTree::<T, B, PAD>::go_to(k, j));
                if *i < a.len() {
                    t[k].data[j] = a[*i];
                }
                // FIXME: figure out a way to get a default vlaue
                *i += 1;
            }
            BTree::to_btree(a, t, i, BTree::<T, B, PAD>::go_to(k, B));
        }
    }

    pub fn new(array: &[T]) -> BTree<T, B, PAD> {
        // always have at least one node
        let n_blocks = (array.len() + B) / B;
        let mut btree = vec![BTreeNode::<T, B, PAD>::new(); n_blocks];
        let mut i: usize = 0;
        let k = 0;
        BTree::<T, B, PAD>::to_btree(&array, &mut btree, &mut i, k);
        BTree { tree: btree }
    }

    fn get(&self, b: usize, i: usize) -> T {
        unsafe { *self.tree.get_unchecked(b).data.get_unchecked(i) }
    }

    // basic searching with no vectorized magic inside the nodes
    pub fn search(&self, q: T, cnt: &mut usize) -> T {
        // completely naive
        let mut mask = 1 << B;
        let mut k = 0;
        let mut res_block = usize::max_value();
        let btree_blocks = self.tree.len();
        let mut jump_to = 0;
        while k < btree_blocks {
            jump_to = 0;
            for j in 0..B {
                let compare_to = self.get(k, j);
                // FIXME: bad early stop
                if q == compare_to {
                    return self.get(k, jump_to);
                }
                if q <= compare_to {
                    break;
                }
                jump_to += 1;
            }
            res_block = k;
            k = BTree::<T, B, PAD>::go_to(k, jump_to);
        }
        return self.get(res_block, jump_to);
    }

    fn cmp(&self, q: T, node: &BTreeNode<T, B, PAD>) -> usize {
        // TODO: make this somehow work on all sizes
        const MASK_SIZE: usize = 16;
        assert!(B == MASK_SIZE);
        let data_simd: Simd<T, 16> =
            std::simd::prelude::Simd::<T, MASK_SIZE>::from_slice(&node.data[0..MASK_SIZE]);
        let q_simd = std::simd::prelude::Simd::<T, MASK_SIZE>::splat(q);
        let mask = q_simd.simd_lt(data_simd);
        //usize::try_from(mask.to_int().reduce_sum().abs()).unwrap()
        mask.first_set().unwrap()
    }

    pub fn search_simd(&self, q: T, cnt: &mut usize) -> T {
        // completely naive
        let mut mask = 1 << B;
        let mut k = 0;
        let mut res_block = usize::max_value();
        let btree_blocks = self.tree.len();
        let mut jump_to = 0;
        while k < btree_blocks {
            jump_to = 0;
            for j in 0..B {
                let compare_to = self.get(k, j);
                // FIXME: bad early stop
                if q == compare_to {
                    return self.get(k, jump_to);
                }
                if q <= compare_to {
                    break;
                }
                jump_to += 1;
            }
            res_block = k;
            k = BTree::<T, B, PAD>::go_to(k, jump_to);
        }
        return self.get(res_block, jump_to);
    }
}

pub type BTreeSearch<T, const COUNT: usize, const PAD: usize> =
    fn(&[BTreeNode<T, COUNT, PAD>], T, &mut usize) -> usize;
// takes as input a sorted array, returns a BTree
pub type ToBTree<T, const COUNT: usize, const PAD: usize> =
    fn(input: Vec<T>) -> Vec<BTreeNode<T, COUNT, PAD>>;

// pub fn btree_search_branchless<const B: usize>(btree: &[u32], q: u32, cnt: &mut usize) -> usize {
//     let mut mask = 1 << B;
//     let mut k = 0;
//     let mut res = usize::MAX;
//     let btree_blocks = btree.len() / B;

//     while k < btree_blocks {
//         let mut jump_to = 0;
//         // I'm searching for the first element that is <= to the searched one
//         for j in 0..B {
//             let compare_to = get(&btree, k * B + j);
//             jump_to += usize::from(q >= compare_to)
//         }
//         if jump_to < B {
//             res = k * B + jump_to;
//         }
//         k = go_to::<B>(k, jump_to);
//     }
//     return res;
// }

mod tests {
    use super::*;
    use crate::experiments_sorted_arrays;

    #[test]
    fn test_b_tree_k_2() {
        let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let computed_out = BTree::<u32, 2, 0>::new(&orig_array);
        println!("{:?}", computed_out);
    }

    #[test]
    fn test_b_tree_k_3() {
        let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = BTree::<u32, 3, 0>::new(&orig_array);
        println!("{:?}", computed_out);
    }

    #[test]
    fn test_btree_search_bottom_layer() {
        let mut array: Vec<u32> = (1..2000).collect();
        array.push(u32::MAX);
        let q = 452;
        let mut cnt: usize = 0;
        let btree = BTree::<u32, 16, 0>::new(&array);
        let btree_res = btree.search(q, &mut cnt);

        let binsearch_res = array[experiments_sorted_arrays::binary_search(&array, q, &mut cnt)];
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }

    #[test]
    fn test_btree_search_top_node() {
        let mut array: Vec<u32> = (1..2000).collect();
        array.push(u32::MAX);
        let q = 289;
        let mut cnt: usize = 0;
        let btree = BTree::<u32, 16, 0>::new(&array);
        let btree_res = btree.search(q, &mut cnt);

        let binsearch_res = array[experiments_sorted_arrays::binary_search(&array, q, &mut cnt)];
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }

    #[test]
    fn test_simd_cmp() {
        let mut array: Vec<u32> = (1..16).collect();
        array.push(u32::MAX);
        let btree = BTree::<u32, 16, 0>::new(&array);
        println!("{}", btree.cmp(1, &btree.tree[0]));
        assert!(false);
    }

    #[test]
    fn test_btree_simd_bottom_layer() {
        let mut array: Vec<u32> = (1..2000).collect();
        array.push(u32::MAX);
        let q = 452;
        let mut cnt: usize = 0;
        let btree = BTree::<u32, 16, 0>::new(&array);
        let btree_res = btree.search_simd(q, &mut cnt);

        let binsearch_res = array[experiments_sorted_arrays::binary_search(&array, q, &mut cnt)];
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }

    #[test]
    fn test_btree_simd_top_node() {
        let mut array: Vec<u32> = (1..2000).collect();
        array.push(u32::MAX);
        let q = 289;
        let mut cnt: usize = 0;
        let btree = BTree::<u32, 16, 0>::new(&array);
        let btree_res = btree.search_simd(q, &mut cnt);

        let binsearch_res = array[experiments_sorted_arrays::binary_search(&array, q, &mut cnt)];
        println!("{btree_res}, {binsearch_res}");
        assert!(btree_res == binsearch_res);
    }
}
