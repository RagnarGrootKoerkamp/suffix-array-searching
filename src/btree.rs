#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
struct BTreeNode<T: Ord + Copy + Default, const B: usize, const Pad: usize> {
    data: [T; B],
    padding: [u8; Pad],
}

#[derive(Debug)]
struct BTree<T: Ord + Copy + Default, const B: usize, const Pad: usize> {
    tree: Vec<BTreeNode<T, B, Pad>>,
}

impl<T: Ord + Copy + Default, const B: usize, const Pad: usize> BTreeNode<T, B, Pad> {
    pub fn new() -> BTreeNode<T, B, Pad> {
        BTreeNode {
            data: [T::default(); B],
            padding: [0; Pad],
        }
    }
}

impl<T: Ord + Copy + Default, const B: usize, const Pad: usize> BTree<T, B, Pad> {
    fn go_to(k: usize, j: usize) -> usize {
        k * (B + 1) + j
    }

    // recursive function to create a btree
    // a is the original sorted array
    // i is the current
    // k is the number of the block
    // i is the position in the original array
    fn to_btree(a: &[T], t: &mut Vec<BTreeNode<T, B, Pad>>, i: &mut usize, k: usize) {
        let num_blocks = (a.len() + B - 1) / B;
        if k < num_blocks {
            for j in 0..B {
                BTree::<T, B, Pad>::to_btree(a, t, i, BTree::<T, B, Pad>::go_to(k, j + 1));
                if *i < a.len() {
                    t[k].data[j] = a[*i];
                }
                // FIXME: figure out a way to get a default vlaue
                *i += 1;
            }
            BTree::to_btree(a, t, i, BTree::<T, B, Pad>::go_to(k, B));
        }
    }

    pub fn new(array: &[T]) -> BTree<T, B, Pad> {
        // => size of node equals K-1
        let n_blocks = (array.len() + B - 1) / B;
        let mut btree = vec![BTreeNode::<T, B, Pad>::new(); n_blocks];
        let mut i: usize = 0;
        let k = 0;
        BTree::<T, B, Pad>::to_btree(&array, &mut btree, &mut i, k);
        BTree { tree: btree }
    }

    fn get(&self, b: usize, i: usize) -> T {
        unsafe { *self.tree.get_unchecked(i).data.get_unchecked(i) }
    }

    // pub fn btree_search(&self, q: T, cnt: &mut usize) -> usize {
    //     // completely naive
    //     let mut mask = 1 << B;
    //     let mut k = 0;
    //     let mut res = usize::MAX;
    //     let btree_blocks = self.tree.len();
    //     while k < btree_blocks {
    //         let mut jump_to = 0;
    //         for j in 0..B {
    //             let compare_to = self.get(k, j);
    //             // FIXME: bad early stop
    //             if q <= compare_to {
    //                 break;
    //             }
    //             jump_to += 1;
    //         }
    //         k = B * k + j;
    //     }
    //     return res;
    // }
}

pub type BTreeSearch<T, const Count: usize, const Pad: usize> =
    fn(&[BTreeNode<T, Count, Pad>], T, &mut usize) -> usize;
// takes as input a sorted array, returns a BTree
pub type ToBTree<T, const Count: usize, const Pad: usize> =
    fn(input: Vec<T>) -> Vec<BTreeNode<T, Count, Pad>>;

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

// pub fn btree_search_simd<const B: usize>(btree: &[u32], q: u32, cnt: &mut usize) -> usize {
//     // for now assume B is 16
//     assert!(B == 16);
//     let mut k = 0;
//     let mut res = usize::MAX;
//     let btree_blocks = btree.len() / B;
//     // load the value q into a vector
//     let q_vec = u32x16::splat(q);
//     while k < btree_blocks {
//         // load the block
//         let block: [u32; 16] = btree[k * B..k * B + 16].try_into().unwrap();
//         let b_vec = u32x16::from_array(block);
//         // compare and assign to another vector
//         let comparison = b_vec.simd_ge(q_vec);
//         let jump_to: usize = match comparison.first_set() {
//             None => 16,
//             Some(i) => i,
//         };
//         if jump_to < B {
//             res = k * B + jump_to;
//         }
//         k = go_to::<B>(k, jump_to);
//     }
//     return res;
// }

mod tests {
    use super::*;

    #[test]
    fn test_b_tree_k_2() {
        let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let computed_out = BTree::<u32, 2, 0>::new(&orig_array);
        println!("{:?}", computed_out);
        assert!(false);
    }

    #[test]
    fn test_b_tree_k_3() {
        let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = BTree::<u32, 3, 0>::new(&orig_array);
        println!("{:?}", computed_out);
        assert!(false);
    }

    // #[test]
    // fn test_b_tree_k_3_not_round() {
    //     let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    //     let corr_output = vec![
    //         4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 4294967295, 4294967295,
    //     ];
    //     let computed_out = to_btree::<3>(orig_array);
    //     println!("{:?}", computed_out);
    //     assert_eq!(computed_out, corr_output);
    // }

    // #[test]
    // fn test_btree_search_oob() {
    //     let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    //     let computed_out = to_btree::<3>(orig_array);
    //     let mut cnt = 0;
    //     let result = btree_search::<3>(&computed_out, 0, &mut cnt);
    // }

    // #[test]
    // fn test_btree_and_btree_simd() {
    //     let array = (20..2000).collect();
    //     let btree = to_btree::<16>(array);
    //     let q = 20;
    //     let mut cnt = 0;
    //     let r1 = btree_search::<16>(&btree, q, &mut cnt);
    //     let r2 = btree_search_simd::<16>(&btree, q, &mut cnt);
    //     println!("results {} {}", r1, r2);
    //     println!("{} {}", btree[r1], btree[r2]);
    // }

    // #[test]
    // fn test_btree_basic_search() {
    //     let mut orig_array = Vec::new();
    //     let size = 1024;
    //     for i in 0..size {
    //         orig_array.push(i);
    //     }
    //     let mut cnt = 0;
    //     let q = 40;
    //     let btree = to_btree::<16>(orig_array);
    //     let i = btree_search::<16>(&btree, q, &mut cnt);
    //     assert_eq!(btree[i], q);
    // }

    // #[test]
    // fn test_btree_basic_search_elem_not_present() {
    //     let mut orig_array = Vec::new();
    //     let size = 1024;
    //     for i in 0..size {
    //         orig_array.push(i);
    //     }
    //     let mut cnt = 0;
    //     let q = 1024;
    //     let btree = to_btree::<16>(orig_array);
    //     let i = btree_search::<16>(&btree, q, &mut cnt);
    //     assert_eq!(i, usize::MAX);
    // }
}
