use std::intrinsics;
use std::intrinsics::simd::simd_ge;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::{SimdInt, SimdUint};
use std::simd::u32x16;

pub type VanillaBinSearch = fn(&[u32], u32, &mut usize) -> usize;
pub type PreprocessArray = fn(input: Vec<u32>) -> Vec<u32>;

// FIXME: is this a good way to go around this?
fn get(array: &[u32], index: usize) -> u32 {
    unsafe { *array.get_unchecked(index) }
}

// completely basic binsearch
pub fn binary_search(array: &[u32], q: u32, cnt: &mut usize) -> usize {
    let mut l = 0;
    let mut r = array.len();
    while l < r {
        *cnt += 1;
        let m = (l + r) / 2;
        // TODO should I prefetch both or just one?
        if get(array, m) < q {
            l = m + 1;
        } else {
            r = m;
        }
    }
    l as usize
}

// branchless search (but does not work branchless yet)
pub fn binary_search_branchless(array: &[u32], q: u32, cnt: &mut usize) -> usize {
    let mut base = 0;
    let mut len = array.len();
    while len > 1 {
        let half = len / 2;
        *cnt += 1;
        base += (get(&array, base + half - 1) < q) as usize * half;
        len = len - half;
    }
    base
}

// branchless search (but does not work branchless yet)
pub fn binary_search_branchless_prefetched(array: &[u32], q: u32, cnt: &mut usize) -> usize {
    let mut base = 0;
    let mut len = array.len();
    while len > 1 {
        let half = len / 2;
        *cnt += 1;
        unsafe {
            let ptr_right = &array[base + half + len / 2 - 1] as *const u32;
            let ptr_left = &array[base + len / 2 - 1] as *const u32;
            std::intrinsics::prefetch_read_data(ptr_left, 3);
            std::intrinsics::prefetch_read_data(ptr_right, 3);
        }
        base += (get(array, base + half - 1) < q) as usize * half;
        len = len - half;
    }
    base
}

pub fn eytzinger(array: &[u32], q: u32, cnt: &mut usize) -> usize {
    let mut index = 1;
    while index < array.len() {
        index = 2 * index + usize::from(q > get(array, index));
    }
    let zeros = index.trailing_ones() + 1;
    index >> zeros
}

pub fn eytzinger_prefetched(array: &[u32], q: u32, cnt: &mut usize) -> usize {
    let mut index: usize = 1;
    while index < array.len() {
        index = 2 * index + usize::from(q > get(array, index));
        unsafe {
            let ptr = (&array[0] as *const u32).offset((index * 16).try_into().unwrap());
            std::intrinsics::prefetch_read_data(ptr, 3);
        };
    }
    let zeros = index.trailing_ones() + 1;
    index >> zeros
}

// analogous to algorithmica
fn go_to<const B: usize>(k: usize, i: usize) -> usize {
    return k * (B + 1) + i + 1;
}

pub fn btree_search<const B: usize>(btree: &[u32], q: u32, cnt: &mut usize) -> usize {
    // completely naive
    let mut mask = 1 << B;
    let mut k = 0;
    let mut res = usize::MAX;
    let btree_blocks = btree.len() / B;
    while k < btree_blocks {
        let mut jump_to = 0;
        for j in 0..B {
            let compare_to = get(&btree, k * B + j);
            // FIXME: bad early stop
            if q <= compare_to {
                break;
            }
            jump_to += 1;
        }
        if jump_to < B {
            res = k * B + jump_to;
        }
        k = go_to::<B>(k, jump_to);
    }
    return res;
}

pub fn btree_search_branchless<const B: usize>(btree: &[u32], q: u32, cnt: &mut usize) -> usize {
    let mut mask = 1 << B;
    let mut k = 0;
    let mut res = usize::MAX;
    let btree_blocks = btree.len() / B;

    while k < btree_blocks {
        let mut jump_to = 0;
        // I'm searching for the first element that is <= to the searched one
        for j in 0..B {
            let compare_to = get(&btree, k * B + j);
            jump_to += usize::from(q >= compare_to)
        }
        if jump_to < B {
            res = k * B + jump_to;
        }
        k = go_to::<B>(k, jump_to);
    }
    return res;
}

pub fn btree_search_simd<const B: usize>(btree: &[u32], q: u32, cnt: &mut usize) -> usize {
    // for now assume B is 16
    assert!(B == 16);
    let mut k = 0;
    let mut res = usize::MAX;
    let btree_blocks = btree.len() / B;
    // load the value q into a vector
    let q_vec = u32x16::splat(q);
    while k < btree_blocks {
        // load the block
        let block: [u32; 16] = btree[k * B..k * B + 16].try_into().unwrap();
        let b_vec = u32x16::from_array(block);
        // compare and assign to another vector
        let comparison = b_vec.simd_ge(q_vec);
        let jump_to: usize = match comparison.first_set() {
            None => 16,
            Some(i) => i,
        };
        if jump_to < B {
            res = k * B + jump_to;
        }
        k = go_to::<B>(k, jump_to);
    }
    return res;
}

// a recursive function to actually perform the Eytzinger transformation
// FIXME: this is not in-place (which is okay for us), but we might have to implement this in-place
fn _to_eytzinger(a: &[u32], t: &mut Vec<u32>, i: &mut usize, k: usize) {
    if (k <= a.len()) {
        _to_eytzinger(a, t, i, 2 * k);
        t[k] = a[*i];
        *i += 1;
        _to_eytzinger(a, t, i, 2 * k + 1);
    }
}

pub fn to_eytzinger(array: Vec<u32>) -> Vec<u32> {
    let mut eytzinger = vec![0; array.len() + 1]; // +1 for one-based indexing
    eytzinger[0] = u32::MAX;
    let mut i: usize = 0;
    let k: usize = 1;
    _to_eytzinger(&array, &mut eytzinger, &mut i, k);
    eytzinger
}

pub fn _to_btree<const B: usize>(a: &[u32], t: &mut Vec<u32>, i: &mut usize, k: usize) {
    let num_blocks = (a.len() + B - 1) / B;
    if k < num_blocks {
        for j in 0..B {
            _to_btree::<B>(a, t, i, go_to::<B>(k, j));
            if *i < a.len() {
                let x = a[*i];
                t[k * B + j] = x;
            } else {
                t[k * B + j] = u32::MAX;
            }
            *i += 1;
        }
        _to_btree::<B>(a, t, i, go_to::<B>(k, B));
    }
}

pub fn to_btree<const B: usize>(array: Vec<u32>) -> Vec<u32> {
    // => size of node equals K-1
    let n_blocks = (array.len() + B - 1) / B;
    let mut btree = vec![0; n_blocks * B];
    let mut i: usize = 0;
    let k = 0;
    _to_btree::<B>(&array, &mut btree, &mut i, k);
    btree
}

mod tests {
    use super::*;

    #[test]
    fn branchless_test_oob() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let q = 16;
        let mut cnt = 0;
        let result = binary_search_branchless(&input, q, &mut cnt);
        // result should be out-of-bounds of the array
        assert!(result == 15);
    }

    #[test]
    fn eytzinger_test_pow2_min_1() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let corr_output = vec![u32::MAX, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15];
        let output = to_eytzinger(input);
        assert_eq!(output.len(), corr_output.len());
        let incorrect = corr_output
            .iter()
            .zip(&output)
            .filter(|&(a, b)| a != b)
            .count();
        assert_eq!(incorrect, 0);
    }

    #[test]
    fn eytzinger_test_non_pow2() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let corr_output = vec![u32::MAX, 6, 3, 8, 1, 5, 7, 9, 0, 2, 4];
        let output = to_eytzinger(input);
        let incorrect = corr_output
            .iter()
            .zip(&output)
            .filter(|&(a, b)| a != b)
            .count();
        assert_eq!(incorrect, 0);
    }

    #[test]
    fn eyetzinger_search_test() {
        let eyetzinger_array = vec![u32::MAX, 6, 3, 8, 1, 5, 7, 9, 0, 2, 4];
        let q: u32 = 3;
        let mut cnt: usize = 0;
        let result = eytzinger(&eyetzinger_array, q, &mut cnt);
        assert_eq!(eyetzinger_array[result], 3);
    }

    #[test]
    fn eyetzinger_search_oob() {
        let eyetzinger_array = vec![u32::MAX, 6, 3, 8, 1, 5, 7, 9, 0, 2, 4];
        let q: u32 = 12;
        let mut cnt: usize = 0;
        let result = eytzinger(&eyetzinger_array, q, &mut cnt);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_b_tree_k_2() {
        let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let correct_output = vec![3, 6, 1, 2, 4, 5, 7, 8];
        let computed_out = to_btree::<2>(orig_array);
        println!("{:?}", computed_out);
        assert_eq!(correct_output, computed_out);
    }

    #[test]
    fn test_b_tree_k_3() {
        let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let correct_output = vec![4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        let computed_out = to_btree::<3>(orig_array);
        println!("{:?}", computed_out);
        assert_eq!(correct_output, computed_out);
    }

    #[test]
    fn test_b_tree_k_3_not_round() {
        let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        let corr_output = vec![
            4, 8, 12, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 4294967295, 4294967295,
        ];
        let computed_out = to_btree::<3>(orig_array);
        println!("{:?}", computed_out);
        assert_eq!(computed_out, corr_output);
    }

    #[test]
    fn test_btree_search_oob() {
        let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        let computed_out = to_btree::<3>(orig_array);
        let mut cnt = 0;
        let result = btree_search::<3>(&computed_out, 0, &mut cnt);
    }

    #[test]
    fn test_btree_and_btree_simd() {
        let array = (20..2000).collect();
        let btree = to_btree::<16>(array);
        let q = 20;
        let mut cnt = 0;
        let r1 = btree_search::<16>(&btree, q, &mut cnt);
        let r2 = btree_search_simd::<16>(&btree, q, &mut cnt);
        println!("results {} {}", r1, r2);
        println!("{} {}", btree[r1], btree[r2]);
    }

    #[test]
    fn test_btree_basic_search() {
        let mut orig_array = Vec::new();
        let size = 1024;
        for i in 0..size {
            orig_array.push(i);
        }
        let mut cnt = 0;
        let q = 40;
        let btree = to_btree::<16>(orig_array);
        let i = btree_search::<16>(&btree, q, &mut cnt);
        assert_eq!(btree[i], q);
    }

    #[test]
    fn test_btree_basic_search_elem_not_present() {
        let mut orig_array = Vec::new();
        let size = 1024;
        for i in 0..size {
            orig_array.push(i);
        }
        let mut cnt = 0;
        let q = 1024;
        let btree = to_btree::<16>(orig_array);
        let i = btree_search::<16>(&btree, q, &mut cnt);
        assert_eq!(i, usize::MAX);
    }
}
