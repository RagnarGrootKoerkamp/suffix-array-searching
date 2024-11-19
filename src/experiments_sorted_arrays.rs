use std::intrinsics;

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
        base += (get(array, base + half) < q) as usize * half;
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
        base += (get(array, base + half) < q) as usize * half;
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
    let mut i: usize = 0;
    let k: usize = 1;
    _to_eytzinger(&array, &mut eytzinger, &mut i, k);
    eytzinger
}

// analogous to algorithmica
fn go_to<const B: usize>(k: usize, i: usize) -> usize {
    return k * (B + 1) + i + 1;
}

pub fn _to_btree<const B: usize>(a: &[u32], t: &mut Vec<u32>, i: &mut usize, k: usize) {
    let num_blocks = (a.len() + B - 1) / B;
    if k < num_blocks {
        for j in 0..B {
            _to_btree::<B>(a, t, i, go_to::<B>(k, j));
            if *i < a.len() {
                let x = a[*i];
                t[k * B + j] = x;
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
    fn eytzinger_test_pow2_min_1() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let corr_output = vec![0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15];
        let output = to_eytzinger(input);
        assert_eq!(output.len(), corr_output.len());
        let incorrect = corr_output.iter().zip(&output).filter(|&(a, b)| a != b).count();
        assert_eq!(incorrect, 0);
    }

    #[test]
    fn eytzinger_test_non_pow2() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let corr_output = vec![0, 6, 3, 8, 1, 5, 7, 9, 0, 2, 4];
        let output = to_eytzinger(input);
        let incorrect = corr_output.iter().zip(&output).filter(|&(a, b)| a != b).count();
        assert_eq!(incorrect, 0);
    }

    #[test]
    fn eyetzinger_search_test() {
        let eyetzinger_array = vec![0, 6, 3, 8, 1, 5, 7, 9, 0, 2, 4];
        let q: u32 = 3;
        let mut cnt: usize = 0;
        let result = eytzinger(&eyetzinger_array, 3, &mut cnt);
        assert_eq!(eyetzinger_array[result], 3);
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
        let orig_array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        // let corr_output = vec![0, 3, 6, 1, 2, 4, 5, 7, 8];
        let computed_out = to_btree::<3>(orig_array);
        println!("{:?}", computed_out);
    }
}

