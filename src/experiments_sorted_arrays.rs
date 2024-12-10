#![allow(dead_code)]

use crate::prefetch_index;

pub type VanillaBinSearch = fn(&[u32], u32, &mut usize) -> usize;
pub type PreprocessArray = fn(input: &Vec<u32>) -> Vec<u32>;

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
// FIXME: branchless is inconsistent with normal binsearch when the query is larger than the largest value in the array
// Ragnar: That's fine. I'd say we leave that case as 'implementation defined' behaviour, since it can easily be fixed by padding with u32::MAX
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
// FIXME: branchless is inconsistent with normal binsearch when the query is larger than the largest value in the array
pub fn binary_search_branchless_prefetched(array: &[u32], q: u32, cnt: &mut usize) -> usize {
    let mut base = 0;
    let mut len = array.len();
    while len > 1 {
        let half = len / 2;
        *cnt += 1;
        prefetch_index(array, base + half + len / 2 - 1);
        prefetch_index(array, base + len / 2 - 1);
        base += (get(array, base + half - 1) < q) as usize * half;
        len = len - half;
    }
    base
}

pub fn eytzinger(array: &[u32], q: u32, _cnt: &mut usize) -> usize {
    let mut index = 1;
    while index < array.len() {
        index = 2 * index + usize::from(q > get(array, index));
    }
    let zeros = index.trailing_ones() + 1;
    index >> zeros
}

pub fn eytzinger_prefetched(array: &[u32], q: u32, _cnt: &mut usize) -> usize {
    let mut index: usize = 1;
    while index < array.len() {
        index = 2 * index + usize::from(q > get(array, index));
        prefetch_index(array, 16 * index);
    }
    let zeros = index.trailing_ones() + 1;
    index >> zeros
}

// a recursive function to actually perform the Eytzinger transformation
// FIXME: this is not in-place (which is okay for us), but we might have to implement this in-place
fn _to_eytzinger(a: &[u32], t: &mut Vec<u32>, i: &mut usize, k: usize) {
    if k <= a.len() {
        _to_eytzinger(a, t, i, 2 * k);
        t[k] = a[*i];
        *i += 1;
        _to_eytzinger(a, t, i, 2 * k + 1);
    }
}

pub fn to_eytzinger(array: &Vec<u32>) -> Vec<u32> {
    let mut eytzinger = vec![0; array.len() + 1]; // +1 for one-based indexing
    eytzinger[0] = u32::MAX;
    let mut i: usize = 0;
    let k: usize = 1;
    _to_eytzinger(&array, &mut eytzinger, &mut i, k);
    eytzinger
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eytzinger_vs_binsearch() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let preprocessed = to_eytzinger(&input);
        let q = 5;
        let mut cnt: usize = 0;
        let ey_res = eytzinger(&preprocessed, q, &mut cnt);
        let bin_res = binary_search(&input, q, &mut cnt);
        println!("{ey_res}, {bin_res}");
    }

    #[test]
    fn eytzinger_test_pow2_min_1() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let corr_output = vec![u32::MAX, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15];
        let output = to_eytzinger(&input);
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
        let output = to_eytzinger(&input);
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
}
