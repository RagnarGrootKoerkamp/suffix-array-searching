pub type VanillaBinSearch = fn(&[u32], u32, &mut usize) -> usize;

// completely basic binsearch
pub fn binary_search(array: &[u32], q: u32, cnt: &mut usize) -> usize {
    let mut l = 0;
    let mut r = array.len();
    while l < r {
        *cnt += 1;
        let m = (l + r) / 2;
        if array[m] < q {
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
        base += (array[base + half] < q) as usize * half;
        len = len - half;
    }
    base
}

// a recursive function to actually perform the Eytzinger transformation
fn _to_eytzinger(a: &[u32], t: &mut Vec<u32>, i: &mut usize, k: usize) {
    if (k <= a.len()) {
        _to_eytzinger(a, t, i, 2 * k);
        t[k] = a[*i];
        *i += 1;
        _to_eytzinger(a, t, i, 2 * k + 1);
    }
}

pub fn to_eytzinger(array: &[u32]) -> Vec<u32> {
    let mut eytzinger = vec![0; array.len() + 1]; // +1 for one-based indexing
    let mut i: usize = 0;
    let k: usize = 1;
    _to_eytzinger(array, &mut eytzinger, &mut i, k);
    eytzinger
}

mod tests {
    use super::*;

    #[test]
    fn eytzinger_test_pow2() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let corr_output = vec![0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15];
        let output = to_eytzinger(&input);
        let incorrect = corr_output.iter().zip(&output).filter(|&(a, b)| a != b).count();
        assert_eq!(incorrect, 0);
    }

    #[test]
    fn eytzinger_test_non_pow2() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let corr_output = vec![0, 6, 3, 8, 1, 5, 7, 9, 0, 2, 4];
        let output = to_eytzinger(&input);
        let incorrect = corr_output.iter().zip(&output).filter(|&(a, b)| a != b).count();
        println!("{:?}", output);
        assert_eq!(incorrect, 0);
    }
}

