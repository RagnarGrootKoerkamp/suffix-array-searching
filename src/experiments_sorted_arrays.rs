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

// completely basic binsearch
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


