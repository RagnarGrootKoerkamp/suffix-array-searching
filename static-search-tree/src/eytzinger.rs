use crate::{prefetch_index, SearchIndex};

pub struct Eytzinger {
    vals: Vec<u32>,
}

impl Eytzinger {
    fn get(&self, index: usize) -> u32 {
        unsafe { *self.vals.get_unchecked(index) }
    }
}

impl SearchIndex for Eytzinger {
    fn new(vals: &[u32]) -> Self {
        let mut e = Eytzinger {
            vals: vec![0; vals.len() + 1], // +1 for one-based indexing
        };
        e.vals[0] = u32::MAX;

        /// A recursive function to actually perform the Eytzinger transformation
        /// NOTE: This is not in-place.
        fn recurse(e: &mut Eytzinger, a: &[u32], i: &mut usize, k: usize) {
            if k <= a.len() {
                recurse(e, a, i, 2 * k);
                e.vals[k] = a[*i];
                *i += 1;
                recurse(e, a, i, 2 * k + 1);
            }
        }

        recurse(&mut e, &vals, &mut 0, 1);
        e
    }
}

impl Eytzinger {
    pub fn search(&self, q: u32) -> u32 {
        let mut idx = 1;
        while idx < self.vals.len() {
            idx = 2 * idx + (q > self.get(idx)) as usize;
        }
        let zeros = idx.trailing_ones() + 1;
        let idx = idx >> zeros;
        self.get(idx)
    }

    /// L: number of levels ahead to prefetch.
    pub fn search_prefetch<const L: usize>(&self, q: u32) -> u32 {
        let mut idx = 1;
        while idx < self.vals.len() {
            idx = 2 * idx + (q > self.get(idx)) as usize;
            if L * idx < self.vals.len() {
                prefetch_index(&self.vals, (1 << L) * idx);
            }
        }
        let zeros = idx.trailing_ones() + 1;
        let idx = idx >> zeros;
        self.get(idx)
    }
}

#[cfg(test)]
mod tests {

    use crate::{binary_search::SortedVec, SearchIndex};

    use super::*;

    #[test]
    fn eytzinger_vs_binsearch() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let q = 5;
        let ey_res = Eytzinger::new(&input).search(q);
        let bin_res = SortedVec::new(&input).binary_search(q);
        println!("{ey_res}, {bin_res}");
    }

    #[test]
    fn eytzinger_test_pow2_min_1() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        #[rustfmt::skip]
        let corr_output = vec![u32::MAX, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15];
        let e = Eytzinger::new(&input);
        assert_eq!(e.vals, corr_output);
    }

    #[test]
    fn eytzinger_test_non_pow2() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let corr_output = vec![u32::MAX, 6, 3, 8, 1, 5, 7, 9, 0, 2, 4];
        let e = Eytzinger::new(&input);
        assert_eq!(e.vals, corr_output);
    }

    #[test]
    fn eyetzinger_search_test() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let q: u32 = 3;
        let ey_res = Eytzinger::new(&input).search(q);
        assert_eq!(ey_res, 3);
    }

    #[test]
    fn eyetzinger_search_oob() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let q: u32 = 12;
        let ey_res = Eytzinger::new(&input).search(q);
        assert_eq!(ey_res, u32::MAX);
    }
}
