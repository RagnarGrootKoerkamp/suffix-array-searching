use crate::prefetch_index;

pub struct Eytzinger {
    vals: Vec<u32>,
}

impl Eytzinger {
    pub fn new(vals: Vec<u32>) -> Eytzinger {
        let mut e = Eytzinger {
            vals: vec![0; vals.len() + 1], // +1 for one-based indexing
        };
        e.vals[0] = u32::MAX;
        e._to_eytzinger(&vals, &mut 0, 1);
        e
    }

    /// A recursive function to actually perform the Eytzinger transformation
    /// NOTE: This is not in-place.
    fn _to_eytzinger(&mut self, a: &[u32], i: &mut usize, k: usize) {
        if k <= a.len() {
            self._to_eytzinger(a, i, 2 * k);
            self.vals[k] = a[*i];
            *i += 1;
            self._to_eytzinger(a, i, 2 * k + 1);
        }
    }

    fn get(&self, index: usize) -> u32 {
        unsafe { *self.vals.get_unchecked(index) }
    }

    pub fn search(&self, q: u32) -> u32 {
        let mut index = 1;
        while index < self.vals.len() {
            index = 2 * index + (q > self.get(index)) as usize;
        }
        let zeros = index.trailing_ones() + 1;
        let idx = index >> zeros;
        self.get(idx)
    }

    pub fn search_prefetch<const B: usize>(&self, q: u32) -> u32 {
        let mut index = 1;
        while index < self.vals.len() {
            index = 2 * index + (q > self.get(index)) as usize;
            if B * index < self.vals.len() {
                prefetch_index(&self.vals, B * index);
            }
        }
        let zeros = index.trailing_ones() + 1;
        let idx = index >> zeros;
        self.get(idx)
    }
}

#[cfg(test)]
mod tests {

    use crate::binary_search::BinarySearch;

    use super::*;

    #[test]
    fn eytzinger_vs_binsearch() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let e = Eytzinger::new(input.clone());
        let b = BinarySearch::new(input);
        let q = 5;
        let ey_res = e.search(q);
        let bin_res = b.search(q);
        println!("{ey_res}, {bin_res}");
    }

    #[test]
    fn eytzinger_test_pow2_min_1() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        #[rustfmt::skip]
        let corr_output = vec![u32::MAX, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15];
        let e = Eytzinger::new(input.clone());
        assert_eq!(e.vals, corr_output);
    }

    #[test]
    fn eytzinger_test_non_pow2() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let corr_output = vec![u32::MAX, 6, 3, 8, 1, 5, 7, 9, 0, 2, 4];
        let e = Eytzinger::new(input.clone());
        assert_eq!(e.vals, corr_output);
    }

    #[test]
    fn eyetzinger_search_test() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let e = Eytzinger::new(input.clone());
        let q: u32 = 3;
        let result = e.search(q);
        assert_eq!(result, 3);
    }

    #[test]
    fn eyetzinger_search_oob() {
        let input = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let e = Eytzinger::new(input.clone());
        let q: u32 = 12;
        let result = e.search(q);
        assert_eq!(result, u32::MAX);
    }
}
