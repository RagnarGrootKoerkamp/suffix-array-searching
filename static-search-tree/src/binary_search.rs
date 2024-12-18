#![allow(dead_code)]

use crate::prefetch_index;

// completely basic binsearch
pub struct BinarySearch {
    vals: Vec<u32>,
}

impl BinarySearch {
    pub fn new(vals: Vec<u32>) -> Self {
        BinarySearch { vals }
    }

    fn get(&self, index: usize) -> u32 {
        unsafe { *self.vals.get_unchecked(index) }
    }

    /// Return the value of the first value >= query.
    pub fn search(&self, q: u32) -> u32 {
        let mut l = 0;
        let mut r = self.vals.len();
        while l < r {
            let m = (l + r) / 2;
            if self.get(m) < q {
                l = m + 1;
            } else {
                r = m;
            }
        }
        self.get(l)
    }

    /// branchless search (but does not work branchless yet)
    pub fn search_branchless(&self, q: u32) -> u32 {
        let mut base = 0;
        let mut len = self.vals.len();
        while len > 1 {
            let half = len / 2;
            base += (self.get(base + half - 1) < q) as usize * half;
            len = len - half;
        }
        self.get(base)
    }

    /// branchless search with prefetching (but does not work branchless yet)
    pub fn search_branchless_prefetch(&self, q: u32) -> u32 {
        let mut base = 0;
        let mut len = self.vals.len();
        while len > 1 {
            let half = len / 2;
            prefetch_index(&self.vals, base + half + len / 2 - 1);
            prefetch_index(&self.vals, base + len / 2 - 1);
            base += (self.get(base + half - 1) < q) as usize * half;
            len = len - half;
        }
        self.get(base)
    }
}
