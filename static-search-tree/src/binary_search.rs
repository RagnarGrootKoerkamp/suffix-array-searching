#![allow(dead_code)]

use crate::{prefetch_index, vec_on_hugepages, SearchIndex};

pub struct SortedVec {
    pub(super) vals: Vec<u32>,
}

impl SortedVec {
    pub(super) fn get(&self, index: usize) -> u32 {
        unsafe { *self.vals.get_unchecked(index) }
    }
}

impl SearchIndex for SortedVec {
    fn new(vals: &[u32]) -> Self {
        assert!(vals.is_sorted());
        let mut vec = vec_on_hugepages(vals.len()).unwrap();
        vec.copy_from_slice(vals);
        SortedVec { vals: vec }
    }

    fn layers(&self) -> usize {
        self.vals.len().ilog2() as usize + 1
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self.vals.as_slice())
    }
}

impl SortedVec {
    /// Return the value of the first value >= query.
    pub fn binary_search(&self, q: u32) -> u32 {
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

    /// Return the value of the first value >= query.
    pub fn binary_search_std(&self, q: u32) -> u32 {
        let idx = self.vals.binary_search(&q).unwrap_or_else(|i| i);
        self.vals[idx]
    }

    /// branchless search (but does not work branchless yet)
    pub fn binary_search_branchless(&self, q: u32) -> u32 {
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
    pub fn binary_search_branchless_prefetch(&self, q: u32) -> u32 {
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

    /// might make sense to make it branchless, but we do not know yet how,
    /// as the non-batched branchless implementation still compiles to branchy
    pub fn batch_impl_binary_search_std<const P: usize>(&self, qb: &[u32; P]) -> [u32; P] {
        let mut bases = [0; P];
        let mut len = self.vals.len();
        while len > 1 {
            let half = len / 2;
            for i in 0..P {
                bases[i] += (self.get(bases[i] + half - 1) < qb[i]) as usize * half;
            }
            len = len - half;
        }

        bases.map(|x| self.get(x))
    }
}
