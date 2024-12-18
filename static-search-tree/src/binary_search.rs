#![allow(dead_code)]

use crate::{prefetch_index, SearchIndex, SearchScheme};

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
        SortedVec {
            vals: vals.to_vec(),
        }
    }
}

pub struct BinarySearch;
impl SearchScheme for BinarySearch {
    type INDEX = SortedVec;

    /// Return the value of the first value >= query.
    fn query_one(&self, index: &SortedVec, q: u32) -> u32 {
        let mut l = 0;
        let mut r = index.vals.len();
        while l < r {
            let m = (l + r) / 2;
            if index.get(m) < q {
                l = m + 1;
            } else {
                r = m;
            }
        }
        index.get(l)
    }
}

pub struct BinarySearchStd;
impl SearchScheme for BinarySearchStd {
    type INDEX = SortedVec;

    /// Return the value of the first value >= query.
    fn query_one(&self, index: &SortedVec, q: u32) -> u32 {
        let idx = index.vals.binary_search(&q).unwrap_or_else(|i| i);
        index.vals[idx]
    }
}

pub struct BinarySearchBranchless;
impl SearchScheme for BinarySearchBranchless {
    type INDEX = SortedVec;

    /// branchless search (but does not work branchless yet)
    fn query_one(&self, index: &SortedVec, q: u32) -> u32 {
        let mut base = 0;
        let mut len = index.vals.len();
        while len > 1 {
            let half = len / 2;
            base += (index.get(base + half - 1) < q) as usize * half;
            len = len - half;
        }
        index.get(base)
    }
}

pub struct BinarySearchBranchlessPrefetch;
impl SearchScheme for BinarySearchBranchlessPrefetch {
    type INDEX = SortedVec;

    /// branchless search with prefetching (but does not work branchless yet)
    fn query_one(&self, index: &SortedVec, q: u32) -> u32 {
        let mut base = 0;
        let mut len = index.vals.len();
        while len > 1 {
            let half = len / 2;
            prefetch_index(&index.vals, base + half + len / 2 - 1);
            prefetch_index(&index.vals, base + len / 2 - 1);
            base += (index.get(base + half - 1) < q) as usize * half;
            len = len - half;
        }
        index.get(base)
    }
}
