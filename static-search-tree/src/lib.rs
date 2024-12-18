#![feature(array_chunks, portable_simd)]

pub mod bench;
pub mod binary_search;
pub mod bplustree;
pub mod btree;
pub mod eytzinger;
pub mod interp_search;
mod node;
pub mod util;

pub use btree::BTree16;
pub use eytzinger::Eytzinger;
pub use interp_search::InterpolationSearch;
pub use util::*;

#[ctor::ctor]
fn init_color_backtrace() {
    color_backtrace::install();
}

/// Construct the data structure from a sorted vector.
pub trait SearchIndex {
    fn new(vals: &[u32]) -> Self;

    // Convenience methods to forward to a search scheme.
    fn query_one(&self, q: u32, scheme: impl SearchScheme<INDEX = Self>) -> u32 {
        scheme.query_one(&self, q)
    }
    fn query(&self, qs: &[u32], scheme: impl SearchScheme<INDEX = Self>) -> Vec<u32> {
        scheme.query(&self, qs)
    }
}

/// Add a search scheme to an index.
pub trait SearchScheme: Sync + Send {
    type INDEX;
    fn query_one(&self, index: &Self::INDEX, q: u32) -> u32 {
        self.query(index, &vec![q])[0]
    }
    fn query(&self, index: &Self::INDEX, qs: &[u32]) -> Vec<u32> {
        qs.iter().map(|&q| self.query_one(index, q)).collect()
    }
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

impl<I> SearchScheme for &dyn SearchScheme<INDEX = I> {
    type INDEX = I;
    fn query_one(&self, index: &I, q: u32) -> u32 {
        (*self).query_one(index, q)
    }
    fn query(&self, index: &I, qs: &[u32]) -> Vec<u32> {
        (*self).query(index, qs)
    }
    fn name(&self) -> &'static str {
        (*self).name()
    }
}
