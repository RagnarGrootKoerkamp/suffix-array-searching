#![feature(array_chunks, portable_simd)]

mod bench;
pub mod binary_search;
pub mod bplustree;
pub mod btree;
pub mod eytzinger;
pub mod interp_search;
mod node;
pub mod util;

use util::*;

pub use bench::SearchFunctions;

#[ctor::ctor]
fn init_color_backtrace() {
    color_backtrace::install();
}

/// Construct the data structure from a sorted vector.
pub trait SearchIndex: Sized {
    fn new(vals: &[u32]) -> Self;

    // Convenience methods to forward to a search scheme.
    fn query_one(&self, q: u32, scheme: &(impl SearchScheme<Self> + ?Sized)) -> u32 {
        scheme.query_one(&self, q)
    }
    fn query(&self, qs: &[u32], scheme: &(impl SearchScheme<Self> + ?Sized)) -> Vec<u32> {
        scheme.query(&self, qs)
    }
}

/// Add a search scheme to an index.
pub trait SearchScheme<INDEX>: Sync + Send {
    fn query_one(&self, index: &INDEX, q: u32) -> u32 {
        self.query(index, &vec![q])[0]
    }
    fn query(&self, index: &INDEX, qs: &[u32]) -> Vec<u32> {
        qs.iter().map(|&q| self.query_one(index, q)).collect()
    }
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}
