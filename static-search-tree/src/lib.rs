#![feature(array_chunks, portable_simd)]

pub mod bench;
pub mod bplustree;
pub mod btree;
pub mod experiments_sorted_arrays;
pub mod interp_search;
pub mod util;

pub use btree::BTree16;
pub use experiments_sorted_arrays::{BinarySearch, Eytzinger};
pub use interp_search::InterpolationSearch;
pub use util::*;

#[ctor::ctor]
fn init_color_backtrace() {
    color_backtrace::install();
}
