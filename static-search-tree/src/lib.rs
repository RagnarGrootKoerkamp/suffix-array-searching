#![feature(array_chunks, portable_simd)]

pub mod bench;
pub mod binary_search;
pub mod bplustree;
pub mod btree;
pub mod interp_search;
pub mod util;

pub use binary_search::{BinarySearch, Eytzinger};
pub use btree::BTree16;
pub use interp_search::InterpolationSearch;
pub use util::*;

#[ctor::ctor]
fn init_color_backtrace() {
    color_backtrace::install();
}
