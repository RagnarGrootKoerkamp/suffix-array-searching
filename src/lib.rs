#![feature(array_chunks, portable_simd)]

pub mod bplustree;
pub mod btree;
pub mod experiments_sorted_arrays;
pub mod sa_search;
pub mod util;

use bplustree::{BpTree15, BpTree16, BpTree16R};
pub use btree::BTree16;
use btree::MAX;
pub use experiments_sorted_arrays::{BinarySearch, Eytzinger};
use itertools::Itertools;
use pyo3::prelude::*;
use rand::Rng;
use rdst::RadixSort;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;
use tracing::info;
pub use util::*;

const LOWEST_GENERATED: u32 = 0;
const HIGHEST_GENERATED: u32 = 42000000;

pub type Fn<T> = (&'static str, fn(&mut T, u32) -> u32);
pub type BFn<const B: usize, T> = (&'static str, fn(&mut T, &[u32; B]) -> [u32; B]);

pub fn run<T>(searcher: &mut T, search: Fn<T>, queries: &[u32]) -> Vec<u32> {
    queries
        .into_iter()
        .map(|q| search.1(searcher, *q))
        .collect()
}
pub fn run_batch<const B: usize, T>(
    searcher: &mut T,
    search: BFn<B, T>,
    queries: &[u32],
) -> Vec<u32> {
    queries
        .array_chunks::<B>()
        .flat_map(|qs| search.1(searcher, qs))
        .collect()
}

pub fn bench<T>(searcher: &mut T, search: Fn<T>, queries: &[u32]) -> f64 {
    info!("Benching {}", search.0);
    let start = Instant::now();
    for q in queries {
        black_box(search.1(searcher, *q));
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / queries.len() as f64
}

pub fn bench_batch<const B: usize, T>(searcher: &mut T, search: BFn<B, T>, queries: &[u32]) -> f64 {
    info!("Benching {}", search.0);
    let start = Instant::now();
    for qs in queries.array_chunks::<B>() {
        black_box(search.1(searcher, qs));
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / queries.len() as f64
}

pub fn gen_queries(n: usize) -> Vec<u32> {
    (0..n)
        .map(|_| rand::thread_rng().gen_range(LOWEST_GENERATED..HIGHEST_GENERATED))
        .collect()
}

/// Generate a u32 array of the given *size* in bytes, and ending in i32::MAX.
pub fn gen_vals(size: usize, sort: bool) -> Vec<u32> {
    let n = size / std::mem::size_of::<u32>();
    // TODO: generate a new array
    let mut vals = (0..n - 1)
        .map(|_| rand::thread_rng().gen_range(LOWEST_GENERATED..HIGHEST_GENERATED))
        .collect_vec();
    vals.push(MAX);
    if sort {
        vals.radix_sort_unstable();
    }
    vals
}

#[pyclass]
struct BenchmarkSortedArray {
    bs: Vec<Fn<BinarySearch>>,
    eyt: Vec<Fn<Eytzinger>>,
    bt: Vec<Fn<BTree16>>,
    bp: Vec<Fn<BpTree16>>,
}

impl BenchmarkSortedArray {
    #[allow(unused)]
    fn test_all(&self) -> bool {
        const TEST_START_POW2: usize = 6;
        const TEST_END_POW2: usize = 20;
        const TEST_QUERIES: usize = 10000usize.next_multiple_of(128);

        let mut correct = true;
        for pow2 in TEST_START_POW2..=TEST_END_POW2 {
            let size = 2usize.pow(pow2 as u32);
            let vals = gen_vals(size, true);
            let queries = &gen_queries(TEST_QUERIES);

            let mut results = vec![];

            let bs = &mut BinarySearch::new(vals.clone());

            for &f in &self.bs {
                let new_results = run(bs, f, queries);
                if results.is_empty() {
                    results = new_results;
                } else {
                    if results != new_results {
                        correct = false;
                    }
                }
            }

            let eyt = &mut Eytzinger::new(vals.clone());

            for &f in &self.eyt {
                let new_results = run(eyt, f, queries);
                if results != new_results {
                    correct = false;
                }
            }

            let bt = &mut BTree16::new(vals.clone());

            for &f in &self.bt {
                let new_results = run(bt, f, queries);
                if results != new_results {
                    correct = false;
                }
            }

            let bp = &mut BpTree16::new(vals.clone());

            for &f in &self.bp {
                let new_results = run(bp, f, queries);
                if results != new_results {
                    correct = false;
                }
            }

            let f: BFn<128, _> = ("bp_batch", BpTree16::batch::<128>);
            let new_results = run_batch(bp, f, queries);
            assert_eq!(results, new_results, "{}\n{:?}", f.0, vals);

            let f: BFn<128, _> = ("bp_batch_prefetch", BpTree16::batch_prefetch::<128>);
            let new_results = run_batch(bp, f, queries);
            assert_eq!(results.len(), new_results.len(), "{}", f.0);
            assert_eq!(results, new_results, "{}", f.0);

            let f: BFn<128, _> = ("bp_batch_ptr", BpTree16::batch_ptr::<128>);
            let new_results = run_batch(bp, f, queries);
            assert_eq!(results.len(), new_results.len(), "{}", f.0);
            assert_eq!(results, new_results, "{}", f.0);

            let f: BFn<128, _> = ("bp_batch_ptr", BpTree16::batch_ptr2::<128>);
            let new_results = run_batch(bp, f, queries);
            assert_eq!(results, new_results, "{}", f.0);

            let f: BFn<128, _> = ("bp_batch_ptr3", BpTree16::batch_ptr3::<128, false>);
            let new_results = run_batch(bp, f, queries);
            assert_eq!(results, new_results, "{}\n{vals:?}", f.0);

            let f: BFn<128, _> = ("bp_batch_ptr3_last", BpTree16::batch_ptr3::<128, true>);
            let last_results = run_batch(bp, f, queries);

            let bp = &mut BpTree15::new(vals.clone());

            let f: BFn<128, _> = ("bp_batch", BpTree15::batch::<128>);
            let new_results = run_batch(bp, f, queries);
            assert_eq!(results, new_results, "{}", f.0);

            let f: BFn<128, _> = ("bp_batch_prefetch", BpTree15::batch_prefetch::<128>);
            let new_results = run_batch(bp, f, queries);
            assert_eq!(results, new_results, "{}", f.0);

            let f: BFn<128, _> = ("bp_batch_ptr", BpTree15::batch_ptr::<128>);
            let new_results = run_batch(bp, f, queries);
            assert_eq!(results, new_results, "{}", f.0);

            let f: BFn<128, _> = ("bp_batch_ptr", BpTree15::batch_ptr2::<128>);
            let new_results = run_batch(bp, f, queries);
            assert_eq!(results, new_results, "{}", f.0);

            let bpr = &mut BpTree16R::new(vals.clone());

            let f: BFn<128, _> = ("bp_batch_ptr3_rev", BpTree16R::batch_ptr3::<128, false>);
            let new_results = run_batch(bpr, f, queries);
            assert_eq!(results, new_results, "{}\n{vals:?}", f.0);

            let f: BFn<128, _> = ("bp_batch_ptr3_rev_last", BpTree16R::batch_ptr3::<128, true>);
            let new_last_results = run_batch(bpr, f, queries);
            assert_eq!(last_results, new_last_results, "{}\n{vals:?}", f.0);
        }
        correct
    }
}

#[pymethods]
impl BenchmarkSortedArray {
    #[new]
    fn new() -> Self {
        *INIT_TRACE;

        let bs: Vec<Fn<_>> = vec![
            ("bs_search", BinarySearch::search),
            ("bs_branchless", BinarySearch::search_branchless),
            (
                "bs_branchless_prefetch",
                BinarySearch::search_branchless_prefetch,
            ),
        ];
        let eyt: Vec<Fn<_>> = vec![
            ("eyt_search", Eytzinger::search),
            ("eyt_prefetch_4", Eytzinger::search_prefetch::<4>),
            ("eyt_prefetch_8", Eytzinger::search_prefetch::<8>),
            ("eyt_prefetch_16", Eytzinger::search_prefetch::<16>),
        ];
        let bt: Vec<Fn<_>> = vec![
            ("bt_search", BTree16::search),
            ("bt_loop", BTree16::search_loop),
            ("bt_simd", BTree16::search_simd),
        ];
        let bp: Vec<Fn<_>> = vec![
            ("bp_search", BpTree16::search),
            ("bp_search_split", BpTree16::search_split),
        ];

        BenchmarkSortedArray { bs, eyt, bt, bp }
    }

    fn benchmark(
        &self,
        start_pow2: usize,
        stop_pow2: usize,
        queries: usize,
    ) -> HashMap<&str, Vec<(usize, f64, usize)>> {
        let mut results: HashMap<&str, Vec<(usize, f64, usize)>> = HashMap::new();

        let start = Instant::now();
        let queries = &gen_queries(queries);
        let mut vals = gen_vals(1 << stop_pow2, false);
        vals[0] = MAX;
        info!("Generating took {:?}", start.elapsed());

        for p in start_pow2..=stop_pow2 {
            let size = 1 << p;
            info!("Benchmarking size {}", size);
            let len = size / std::mem::size_of::<u32>();
            // Sort the fist size elements of vals.
            let start = Instant::now();
            vals[..len].radix_sort_unstable();
            info!("Sorting took {:?}", start.elapsed());

            // let bs = &mut BinarySearch::new(vals[..len].to_vec());

            // for &f in &self.bs {
            //     let c0 = bs.cnt;
            //     let t = bench(bs, f, queries);
            //     results.entry(f.0).or_default().push((size, t, bs.cnt - c0));
            // }

            // let eyt = &mut Eytzinger::new(vals[..len].to_vec());

            // for &f in &self.eyt {
            //     let c0 = eyt.cnt;
            //     let t = bench(eyt, f, queries);
            //     results
            //         .entry(f.0)
            //         .or_default()
            //         .push((size, t, eyt.cnt - c0));
            // }

            // let bt = &mut BTree16::new(vals[..len].to_vec());

            // for &f in &self.bt {
            //     let c0 = bt.cnt;
            //     let t = bench(bt, f, queries);
            //     results.entry(f.0).or_default().push((size, t, bt.cnt - c0));
            // }

            // let bp = &mut BpTree16::new(vals[..len].to_vec());

            // for &f in &self.bp {
            //     let c0 = bp.cnt;
            //     let t = bench(bp, f, queries);
            //     results.entry(f.0).or_default().push((size, t, bp.cnt - c0));
            // }

            // let f: BFn<4, _> = ("bp_batch4", BpTree16::batch::<4>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<8, _> = ("bp_batch8", BpTree16::batch::<8>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<16, _> = ("bp_batch16", BpTree16::batch::<16>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<32, _> = ("bp_batch32", BpTree16::batch::<32>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<64, _> = ("bp_batch64", BpTree16::batch::<64>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<128, _> = ("bp_batch128", BpTree16::batch::<128>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<4, _> = ("bp_batch_ptr4", BpTree16::batch_ptr::<4>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<8, _> = ("bp_batch_ptr8", BpTree16::batch_ptr::<8>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<16, _> = ("bp_batch_ptr16", BpTree16::batch_ptr::<16>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<32, _> = ("bp_batch_ptr32", BpTree16::batch_ptr::<32>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<64, _> = ("bp_batch_ptr64", BpTree16::batch_ptr::<64>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<128, _> = ("bp_batch_ptr128", BpTree16::batch_ptr::<128>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let bp = &mut BpTree16::new(vals[..len].to_vec());

            // let f: BFn<128, _> = ("bp_batch_prefetch", BpTree16::batch_prefetch::<128>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));
            // let f: BFn<128, _> = ("bp_batch_ptr", BpTree16::batch_ptr::<128>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));
            // let f: BFn<128, _> = ("bp_batch_ptr2", BpTree16::batch_ptr2::<128>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));
            // let f: BFn<128, _> = ("bp_batch_ptr3", BpTree16::batch_ptr3::<128, false>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));
            // let f: BFn<128, _> = ("bp_batch_ptr3_last", BpTree16::batch_ptr3::<128, true>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            let bp = &mut BpTree16R::new(vals[..len].to_vec());

            let f: BFn<128, _> = ("bp_batch_ptr3_rev", BpTree16R::batch_ptr3::<128, false>);
            let t = bench_batch(bp, f, queries);
            results.entry(f.0).or_default().push((size, t, 0));
            let f: BFn<128, _> = ("bp_batch_ptr3_rev_last", BpTree16R::batch_ptr3::<128, true>);
            let t = bench_batch(bp, f, queries);
            results.entry(f.0).or_default().push((size, t, 0));

            // let f: BFn<32, _> = ("bp_batch_ptr3_32", BpTree16::batch_ptr3::<32>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));
            // let f: BFn<64, _> = ("bp_batch_ptr3_64", BpTree16::batch_ptr3::<64>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));
            // let f: BFn<256, _> = ("bp_batch_ptr3_256", BpTree16::batch_ptr3::<256>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));

            // let bp = &mut BpTree15::new(vals[..len].to_vec());

            // let f: BFn<128, _> = ("bp15_batch_prefetch", BpTree15::batch_prefetch::<128>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));
            // let f: BFn<128, _> = ("bp15_batch_ptr", BpTree15::batch_ptr::<128>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));
            // let f: BFn<128, _> = ("bp15_batch_ptr2", BpTree15::batch_ptr2::<128>);
            // let t = bench_batch(bp, f, queries);
            // results.entry(f.0).or_default().push((size, t, 0));
        }
        results
    }
}

#[pymodule]
fn sa_layout(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BenchmarkSortedArray>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::BenchmarkSortedArray;

    #[test]
    fn test_benchmarks() {
        let benchmark = BenchmarkSortedArray::new();
        let correct = benchmark.test_all();
        assert!(correct);
    }
}
