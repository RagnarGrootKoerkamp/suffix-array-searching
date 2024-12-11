#![feature(array_chunks, portable_simd)]

pub mod btree;
pub mod experiments_sorted_arrays;
pub mod sa_search;
pub mod util;

use btree::BTree16;
use experiments_sorted_arrays::{BinarySearch, Eytzinger};
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

type Fn<T> = (&'static str, fn(&mut T, u32) -> u32);

fn run<T>(searcher: &mut T, search: Fn<T>, queries: &[u32]) -> Vec<u32> {
    queries
        .into_iter()
        .map(|q| search.1(searcher, *q))
        .collect()
}

fn bench<T>(searcher: &mut T, search: Fn<T>, queries: &[u32]) -> f64 {
    info!("Benching {}", search.0);
    let start = Instant::now();
    for q in queries {
        black_box(search.1(searcher, *q));
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / queries.len() as f64
}

fn gen_queries(n: usize) -> Vec<u32> {
    (0..n)
        .map(|_| rand::thread_rng().gen_range(LOWEST_GENERATED..HIGHEST_GENERATED))
        .collect()
}

/// Generate a u32 array of the given *size* in bytes, and ending in u32::MAX.
fn gen_vals(size: usize, sort: bool) -> Vec<u32> {
    let n = size / std::mem::size_of::<u32>();
    // TODO: generate a new array
    let mut vals = (0..n - 1)
        .map(|_| rand::thread_rng().gen_range(LOWEST_GENERATED..HIGHEST_GENERATED))
        .collect_vec();
    vals.push(u32::MAX);
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
}

impl BenchmarkSortedArray {
    #[allow(unused)]
    fn test_all(&self) -> bool {
        const TEST_START_POW2: usize = 3;
        const TEST_END_POW2: usize = 20;
        const TEST_QUERIES: usize = 10000;

        let mut correct = true;
        for pow2 in TEST_START_POW2..TEST_END_POW2 + 1 {
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
                        eprintln!("{} failed", f.0);
                    }
                }
            }

            let eyt = &mut Eytzinger::new(vals.clone());

            for &f in &self.eyt {
                let new_results = run(eyt, f, queries);
                if results != new_results {
                    correct = false;
                    eprintln!("{} failed", f.0);
                }
            }

            let bt = &mut BTree16::new(vals);

            for &f in &self.bt {
                let new_results = run(bt, f, queries);
                if results != new_results {
                    correct = false;
                    eprintln!("{} failed", f.0);
                }
            }
        }
        correct
    }
}

#[pymethods]
impl BenchmarkSortedArray {
    #[new]
    fn new() -> Self {
        *INIT_TRACE;

        let bs: Vec<(&'static str, fn(&mut BinarySearch, u32) -> u32)> = vec![
            ("bs_search", BinarySearch::search),
            ("bs_branchless", BinarySearch::search_branchless),
            (
                "bs_branchless_prefetch",
                BinarySearch::search_branchless_prefetch,
            ),
        ];
        let eyt: Vec<(&'static str, fn(&mut Eytzinger, u32) -> u32)> = vec![
            // ("eyt_search", Eytzinger::search),
            ("eyt_prefetch_4", Eytzinger::search_prefetch::<4>),
            ("eyt_prefetch_8", Eytzinger::search_prefetch::<8>),
            ("eyt_prefetch_16", Eytzinger::search_prefetch::<16>),
        ];
        let bt: Vec<(&'static str, fn(&mut BTree16, u32) -> u32)> = vec![
            ("bt_search", BTree16::search),
            ("bt_loop", BTree16::search_loop),
            ("bt_simd", BTree16::search_simd),
        ];

        BenchmarkSortedArray { bs, eyt, bt }
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
        vals[0] = u32::MAX;
        info!("Generating took {:?}", start.elapsed());

        for p in start_pow2..=stop_pow2 {
            let size = 1 << p;
            info!("Benchmarking size {}", size);
            let len = size / std::mem::size_of::<u32>();
            // Sort the fist size elements of vals.
            let start = Instant::now();
            vals[..len].radix_sort_unstable();
            info!("Sorting took {:?}", start.elapsed());

            let bs = &mut BinarySearch::new(vals[..len].to_vec());

            for &f in &self.bs {
                let c0 = bs.cnt;
                let t = bench(bs, f, queries);
                results.entry(f.0).or_default().push((size, t, bs.cnt - c0));
            }

            let eyt = &mut Eytzinger::new(vals[..len].to_vec());

            for &f in &self.eyt {
                let c0 = bs.cnt;
                let t = bench(eyt, f, queries);
                results.entry(f.0).or_default().push((size, t, bs.cnt - c0));
            }

            let bt = &mut BTree16::new(vals[..len].to_vec());

            for &f in &self.bt {
                let c0 = bs.cnt;
                let t = bench(bt, f, queries);
                results.entry(f.0).or_default().push((size, t, bs.cnt - c0));
            }
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
