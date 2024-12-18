use crate::binary_search::{
    BinarySearch, BinarySearchBranchless, BinarySearchBranchlessPrefetch, SortedVec,
};
use crate::bplustree::{BpTree, BpTree15, BpTree16, BpTree16R};
use crate::btree::{BTree16, BTreeSearch, BTreeSearchLoop, BTreeSearchSimd};
use crate::eytzinger::{Eytzinger, EytzingerPrefetch, EytzingerSearch};
use crate::interp_search::InterpolationSearch;
use crate::node::MAX;
use crate::{util::*, SearchIndex, SearchScheme};
use log::info;
use pyo3::prelude::*;
use rdst::RadixSort;
use std::collections::HashMap;
use std::time::Instant;

#[pyclass]
pub struct SearchFunctions {
    bs: Vec<&'static dyn SearchScheme<INDEX = SortedVec>>,
    eyt: Vec<&'static dyn SearchScheme<INDEX = Eytzinger>>,
    bt: Vec<&'static dyn SearchScheme<INDEX = BTree16>>,
    bp: Vec<&'static dyn SearchScheme<INDEX = BpTree16>>,
    bp15: Vec<&'static dyn SearchScheme<INDEX = BpTree15>>,
    bpr: Vec<&'static dyn SearchScheme<INDEX = BpTree16R>>,
}

#[pymethods]
impl SearchFunctions {
    #[new]
    pub fn new() -> Self {
        *INIT_TRACE;

        let bs = vec![
            &BinarySearch as &dyn SearchScheme<INDEX = _>,
            &BinarySearchBranchless,
            &BinarySearchBranchlessPrefetch,
            &InterpolationSearch,
        ];
        let eyt = vec![
            &EytzingerSearch as &dyn SearchScheme<INDEX = _>,
            &EytzingerPrefetch::<4>,
            &EytzingerPrefetch::<8>,
            &EytzingerPrefetch::<16>,
        ];
        let bt = vec![
            &BTreeSearch as &dyn SearchScheme<INDEX = _>,
            &BTreeSearchLoop,
            &BTreeSearchSimd,
        ];
        let bp = const {
            [
                &BpTree::search() as &dyn SearchScheme<INDEX = _>,
                &BpTree::search_split(),
                &BpTree::search_batch::<4>(),
                &BpTree::search_batch::<8>(),
                &BpTree::search_batch::<16>(),
                &BpTree::search_batch::<32>(),
                &BpTree::search_batch::<64>(),
                &BpTree::search_batch::<128>(),
                &BpTree::search_batch_prefetch::<128>(),
                &BpTree::search_batch_ptr::<128>(),
                &BpTree::search_batch_ptr2::<128>(),
                &BpTree::search_batch_ptr3::<128, false>(),
                &BpTree::search_batch_no_prefetch::<64, false, 1>(),
                &BpTree::search_batch_no_prefetch::<64, false, 2>(),
                &BpTree::search_interleave::<64, false>(),
                // &BpTree::search_batch_ptr3::<128, true>(),
                // &BpTree::search_interleave::<64, true>(),
            ]
        }
        .to_vec();

        let bp15 = const {
            [
                &BpTree::search() as &dyn SearchScheme<INDEX = _>,
                &BpTree::search_split(),
                &BpTree::search_batch::<128>(),
                &BpTree::search_batch_prefetch::<128>(),
                &BpTree::search_batch_ptr::<128>(),
                &BpTree::search_batch_ptr2::<128>(),
                &BpTree::search_batch_ptr3::<128, false>(),
                &BpTree::search_batch_no_prefetch::<128, false, 1>(),
                &BpTree::search_batch_no_prefetch::<128, false, 2>(),
                &BpTree::search_interleave::<64, false>(),
                // &BpTree::search_batch_ptr3::<128, true>(),
                // &BpTree::search_interleave::<64, true>(),
            ]
        }
        .to_vec();

        let bpr = const {
            [
                // &BpTree::search() as &dyn SearchScheme<INDEX = _>,
                // &BpTree::search_split(),
                // &BpTree::search_batch::<128>(),
                // &BpTree::search_batch_prefetch::<128>(),
                // &BpTree::search_batch_ptr::<128>(),
                // &BpTree::search_batch_ptr2::<128>(),
                &BpTree::search_batch_ptr3::<32, false>() as &dyn SearchScheme<INDEX = _>,
                &BpTree::search_batch_ptr3::<64, false>(),
                &BpTree::search_batch_ptr3::<128, false>(),
                &BpTree::search_batch_ptr3::<256, false>(),
                &BpTree::search_batch_no_prefetch::<128, false, 1>(),
                &BpTree::search_batch_no_prefetch::<128, false, 2>(),
                &BpTree::search_interleave::<64, false>(),
                // &BpTree::search_batch_ptr3::<128, true>(),
                // &BpTree::search_interleave::<64, true>(),
            ]
        }
        .to_vec();

        SearchFunctions {
            bs,
            eyt,
            bt,
            bp,
            bp15,
            bpr,
        }
    }

    pub fn benchmark_one(
        &self,
        fname: String,
        start_pow2: usize,
        stop_pow2: usize,
        queries: usize,
    ) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
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
            let vals = &mut vals[..len];
            vals.radix_sort_unstable();
            info!("Sorting took {:?}", start.elapsed());

            fn map<I: SearchIndex>(
                schemes: &Vec<&dyn SearchScheme<INDEX = I>>,
                fname: &str,
                vals: &[u32],
                qs: &[u32],
                results: &mut Vec<(usize, f64)>,
            ) {
                for &scheme in schemes {
                    if scheme.name() == fname {
                        let index = &I::new(vals);
                        let t = bench_scheme(index, scheme, qs);
                        results.push((vals.len(), t));
                    }
                }
            }

            map(&self.bs, &fname, &vals, queries, &mut results);
            map(&self.eyt, &fname, &vals, queries, &mut results);
            map(&self.bt, &fname, &vals, queries, &mut results);
            map(&self.bp, &fname, &vals, queries, &mut results);
            map(&self.bp15, &fname, &vals, queries, &mut results);
            map(&self.bpr, &fname, &vals, queries, &mut results);
        }
        if results.len() > (stop_pow2 - start_pow2 + 1) {
            panic!("The function with the same name must exist multiple times!")
        }
        results
    }

    fn benchmark(
        &self,
        start_pow2: usize,
        stop_pow2: usize,
        queries: usize,
    ) -> HashMap<&str, Vec<(usize, f64)>> {
        type Results = HashMap<&'static str, Vec<(usize, f64)>>;
        let mut results: Results = HashMap::new();

        let start = Instant::now();
        let qs = &gen_queries(queries);
        let mut vals = gen_vals(1 << stop_pow2, false);
        vals[0] = MAX;
        info!("Generating took {:?}", start.elapsed());
        for p in start_pow2..=stop_pow2 {
            let size = 1 << p;
            info!("Benchmarking size {}", size);
            let len = size / std::mem::size_of::<u32>();
            // Sort the fist size elements of vals.
            let start = Instant::now();
            let vals = &mut vals[..len];
            vals.radix_sort_unstable();
            info!("Sorting took {:?}", start.elapsed());

            // Helper to extract type `I` and build the index.
            fn map<I: SearchIndex>(
                schemes: &Vec<&dyn SearchScheme<INDEX = I>>,
                vals: &[u32],
                qs: &[u32],
                size: usize,
                results: &mut Results,
            ) {
                map_idx(schemes, &I::new(vals), qs, size, results);
            }

            fn map_idx<I: SearchIndex>(
                schemes: &Vec<&(dyn SearchScheme<INDEX = I>)>,
                index: &I,
                qs: &[u32],
                size: usize,
                results: &mut Results,
            ) {
                for &scheme in schemes {
                    let t = bench_scheme(index, scheme, qs);
                    results.entry(scheme.name()).or_default().push((size, t));
                }
            }

            let results = &mut results;
            map(&self.bs, &vals, qs, size, results);
            map(&self.eyt, &vals, qs, size, results);
            map(&self.bt, &vals, qs, size, results);
            map(&self.bp, &vals, qs, size, results);
            map(&self.bp15, &vals, qs, size, results);
            map(&self.bpr, &vals, qs, size, results);
            map_idx(
                &self.bpr,
                &BpTree16R::new_fwd(&vals, false),
                qs,
                size,
                results,
            );
            map_idx(
                &self.bpr,
                &BpTree16R::new_fwd(&vals, true),
                qs,
                size,
                results,
            );

            let bpr = BpTree16R::new_fwd(&vals, false);
            let strings = ["", "t1", "t2", "t3", "t4", "t5", "t6"];
            for threads in 1..=6 {
                let scheme = BpTree::search_batch_no_prefetch::<128, false, 1>();
                let t = bench_scheme_par(&bpr, &scheme, qs, threads);
                results.entry(strings[threads]).or_default().push((size, t));
            }
        }
        results
    }
}

#[pymodule]
fn sa_layout(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchFunctions>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use std::any::type_name;

    use super::*;

    #[test]
    fn test_search() {
        let fs = SearchFunctions::new();

        const TEST_START_POW2: usize = 6;
        const TEST_END_POW2: usize = 26;
        const TEST_QUERIES: usize = 10000;

        let correct = &mut true;
        for pow2 in TEST_START_POW2..=TEST_END_POW2 {
            let size = 2usize.pow(pow2 as u32);
            let vals = gen_vals(size, true);
            eprintln!("LEN: {}", vals.len());
            let qs = &gen_queries(TEST_QUERIES.next_multiple_of(256));

            let results = &mut vec![];

            // Helper to extract type `I` and build the index.
            fn map<I: SearchIndex>(
                schemes: &Vec<&dyn SearchScheme<INDEX = I>>,
                vals: &[u32],
                qs: &[u32],
                results: &mut Vec<u32>,
                correct: &mut bool,
            ) {
                eprintln!("Building index for {:?}", type_name::<I>());
                map_idx(schemes, &I::new(vals), qs, results, correct);
            }

            fn map_idx<I: SearchIndex>(
                schemes: &Vec<&(dyn SearchScheme<INDEX = I>)>,
                index: &I,
                qs: &[u32],
                results: &mut Vec<u32>,
                correct: &mut bool,
            ) {
                for &scheme in schemes {
                    eprintln!("Testing scheme {:?}", scheme.name());
                    let new_results = index.query(qs, scheme);
                    if new_results.is_empty() {
                        continue;
                    }
                    if results.is_empty() {
                        *results = new_results;
                    } else {
                        if *results != new_results {
                            eprintln!("Expected\n{results:?}\ngot\n{new_results:?}");
                            *correct = false;
                        }
                    }
                }
            }

            map(&fs.bs, &vals, qs, results, correct);
            map(&fs.eyt, &vals, qs, results, correct);
            map(&fs.bt, &vals, qs, results, correct);
            map(&fs.bp, &vals, qs, results, correct);
            map(&fs.bp15, &vals, qs, results, correct);
            map(&fs.bpr, &vals, qs, results, correct);
            eprintln!(
                "Building index for {:?} fwd(false)",
                type_name::<BpTree16R>()
            );
            map_idx(
                &fs.bpr,
                &BpTree16R::new_fwd(&vals, false),
                qs,
                results,
                correct,
            );
            eprintln!(
                "Building index for {:?} fwd(true)",
                type_name::<BpTree16R>()
            );
            map_idx(
                &fs.bpr,
                &BpTree16R::new_fwd(&vals, true),
                qs,
                results,
                correct,
            );
            assert!(*correct);
        }
    }
}
