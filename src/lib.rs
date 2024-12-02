#![allow(unused)]
#![feature(array_chunks, portable_simd, core_intrinsics)]
pub mod experiments_sorted_arrays;
pub mod sa_search;
pub mod util;
pub use util::*;

fn vec_eq(va: &Vec<u32>, vb: &Vec<u32>) -> bool {
    (va.len() == vb.len()) &&  // zip stops at the shortest
     va.iter()
       .zip(vb)
       .all(|(a,b)| a == b)
}

pub mod py {
    use crate::experiments_sorted_arrays::VanillaBinSearch;

    use super::*;
    use experiments_sorted_arrays;
    use pyo3::prelude::*;
    use rand::Rng;
    use std::collections::HashMap;
    const LOWEST_GENERATED: u32 = 0;
    const HIGHEST_GENERATED: u32 = 4200000000;
    const UPPER_BOUND: u32 = u32::MAX;

    const TEST_START_POW2: usize = 3;
    const TEST_END_POW2: usize = 20;
    const TEST_QUERIES: usize = 10;
    const TRUSTED_FUNCTION: experiments_sorted_arrays::VanillaBinSearch =
        experiments_sorted_arrays::binary_search;

    #[pyclass]
    struct BenchmarkSortedArray {
        func_map: HashMap<&'static str, experiments_sorted_arrays::VanillaBinSearch>,
        preprocess_map: HashMap<&'static str, experiments_sorted_arrays::PreprocessArray>,
        // a workaround for my bad knowledge of Rust;
        // stores names of functions that are a part of the benchmark before they are all run
        to_bench_map: Vec<String>,
    }

    fn gen_random_array(size: usize, min: u32, max: u32) -> Vec<u32> {
        // TODO: generate a new array
        let mut array = Vec::new();
        let mut rng = rand::thread_rng();
        for i in 0..size {
            let num = rng.gen_range(min..max);
            array.push(num);
        }
        array.sort();
        array
    }

    impl BenchmarkSortedArray {
        // returns a vector of values the function found. Useful for comparing outputs of different implementations.
        fn test_one(
            &self,
            func: experiments_sorted_arrays::VanillaBinSearch,
            preprocessed_array: &[u32],
            searched_values: &[u32],
        ) -> Vec<u32> {
            let mut results = Vec::new();
            let mut cnt: usize = 0;
            let repetitions = searched_values.len();
            for i in 0..repetitions {
                let index = func(&preprocessed_array, searched_values[i], &mut cnt);
                if index < preprocessed_array.len() {
                    results.push(preprocessed_array[index]);
                } else {
                    results.push(UPPER_BOUND);
                }
            }
            results
        }

        // FIXME: this is very duplicated code as compared to the code for benchmarking
        fn test_all(&self) -> bool {
            let mut correct = true;
            for pow2 in TEST_START_POW2..TEST_END_POW2 + 1 {
                let size = 2usize.pow(pow2 as u32);
                let array = gen_random_array(size, LOWEST_GENERATED, HIGHEST_GENERATED);
                let mut searched_values = Vec::new();
                for _ in 0..TEST_QUERIES {
                    let query = rand::thread_rng().gen_range(LOWEST_GENERATED..HIGHEST_GENERATED);
                    searched_values.push(query);
                }
                let mut trusted_results = self.test_one(TRUSTED_FUNCTION, &array, &searched_values);
                for (fname, func) in &self.func_map {
                    let mut preprocessed_array = array.clone();
                    if self.preprocess_map.contains_key(&fname as &str) {
                        preprocessed_array =
                            (self.preprocess_map[&fname as &str])(preprocessed_array);
                    }
                    let new_results =
                        self.test_one(func.clone(), &preprocessed_array, &searched_values);
                    // check vector equality
                    if !vec_eq(&trusted_results, &new_results) {
                        eprintln!(
                            "The output of {} differs from the trusted function for size {}!",
                            fname, size
                        );
                        correct = false;
                    }
                }
            }
            correct
        }

        fn bench(&self, preprocessed_array: &[u32], repetitions: usize, fname: &str) -> (f64, f64) {
            let mut timing = std::time::Duration::new(0, 0);
            let mut cnt = 0;
            let mut results = 0;
            let func = self.func_map[fname];
            let mut searched_values = Vec::new();
            // FIXME this is awful
            for i in 0..repetitions {
                let query = rand::thread_rng().gen_range(LOWEST_GENERATED..HIGHEST_GENERATED);
                searched_values.push(query);
            }

            let start = std::time::Instant::now();
            for i in 0..repetitions {
                results += func(&preprocessed_array, searched_values[i], &mut cnt);
            }
            let elapsed = start.elapsed();
            // FIXME: this is ugly
            (
                elapsed.as_nanos() as f64 / repetitions as f64,
                cnt as f64 / repetitions as f64,
            )
        }
    }

    #[pymethods]
    impl BenchmarkSortedArray {
        #[new]
        fn new() -> Self {
            let mut functions: HashMap<&str, experiments_sorted_arrays::VanillaBinSearch> =
                HashMap::new();
            let mut preprocess_map: HashMap<&str, experiments_sorted_arrays::PreprocessArray> =
                HashMap::new();
            functions.insert("basic_binsearch", experiments_sorted_arrays::binary_search);
            functions.insert(
                "basic_binsearch_branchless",
                experiments_sorted_arrays::binary_search_branchless,
            );
            functions.insert(
                "basic_binsearch_branchless_prefetched",
                experiments_sorted_arrays::binary_search_branchless_prefetched,
            );
            functions.insert("eytzinger", experiments_sorted_arrays::eytzinger);
            functions.insert(
                "eytzinger_prefetched",
                experiments_sorted_arrays::eytzinger_prefetched,
            );
            functions.insert(
                "btree_basic_16",
                experiments_sorted_arrays::btree_search::<16>,
            );
            functions.insert(
                "btree_branchless_16",
                experiments_sorted_arrays::btree_search_branchless::<16>,
            );
            preprocess_map.insert("eytzinger", experiments_sorted_arrays::to_eytzinger);
            preprocess_map.insert(
                "eytzinger_prefetched",
                experiments_sorted_arrays::to_eytzinger,
            );
            preprocess_map.insert("btree_basic_16", experiments_sorted_arrays::to_btree::<16>);
            preprocess_map.insert(
                "btree_branchless_16",
                experiments_sorted_arrays::to_btree::<16>,
            );

            BenchmarkSortedArray {
                func_map: functions,
                preprocess_map: preprocess_map,
                to_bench_map: Vec::new(),
            }
        }

        fn add_func_to_bm(&mut self, fname: &str) {
            self.to_bench_map.push(String::from(fname));
        }

        fn benchmark(
            &self,
            start_pow2: usize,
            stop_pow2: usize,
            repetitions: usize,
        ) -> HashMap<&String, (Vec<f64>, Vec<f64>)> {
            let mut returned_timings = HashMap::new();
            for fname in &self.to_bench_map {
                let mut times = Vec::new();
                let mut comp_cnts = Vec::new();
                returned_timings.insert(fname, (times, comp_cnts));
            }

            for p in start_pow2..stop_pow2 {
                let size = 2usize.pow(p as u32);
                let array: Vec<u32> = gen_random_array(size, LOWEST_GENERATED, HIGHEST_GENERATED);
                // TODO: generate array here
                for fname in &self.to_bench_map {
                    let mut preprocessed_array = array.clone();
                    if self.preprocess_map.contains_key(&fname as &str) {
                        preprocessed_array =
                            (self.preprocess_map[&fname as &str])(preprocessed_array);
                    }

                    let (ref mut timings, cnts) = returned_timings.get_mut(&fname).unwrap();
                    let (timing, cnt) = self.bench(&preprocessed_array, repetitions, &fname);
                    cnts.push(cnt);
                    timings.push(timing);
                }
            }
            returned_timings
        }
    }

    #[pymodule]
    fn sa_layout(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<BenchmarkSortedArray>()?;
        Ok(())
    }

    mod test {
        use super::*;

        #[test]
        fn test_benchmarks() {
            let benchmark = BenchmarkSortedArray::new();
            let correct = benchmark.test_all();
            assert!(correct);
        }
    }
}
