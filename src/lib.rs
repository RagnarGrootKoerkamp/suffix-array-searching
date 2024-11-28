#![allow(unused)]
#![feature(array_chunks, portable_simd, core_intrinsics)]
pub mod experiments_sorted_arrays;
pub mod sa_search;
pub mod util;
pub use util::*;

pub mod py {
    use crate::experiments_sorted_arrays::VanillaBinSearch;

    use super::*;
    use experiments_sorted_arrays;
    use pyo3::prelude::*;
    use rand::Rng;
    use std::collections::HashMap;
    const LOWEST_GENERATED: u32 = 0;
    const HIGHEST_GENERATED: u32 = 4200000000;

    #[pyclass]
    struct BenchmarkSortedArray {
        func_map: HashMap<&'static str, experiments_sorted_arrays::VanillaBinSearch>,
        // TODO: preprocess_map
        preprocess_map: HashMap<&'static str, experiments_sorted_arrays::PreprocessArray>,
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
            }
        }

        fn _bench(&self, size: usize, repetitions: usize, fname: &str) -> (f64, f64) {
            let mut array = gen_random_array(size, LOWEST_GENERATED, HIGHEST_GENERATED);
            if self.preprocess_map.contains_key(fname) {
                array = (self.preprocess_map[fname])(array);
            }
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
                results += func(&array, searched_values[i], &mut cnt);
            }
            let elapsed = start.elapsed();
            // FIXME: this is ugly
            (
                elapsed.as_nanos() as f64 / repetitions as f64,
                cnt as f64 / repetitions as f64,
            )
        }

        fn test(&self, fname: &str, start_pow2: usize, stop_pow2: usize) -> bool {
            false
        }

        fn benchmark(
            &self,
            fname: &str,
            start_pow2: usize,
            stop_pow2: usize,
            repetitions: usize,
        ) -> (Vec<f64>, Vec<f64>) {
            let mut times = Vec::new();
            let mut comp_cnts = Vec::new();
            for p in start_pow2..stop_pow2 {
                let size = 2usize.pow(p as u32);
                let (timing, cnt) = self._bench(size, repetitions, fname);
                comp_cnts.push(cnt);
                times.push(timing);
            }
            (times, comp_cnts)
        }
    }

    #[pymodule]
    fn sa_layout(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<BenchmarkSortedArray>()?;
        Ok(())
    }
}
