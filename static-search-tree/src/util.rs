use std::{hint::black_box, sync::LazyLock, time::Instant};

use itertools::Itertools;
use log::info;
use rand::Rng;
use rdst::RadixSort;

use crate::{node::MAX, SearchIndex, SearchScheme};

pub type Seq = [u8];
pub type Sequence = Vec<u8>;

const LOWEST_GENERATED: u32 = 0;
const HIGHEST_GENERATED: u32 = 1 << 30;

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

/// Prefetch the given cacheline into L1 cache.
pub fn prefetch_index<T>(s: &[T], index: usize) {
    let ptr = unsafe { s.as_ptr().add(index) as *const u64 };
    prefetch_ptr(ptr);
}

/// Prefetch the given cacheline into L1 cache.
pub fn prefetch_ptr<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // TODO: Put this behind a feature flag.
        // std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_LOCALITY3);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        // Do nothing.
    }
}

pub fn time<T>(t: &str, f: impl FnOnce() -> T) -> T {
    eprintln!("{t}: Starting");
    let start = std::time::Instant::now();
    let r = f();
    let elapsed = start.elapsed();
    eprintln!("{t}: Elapsed: {:?}", elapsed);
    r
}

pub fn bench_scheme<I: SearchIndex>(
    index: &I,
    scheme: &dyn SearchScheme<INDEX = I>,
    qs: &[u32],
) -> f64 {
    info!("Benching {}", scheme.name());
    let start = Instant::now();
    black_box(index.query(qs, &scheme));
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / qs.len() as f64
}

pub fn bench_scheme_par<I: SearchIndex + Sync>(
    index: &I,
    scheme: &dyn SearchScheme<INDEX = I>,
    qs: &[u32],
    threads: usize,
) -> f64 {
    info!("Benching {}", scheme.name());
    let chunk_size = qs.len().div_ceil(threads);
    let start = Instant::now();

    rayon::scope(|scope| {
        for idx in 0..threads {
            let index = &index;
            let scheme = &scheme;
            scope.spawn(move |_| {
                let start_idx = idx * chunk_size;
                let end = ((idx + 1) * chunk_size).min(qs.len());
                let qs_thread = &qs[start_idx..end];
                black_box(index.query(qs_thread, scheme));
            });
        }
    });

    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / qs.len() as f64
}

fn init_trace() {
    stderrlog::new()
        .color(stderrlog::ColorChoice::Always)
        .verbosity(5)
        .show_level(true)
        .init()
        .unwrap();
}

pub static INIT_TRACE: LazyLock<()> = LazyLock::new(init_trace);
