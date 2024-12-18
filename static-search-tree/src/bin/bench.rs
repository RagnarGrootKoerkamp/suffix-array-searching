#![feature(generic_arg_infer)]

use clap::Parser;
use rdst::RadixSort;
use static_search_tree::{
    binary_search::SortedVec,
    eytzinger::Eytzinger,
    node::BTreeNode,
    s_tree::STree16,
    util::{gen_queries, gen_vals},
    SearchIndex, SearchScheme,
};
use std::{
    any::type_name,
    hint::black_box,
    path::Path,
    sync::LazyLock,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    #[clap(short, long)]
    from: Option<usize>,
    #[clap(short, long)]
    to: Option<usize>,
    #[clap(short, long)]
    release: bool,
    #[clap(short, long)]
    dense: bool,
}

static ARGS: LazyLock<Args> = LazyLock::new(|| Args::parse());

fn main() {
    let mut results = vec![];

    let runs = if ARGS.release { 3 } else { 1 };
    let sizes = sizes();
    let queries = if ARGS.release { 1_000_000 } else { 100_000 };

    for run in 0..runs {
        // Setup

        let qs = &gen_queries(queries);
        let mut vals = gen_vals(*sizes.last().unwrap(), false);

        for &size in &sizes {
            let len = size / std::mem::size_of::<u32>();
            let vals = &mut vals[..len];
            vals.radix_sort_unstable();

            fn run_exps<I: SearchIndex, const N: usize>(
                results: &mut Vec<Result>,
                size: usize,
                vals: &[u32],
                qs: &Vec<u32>,
                run: usize,
                exps: [&(dyn SearchScheme<I>); N],
            ) {
                let index = I::new(vals);
                for exp in exps {
                    results.push(Result::new(size, &index, qs, exp, run));
                }
            }

            /// Wrapper type for the cast to &dyn.
            type T<I, const N: usize> = [&'static dyn SearchScheme<I>; N];

            let exps: T<_, _> = [&SortedVec::binary_search_std];
            run_exps(&mut results, size, vals, qs, run, exps);

            let exps: T<_, _> = [&Eytzinger::search_prefetch::<4>];
            run_exps(&mut results, size, vals, qs, run, exps);

            let exps: T<_, _> = const {
                [
                    // &STree16::search,
                    &STree16::search_with_find(BTreeNode::find_linear) as _,
                    &STree16::search_with_find(BTreeNode::find_linear_count) as _,
                    // &STree16::search_with_find(BTreeNode::find_split) as _,
                    &STree16::search_with_find(BTreeNode::find_ctz) as _,
                    &STree16::search_with_find(BTreeNode::find_ctz_signed) as _,
                    &STree16::search_with_find(BTreeNode::find_popcnt_portable) as _,
                    &STree16::search_with_find(BTreeNode::find_popcnt) as _,
                ]
            };
            run_exps(&mut results, size, vals, qs, run, exps);
        }

        save_results(&results, "results");
        eprintln!("Saved results after run {}", run + 1);
    }
}

/// Return an iterator over sizes to iterate over.
/// Starts at 32B and goes up to ~1GB.
pub fn sizes() -> Vec<usize> {
    let mut v = vec![];
    let from = ARGS.from.unwrap_or(5);
    let release = ARGS.release;
    let dense = release || ARGS.dense;
    let to = ARGS.to.unwrap_or(if release { 30 } else { 28 });
    for b in from..to {
        let base = 1 << b;
        v.push(base);
        if dense {
            v.push(base * 5 / 4);
            v.push(base * 3 / 2);
            v.push(base * 7 / 4);
        }
    }
    v.push(1 << to);
    v
}

pub fn save_results(results: &Vec<Result>, name: &str) {
    let dir = Path::new("results").to_owned();
    std::fs::create_dir_all(&dir).unwrap();
    let f = if ARGS.release {
        dir.join(&format!("{name}-release"))
    } else {
        dir.join(name)
    };
    let f = f.with_extension("json");
    let f = std::fs::File::create(f).unwrap();
    serde_json::to_writer(f, &results).unwrap();
}

#[derive(serde::Serialize)]
pub struct Result {
    /// Index name
    pub index: String,
    /// SearchScheme name
    pub scheme: String,
    /// Input size in bytes.
    pub size: usize,
    /// Number of iterations.
    pub queries: usize,
    /// Id
    pub run: usize,
    /// Total duration of the experiment.
    pub duration: Duration,
    /// Latency (or reverse throughput) of each operation, in nanoseconds.
    pub latency: f64,
    /// Number of clock cycles per operation.
    pub cycles: f64,
    /// CPU frequency in Hz.
    pub freq: f64,
}

impl Result {
    pub fn new<I: SearchIndex>(
        size: usize,
        index: &I,
        qs: &[u32],
        scheme: &dyn SearchScheme<I>,
        run: usize,
    ) -> Result {
        let start = Instant::now();
        black_box(index.query(qs, scheme));
        let duration = start.elapsed();
        let queries = qs.len();
        let freq = get_cpu_freq().unwrap();
        let latency = duration.as_nanos() as f64 / queries as f64;
        let cycles = latency / 1000000000. * freq;

        let sz = size::Size::from_bytes(size);
        let sz = format!("{}", sz);

        println!("n = {sz:>8}, s/it: {latency:>6.2?}ns cycles/it: {cycles:>7.2} freq: {freq:>10}",);
        Result {
            index: type_name::<I>().to_string(),
            scheme: scheme.name().to_string(),
            size,
            queries,
            run,
            duration,
            latency,
            cycles,
            freq,
        }
    }
}

/// Return the current CPU frequency in Hz.
pub(crate) fn get_cpu_freq() -> Option<f64> {
    let cur_cpu = get_cpu()?;
    let path = format!("/sys/devices/system/cpu/cpu{cur_cpu}/cpufreq/scaling_cur_freq");
    let path = &Path::new(&path);
    if !path.exists() {
        return None;
    }

    let val = std::fs::read_to_string(path).ok()?;
    Some(val.trim().parse::<f64>().ok()? * 1000.)
}

pub(crate) fn get_cpu() -> Option<i32> {
    #[cfg(not(target_os = "macos"))]
    {
        Some(unsafe { libc::sched_getcpu() })
    }
    #[cfg(target_os = "macos")]
    {
        None
    }
}
