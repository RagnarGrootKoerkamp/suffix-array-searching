#![feature(portable_simd)]
use clap::Parser;
use sa_layout::BenchmarkSortedArray;

#[derive(Parser)]
struct Args {
    #[clap(long)]
    start: usize,

    #[clap(long)]
    stop: usize,

    #[clap(long)]
    iters: usize,

    #[clap(long)]
    fname: String,
}

fn main() {
    let args = Args::parse();
    let bench = BenchmarkSortedArray::new();
    let results = bench.benchmark_one(args.fname, args.start, args.stop, args.iters);
    for (size, timing) in results {
        println!("{} {}", size, timing);
    }
}
