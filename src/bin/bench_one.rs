#![feature(portable_simd)]
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sa_layout::BenchmarkSortedArray;
use sa_layout::*;
use std::{iter::repeat, path::PathBuf};
use tracing::{debug, info};

#[derive(Parser)]
struct Args {
    #[clap(short)]
    n: usize,

    #[clap(long)]
    fname: String,
}

fn main() {
    let args = Args::parse();
    let bench = BenchmarkSortedArray::new();
    println!("{} {}", args.n, args.fname);
    let results = bench.benchmark_one(args.fname, 4, 12, 50);
    println!("{:?}", results);
}
