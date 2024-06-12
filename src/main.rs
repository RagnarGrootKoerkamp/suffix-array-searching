use std::{iter::zip, ops::Range};

use clap::Parser;
use rand::Rng;

type Seq = [u8];
type Sequence = Vec<u8>;

/// Generate a random text of length n.
pub fn random_string(n: usize) -> Sequence {
    let mut rng = rand::thread_rng();
    let mut seq = Vec::with_capacity(n);
    for _ in 0..n {
        seq.push(rng.gen_range(0..4));
    }
    seq
}

/// Generate a random subset of queries.
pub fn samples(t: &Seq, n: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(0..t.len())).collect()
}

#[allow(unused)]
pub trait Search<'t> {
    /// Build a suffix array on the given text.
    fn build(t: &'t Seq) -> Self;

    /// Find position of the smallest suffix >= q.
    fn search(&self, q: &Seq) -> usize;

    /// Iterate over all suffixes that have q as a prefix.
    fn search_prefix(&self, q: &Seq) -> impl Iterator<Item = usize> {
        unimplemented!();
        0..0
    }

    /// Iterate over all suffixes in the range.
    fn search_range(&self, r: Range<&Seq>) -> impl Iterator<Item = usize> {
        unimplemented!();
        0..0
    }
}

pub struct SaNaive<'a> {
    t: &'a Seq,
    sa: Vec<u32>,
}

impl<'t> SaNaive<'t> {
    pub fn build(t: &'t Seq) -> Self {
        assert!(t.len() < std::u32::MAX as usize);
        let mut sa: Vec<_> = (0..t.len() as _).collect();
        sa.sort_by_key(|&a| &t[a as usize..]);
        Self { t, sa }
    }

    pub fn suffix(&self, i: usize) -> &Seq {
        &self.t[self.sa[i] as usize..]
    }
}
pub fn binary_search(sa: &SaNaive, q: &Seq) -> usize {
    let mut l = 0;
    let mut r = sa.sa.len();
    while l < r {
        let m = (l + r) / 2;
        if sa.suffix(m) < q {
            l = m + 1;
        } else {
            r = m;
        }
    }
    sa.sa[l] as usize
}

pub fn branchy_search(sa: &SaNaive, q: &Seq) -> usize {
    let mut l = 0;
    let mut r = sa.sa.len();
    while l < r {
        let m = (l + r) / 2;
        let t = sa.suffix(m);
        if t < q {
            l = m + 1;
        } else if t > q {
            r = m;
        } else {
            return m;
        }
    }
    sa.sa[l] as usize
}

pub fn branchfree_search(sa: &SaNaive, q: &Seq) -> usize {
    let mut l = 0;
    let mut n = sa.sa.len();
    while n > 1 {
        let half = n / 2;
        l = if sa.suffix(l + half) < q { l + half } else { l };
        n -= half;
    }
    sa.sa[l + if sa.suffix(l) < q { 1 } else { 0 }] as usize
}

pub fn interpolation_search(sa: &SaNaive, q: &Seq) -> usize {
    let mut l = 0;
    let mut n = sa.sa.len();
    while n > 1 {
        let half = n / 2;
        l = if sa.suffix(l + half) < q { l + half } else { l };
        n -= half;
    }
    sa.sa[l + if sa.suffix(l) < q { 1 } else { 0 }] as usize
}

fn bench(sa: &SaNaive, samples: &[usize], name: &str, f: &fn(&SaNaive, &Seq) -> usize) {
    let start = std::time::Instant::now();
    for &i in samples {
        f(sa, &sa.t[i..]);
    }
    let elapsed = start.elapsed();
    let per_query = elapsed / samples.len() as u32;
    let layers = (sa.sa.len() as f32).log2() as u32;
    let per_branch = elapsed / layers;
    println!("{name:>20}: {elapsed:6.3?} {per_query:6.3?} {per_branch:6.3?}");
}

#[derive(Parser)]
struct Args {
    #[clap(short, default_value_t = 1000000)]
    n: usize,
    #[clap(short, default_value_t = 100000)]
    s: usize,
}

fn main() {
    let args = Args::parse();
    let t = random_string(args.n);
    let sa = SaNaive::build(&t);

    let samples = samples(&t, args.s);

    let funcs = [binary_search, branchy_search, branchfree_search];
    let names = ["binary_search", "branchy_search", "branchfree_search"];

    for (name, f) in zip(names.iter(), funcs.iter()) {
        bench(&sa, &samples, name, f);
    }
}
