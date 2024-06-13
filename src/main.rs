#![feature(array_chunks, portable_simd)]
mod util;

use util::*;

use std::{
    arch::x86_64::_pext_u64,
    cmp::Ordering::{Greater, Less},
    iter::repeat,
    ops::{Index, Range},
    path::PathBuf,
    simd::{cmp::SimdPartialEq, Simd},
    slice::from_raw_parts,
};

use clap::Parser;
use itertools::Itertools;
use log::{debug, info};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

type Seq = [u8];
type Sequence = Vec<u8>;

pub struct SaNaive<'a> {
    t: &'a Seq,
    sa: Vec<u32>,
    // Table on the first p bits.
    // p: usize,
    // table:
}

impl Index<usize> for SaNaive<'_> {
    type Output = u32;

    fn index(&self, i: usize) -> &Self::Output {
        unsafe { self.sa.get_unchecked(i) }
    }
}

impl<'t> SaNaive<'t> {
    pub fn build(t: &'t Seq) -> Self {
        let mut sa = vec![0; t.len() + 100000];
        sais::sais64::parallel::sais(t, &mut sa, None, 5).unwrap();
        sa.resize(t.len(), 0);
        let sa: Vec<u32> = sa.into_iter().map(|x| x as u32).collect();
        for (&x, &y) in sa.iter().tuple_windows() {
            assert!(t[x as usize..] < t[y as usize..]);
        }
        Self { t, sa }
    }

    pub fn suffix(&self, i: usize) -> &Seq {
        self.suffix_at(self[i])
    }
    pub fn suffix_at(&self, i: u32) -> &Seq {
        unsafe { self.t.get_unchecked(i as usize..) }
    }
}

pub fn binary_search(sa: &SaNaive, q: &Seq, cnt: &mut usize) -> usize {
    let mut l = 0;
    let mut r = sa.sa.len();
    while l < r {
        let m = (l + r) / 2;
        *cnt += 1;
        if sa.suffix(m) < q {
            l = m + 1;
        } else {
            r = m;
        }
    }
    sa[l] as usize
}

pub fn binary_search_cmp(sa: &SaNaive, q: &Seq, cnt: &mut usize) -> usize {
    let mut l = 0;
    let mut r = sa.sa.len();
    while l < r {
        let m = (l + r) / 2;
        *cnt += 1;
        let t = sa.suffix(m);
        if cmp(t, q) == Less {
            l = m + 1;
        } else {
            r = m;
        }
    }
    sa[l] as usize
}

pub fn branchy_search(sa: &SaNaive, q: &Seq, cnt: &mut usize) -> usize {
    let mut l = 0;
    let mut r = sa.sa.len();
    while l < r {
        let m = (l + r) / 2;
        *cnt += 1;
        let t = sa.suffix(m);
        if t < q {
            l = m + 1;
        } else if t > q {
            r = m;
        } else {
            return m;
        }
    }
    sa[l] as usize
}

pub fn binary_search_batch<const B: usize>(
    sa: &SaNaive,
    q: [&Seq; B],
    cnt: &mut usize,
) -> [usize; B] {
    let mut l = [0; B];
    let mut r = [sa.sa.len(); B];
    let iterations = sa.sa.len().ilog2() + 1;
    for _ in 0..iterations {
        let mut m = [0; B];
        let mut idx = [0; B];
        let mut t = [&sa.t[..0]; B];
        for i in 0..B {
            m[i] = (l[i] + r[i]) / 2;
            *cnt += 1;
            prefetch_index(&sa.sa, m[i]);
        }
        for i in 0..B {
            idx[i] = sa[m[i]];
            prefetch_index(&sa.t, idx[i] as usize);
            prefetch_index(&sa.t, idx[i] as usize + 15);
        }
        for i in 0..B {
            t[i] = &sa.suffix_at(idx[i]);
            if t[i] < q[i] {
                l[i] = m[i] + 1;
            } else {
                r[i] = m[i];
            }
        }
    }
    l.map(|l| sa.sa[l] as usize)
}

pub fn binary_search_batch_cmp<const B: usize>(
    sa: &SaNaive,
    q: [&Seq; B],
    cnt: &mut usize,
) -> [usize; B] {
    let mut l = [0; B];
    let mut r = [sa.sa.len(); B];
    let iterations = sa.sa.len().ilog2() + 1;
    for _ in 0..iterations {
        let mut m = [0; B];
        let mut idx = [0; B];
        let mut t = [&sa.t[..0]; B];
        for i in 0..B {
            m[i] = (l[i] + r[i]) / 2;
            *cnt += 1;
            prefetch_index(&sa.sa, m[i]);
        }
        for i in 0..B {
            idx[i] = sa[m[i]];
            prefetch_index(&sa.t, idx[i] as usize);
            prefetch_index(&sa.t, idx[i] as usize + 15);
        }
        for i in 0..B {
            t[i] = &sa.suffix_at(idx[i]);
            if cmp(t[i], q[i]) == Less {
                l[i] = m[i] + 1;
            } else {
                r[i] = m[i];
            }
        }
    }
    l.map(|l| sa.sa[l] as usize)
}

pub fn branchfree_search(sa: &SaNaive, q: &Seq, cnt: &mut usize) -> usize {
    let mut l = 0;
    let mut n = sa.sa.len();
    while n > 0 {
        let half = (n + 1) / 2;
        *cnt += 1;
        l = if sa.suffix(l + half) < q { l + half } else { l };
        n -= half;
    }
    *cnt += 1;
    sa.sa[l] as usize
}

pub fn branchfree_search_batch<const B: usize>(
    sa: &SaNaive,
    q: [&Seq; B],
    cnt: &mut usize,
) -> [usize; B] {
    let mut l = [0; B];
    let mut n = sa.sa.len();
    while n > 0 {
        let half = (n + 1) / 2;
        let mut m = [0; B];
        let mut idx = [0; B];
        let mut t = [&sa.t[..0]; B];
        for i in 0..B {
            m[i] = l[i] + half;
            *cnt += 1;
            prefetch_index(&sa.sa, m[i]);
        }
        for i in 0..B {
            idx[i] = sa[m[i]];
            prefetch_index(&sa.t, idx[i] as usize);
            prefetch_index(&sa.t, idx[i] as usize + 15);
        }
        for i in 0..B {
            t[i] = &sa.suffix_at(idx[i]);
            l[i] = if t[i] < q[i] { m[i] } else { l[i] }
        }
        n -= half;
    }
    l.map(|l| sa.sa[l] as usize)
}

/// Compare query to text.
/// Assumes:
/// - q.len() < t.len()
/// - both t and q can be safely indexed past-the-end for ~32 characters.
/// TODO: Keep track of the longest common prefix between query and left/right bounds,
/// so that the equal part can be skipped.
pub fn cmp(t: &[u8], q: &[u8]) -> std::cmp::Ordering {
    let mut len = q.len();
    let mut t = t.as_ptr();
    let mut q = q.as_ptr();
    const B: usize = 16;
    loop {
        unsafe {
            let t_head: [u8; B] = from_raw_parts(t, B).try_into().unwrap();
            let simd_t: Simd<u8, B> = Simd::from_array(t_head);
            let q_head: [u8; B] = from_raw_parts(q, B).try_into().unwrap();
            let simd_q: Simd<u8, B> = Simd::from_array(q_head);
            let eq = simd_t.simd_eq(simd_q).to_bitmask() as u32;
            let cnt = if cfg!(target_endian = "little") {
                eq.trailing_ones()
            } else {
                eq.leading_ones()
            } as usize;
            if cnt < B && cnt < len {
                return if *t.add(cnt) < *q.add(cnt) {
                    Less
                } else {
                    Greater
                };
            }
            if len < B {
                return Greater;
            }
            t = t.add(B);
            q = q.add(B);
            len -= B;
        }
    }
}

pub fn interpolation_search<const K: usize>(sa: &SaNaive, q: &Seq, cnt: &mut usize) -> usize {
    let mut l = 0;
    let mut r = sa.sa.len();
    let mut l_val = 0;
    let mut r_val = 4usize.pow(K as u32);
    let q_val = string_value::<K>(q);
    assert!(
        r_val.checked_mul(r).is_some(),
        "Too large K causes integer overflow."
    );
    while l < r {
        *cnt += 1;
        // The +1 and +2 ensure l<m<r.
        // HOT: The division is slow.
        let mut m = l + ((r - l) * (q_val - l_val + 1)) / (r_val - l_val + 2);
        let low = l + (r - l) / 16;
        let high = l + 15 * (r - l) / 16;
        m = m.clamp(low, high);
        let t = sa.suffix(m);
        let m_val = string_value::<K>(t);
        if t < q {
            l = m + 1;
            l_val = m_val;
        } else {
            r = m;
            r_val = m_val;
        }
    }
    sa.sa[l] as usize
}

fn bench(sa: &SaNaive, queries: &[&Seq], name: &str, f: &fn(&SaNaive, &Seq, &mut usize) -> usize) {
    let start = std::time::Instant::now();
    let mut cnt = 0;
    for &q in queries {
        f(sa, q, &mut cnt);
    }
    let elapsed = start.elapsed();
    let per_query = elapsed / queries.len() as u32;
    let per_suffix = elapsed / cnt as u32;
    let cnt_per_query = cnt as f32 / queries.len() as f32;
    eprintln!(
        "{name:<20}: {elapsed:>8.2?} {per_query:>6.0?} {per_suffix:>6.0?} {cnt_per_query:>5.2?}"
    );
}

fn bench_batch<const B: usize>(
    sa: &SaNaive,
    queries: &[&Seq],
    name: &str,
    f: &fn(&SaNaive, [&Seq; B], &mut usize) -> [usize; B],
) {
    let start = std::time::Instant::now();
    let mut cnt = 0;
    for &q in queries.array_chunks::<B>() {
        f(sa, q, &mut cnt);
    }
    let elapsed = start.elapsed();
    let per_query = elapsed / queries.len() as u32;
    let per_suffix = elapsed / cnt as u32;
    let cnt_per_query = cnt as f32 / queries.len() as f32;
    eprintln!(
        "{name:<20}: {elapsed:>8.2?} {per_query:>6.0?} {per_suffix:>6.0?} {cnt_per_query:>5.2?}"
    );
}

#[derive(Parser)]
struct Args {
    #[clap(short)]
    n: Option<usize>,
    #[clap(short, default_value_t = 10000000)]
    q: usize,

    #[clap(short)]
    path: Option<PathBuf>,

    #[clap(short, default_value_t = 0, action = clap::ArgAction::Count,)]
    verbose: u8,
}

fn main() {
    let args = Args::parse();

    stderrlog::new()
        .verbosity(2 + args.verbose as usize)
        .show_level(false)
        .init()
        .unwrap();

    // Get a fixed seeded rng.
    let rng = &mut ChaCha8Rng::seed_from_u64(31415);

    let mut t = if let Some(path) = args.path {
        info!("Reading {path:?}..");
        let mut t = read_fasta_file(&path);
        info!("Length {}", t.len());
        if let Some(n) = args.n {
            if n < t.len() {
                t.resize(n, 0);
            }
            info!("Cropped to {n}");
        }
        t
    } else {
        debug!("gen string..");
        random_string(args.n.unwrap_or(100_000_000), rng)
    };

    // Padding.
    t.extend(repeat(0).take(200));
    let t = &t[..t.len() - 200];

    debug!("gen queries..");
    let queries = random_queries(t, args.q, rng);

    info!("build SA..");
    let start = std::time::Instant::now();
    let sa = SaNaive::build(t);
    info!("build SA: {:.2?}", start.elapsed());

    info!("start bench..");

    eprintln!(
        "{:<20}  {:>8} {:>6} {:>6} {:>5}",
        "method", "total", "/query", "/loop", "#loops"
    );

    let funcs: &[(&str, fn(&SaNaive, &[u8], &mut usize) -> usize)] = &[
        ("binary", binary_search),
        ("binary_cmp", binary_search_cmp),
        // ("branchy", branchy_search),
        ("branchfree", branchfree_search),
        // ("interpolation_8", interpolation_search::<8>),
        ("interpolation_16", interpolation_search::<16>),
        // ("interpolation_24", interpolation_search::<24>),
    ];

    // for (name, f) in funcs {
    //     bench(&sa, &queries, name, f);
    // }

    // let funcs: fn(&SaNaive, [&[u8]; 2], &mut usize) -> [usize; 2] = binary_search_batch::<2>;
    // bench_batch(&sa, &queries, "binary_batch_2", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 2], &mut usize) -> [usize; 2] = binary_search_batch_cmp::<2>;
    // bench_batch(&sa, &queries, "binary_batch_2_cmp", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 4], &mut usize) -> [usize; 4] = binary_search_batch::<4>;
    // bench_batch(&sa, &queries, "binary_batch_4", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 4], &mut usize) -> [usize; 4] = binary_search_batch_cmp::<4>;
    // bench_batch(&sa, &queries, "binary_batch_4_cmp", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 8], &mut usize) -> [usize; 8] = binary_search_batch::<8>;
    // bench_batch(&sa, &queries, "binary_batch_8", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 8], &mut usize) -> [usize; 8] = binary_search_batch_cmp::<8>;
    // bench_batch(&sa, &queries, "binary_batch_8_cmp", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 16], &mut usize) -> [usize; 16] = binary_search_batch::<16>;
    // bench_batch(&sa, &queries, "binary_batch_16", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 16], &mut usize) -> [usize; 16] = binary_search_batch_cmp::<16>;
    // bench_batch(&sa, &queries, "binary_batch_16_cmp", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 32], &mut usize) -> [usize; 32] = binary_search_batch::<32>;
    // bench_batch(&sa, &queries, "binary_batch_32", &funcs);
    let funcs: fn(&SaNaive, [&[u8]; 32], &mut usize) -> [usize; 32] = binary_search_batch_cmp::<32>;
    bench_batch(&sa, &queries, "binary_batch_32_cmp", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 64], &mut usize) -> [usize; 64] = binary_search_batch::<64>;
    // bench_batch(&sa, &queries, "binary_batch_64", &funcs);
    // let funcs: fn(&SaNaive, [&[u8]; 64], &mut usize) -> [usize; 64] = binary_search_batch_cmp::<64>;
    // bench_batch(&sa, &queries, "binary_batch_64_cmp", &funcs);
}
