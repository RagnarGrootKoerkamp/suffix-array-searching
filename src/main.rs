use std::{
    arch::x86_64::_pext_u64,
    iter::repeat,
    ops::Range,
    path::{Path, PathBuf},
};

use clap::Parser;
use itertools::Itertools;
use log::{debug, info};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

type Seq = [u8];
type Sequence = Vec<u8>;

pub fn read_fasta_file(path: &Path) -> Sequence {
    let mut map = [0; 256];
    map[b'A' as usize] = 0;
    map[b'C' as usize] = 1;
    map[b'G' as usize] = 2;
    map[b'T' as usize] = 3;

    map[b'a' as usize] = 0;
    map[b'c' as usize] = 1;
    map[b'g' as usize] = 2;
    map[b't' as usize] = 3;

    let mut f = needletail::parse_fastx_file(path).unwrap();
    let mut out = vec![];
    while let Some(seq) = f.next() {
        out.extend(seq.unwrap().seq().into_iter().map(|c| map[*c as usize]));
    }
    out
}

/// Generate a random text of length n.
pub fn random_string(n: usize, rng: &mut impl Rng) -> Sequence {
    let mut seq = Vec::with_capacity(n);
    for _ in 0..n {
        seq.push(rng.gen_range(0..4));
    }
    seq
}

/// Generate a random subset of queries.
pub fn random_queries<'t>(t: &'t Seq, n: usize, rng: &mut impl Rng) -> Vec<&'t Seq> {
    (0..n)
        .map(|_| {
            let i = rng.gen_range(0..t.len() - 200);
            let len = rng.gen_range(30..100);
            &t[i..i + len]
        })
        .collect()
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
        &self.t[self.sa[i] as usize..]
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
    sa.sa[l] as usize
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
    sa.sa[l] as usize
}

pub fn branchfree_search(sa: &SaNaive, q: &Seq, cnt: &mut usize) -> usize {
    let mut l = 0;
    let mut n = sa.sa.len();
    while n > 1 {
        let half = n / 2;
        *cnt += 1;
        l = if sa.suffix(l + half) < q { l + half } else { l };
        n -= half;
    }
    *cnt += 1;
    sa.sa[l + if sa.suffix(l) < q { 1 } else { 0 }] as usize
}

fn string_value<const K: usize>(q: &Seq) -> usize {
    // Read two u64 values starting at q. Use bit-extract instructions to extract the low 2 bits of each byte.
    let mask = 0x0303030303030303u64;
    if K == 8 {
        // Read u64 from q.
        unsafe {
            let a = *(q.as_ptr() as *const u64);
            let a = a.swap_bytes();
            let v1 = _pext_u64(a, mask) as usize;
            // assert_eq!(v0, v1, "\n{:?}\n{v0:0b}\n{v1:0b}", &q[..8]);
            return v1;
        }
    }
    if K == 16 {
        unsafe {
            let a = *(q.as_ptr() as *const u64);
            let b = *(q.as_ptr().add(8) as *const u64);
            let a = a.swap_bytes();
            let b = b.swap_bytes();
            let v1 = ((_pext_u64(a, mask) as usize) << 16) + _pext_u64(b, mask) as usize;
            // assert_eq!(v0, v1);
            return v1;
        }
    }
    let v0 = q.iter().take(K).fold(0, |acc, &x| acc * 4 + x as usize);
    v0
}

pub fn interpolation_search<const K: usize>(sa: &SaNaive, q: &Seq, cnt: &mut usize) -> usize {
    let mut l = 0;
    let mut r = sa.sa.len();
    let mut l_val = 0;
    let mut r_val = 4usize.pow(K as u32);
    let q_val = string_value::<K>(q);
    assert!(K <= 20, "K > 20 will cause integer overflow.");
    while l < r {
        // The +1 and +2 ensure l<m<r.
        // HOT: The division is slow.
        let m = l + ((r - l) * (q_val - l_val + 1)) / (r_val - l_val + 2);
        *cnt += 1;
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
    println!("{name:>30}: {elapsed:6.2?} {per_query:6.0?} {per_suffix:6.0?} {cnt_per_query:5.2?}");
}

#[derive(Parser)]
struct Args {
    #[clap(short, default_value_t = 10000000)]
    n: usize,
    #[clap(short, default_value_t = 5000000)]
    q: usize,

    #[clap(short)]
    path: Option<PathBuf>,

    #[clap(short, default_value_t = 0, action = clap::ArgAction::Count,)]
    verbose: usize,
}

fn main() {
    let args = Args::parse();

    stderrlog::new()
        .verbosity(2 + args.verbose)
        .show_level(false)
        .init()
        .unwrap();

    // Get a fixed seeded rng.
    let rng = &mut ChaCha8Rng::seed_from_u64(31415);

    let mut t = if let Some(path) = args.path {
        info!("Reading {path:?}..");
        let t = read_fasta_file(&path);
        info!("Length {}", t.len());
        t
    } else {
        debug!("gen string..");
        random_string(args.n, rng)
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

    let funcs: &[(&str, fn(&SaNaive, &[u8], &mut usize) -> usize)] = &[
        ("binary", binary_search),
        // ("branchy", branchy_search),
        // ("branchfree", branchfree_search),
        // ("interpolation_8", interpolation_search::<8>),
        ("interpolation_16", interpolation_search::<16>),
    ];

    for (name, f) in funcs {
        bench(&sa, &queries, name, f);
    }
}
