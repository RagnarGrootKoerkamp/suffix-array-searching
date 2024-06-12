use std::ops::Range;

use rand::Rng;

type Seq = [u8];
type Sequence = Vec<u8>;

pub fn random_string(n: usize) -> Sequence {
    let mut rng = rand::thread_rng();
    let mut seq = Vec::with_capacity(n);
    for _ in 0..n {
        seq.push(rng.gen());
    }
    seq
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

    pub fn binary_search(&self, q: &Seq) -> usize {
        let mut l = 0;
        let mut r = self.sa.len();
        while l < r {
            let m = (l + r) / 2;
            if self.suffix(m) < q {
                l = m + 1;
            } else {
                r = m;
            }
        }
        self.sa[l] as usize
    }

    pub fn branchy_search(&self, q: &Seq) -> usize {
        let mut l = 0;
        let mut r = self.sa.len();
        while l < r {
            let m = (l + r) / 2;
            let t = self.suffix(m);
            if t < q {
                l = m + 1;
            } else if t > q {
                r = m;
            } else {
                return m;
            }
        }
        self.sa[l] as usize
    }

    pub fn branchfree_search(&self, q: &Seq) -> usize {
        let mut l = 0;
        let mut n = self.sa.len();
        while n > 1 {
            let half = n / 2;
            l = if self.suffix(l + half) < q {
                l + half
            } else {
                l
            };
            n -= half;
        }
        self.sa[l + if self.suffix(l) < q { 1 } else { 0 }] as usize
    }
}
