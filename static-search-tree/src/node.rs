use std::arch::x86_64::{_mm256_movemask_epi8, _mm256_packs_epi32};
use std::simd::prelude::*;

pub(super) const MAX: u32 = i32::MAX as u32;

#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct BTreeNode<const B: usize, const N: usize> {
    pub(super) data: [u32; N],
}

impl<const B: usize, const N: usize> Default for BTreeNode<B, N> {
    fn default() -> BTreeNode<B, N> {
        BTreeNode { data: [0; N] }
    }
}

impl<const B: usize, const N: usize> BTreeNode<B, N> {
    pub fn find(&self, q: u32) -> usize {
        self.find_popcnt(q)
    }

    #[allow(unused)]
    pub fn find_ctz(&self, q: u32) -> usize {
        let data_simd: Simd<u32, 16> = Simd::from_slice(&self.data[0..N]);
        let q_simd = Simd::splat(q);
        let mask = q_simd.simd_le(data_simd);
        mask.first_set().unwrap_or(B)
    }

    /// Return the index of the first element >=q.
    /// Assumes that all elements fit in an i32, since SIMD doesn't have
    /// unsigned comparisons.
    pub fn find_popcnt(&self, q: u32) -> usize {
        let low: Simd<u32, 8> = Simd::from_slice(&self.data[0..N / 2]);
        let high: Simd<u32, 8> = Simd::from_slice(&self.data[N / 2..N]);
        let q_simd = Simd::<_, 8>::splat(q);
        // Merge the two masks, and convert to a single shuffled(!) mask.
        // But that's OK since popcount doesn't care about order.
        // TODO: Can we do this using portable SIMD?
        unsafe {
            let q_simd: Simd<i32, 8> = t(q_simd);
            let mask_low = q_simd.simd_gt(t(low));
            let mask_high = q_simd.simd_gt(t(high));
            use std::mem::transmute as t;
            let merged = _mm256_packs_epi32(t(mask_low), t(mask_high));
            let mask = _mm256_movemask_epi8(t(merged));
            mask.count_ones() as usize / 2
        }
    }

    pub fn find_splat(&self, q_simd: Simd<u32, 8>) -> usize {
        let low: Simd<u32, 8> = Simd::from_slice(&self.data[0..N / 2]);
        let high: Simd<u32, 8> = Simd::from_slice(&self.data[N / 2..N]);
        unsafe {
            let q_simd: Simd<i32, 8> = t(q_simd);
            let mask_low = q_simd.simd_gt(t(low));
            let mask_high = q_simd.simd_gt(t(high));
            use std::mem::transmute as t;
            let merged = _mm256_packs_epi32(t(mask_low), t(mask_high));
            let mask = _mm256_movemask_epi8(t(merged));
            mask.count_ones() as usize / 2
        }
    }

    /// This returns the popcount multiplied by 64.
    pub fn find_splat64(&self, q_simd: Simd<u32, 8>) -> usize {
        let low: Simd<u32, 8> = Simd::from_slice(&self.data[0..N / 2]);
        let high: Simd<u32, 8> = Simd::from_slice(&self.data[N / 2..N]);
        unsafe {
            let q_simd: Simd<i32, 8> = t(q_simd);
            let mask_low = q_simd.simd_gt(t(low));
            let mask_high = q_simd.simd_gt(t(high));
            use std::mem::transmute as t;
            let merged = _mm256_packs_epi32(t(mask_low), t(mask_high));
            let mask = _mm256_movemask_epi8(t(merged));
            mask.count_ones() as usize * 32
        }
    }

    /// This returns the popcount multiplied by 64.
    /// Normal:   last index < query.
    /// Reversed: last index <= query.
    pub fn find_splat_last(&self, q_simd: Simd<u32, 8>) -> usize {
        let low: Simd<u32, 8> = Simd::from_slice(&self.data[0..N / 2]);
        let high: Simd<u32, 8> = Simd::from_slice(&self.data[N / 2..N]);
        unsafe {
            let q_simd: Simd<i32, 8> = t(q_simd);
            let mask_low = q_simd.simd_ge(t(low));
            let mask_high = q_simd.simd_ge(t(high));
            use std::mem::transmute as t;
            let merged = _mm256_packs_epi32(t(mask_low), t(mask_high));
            let mask = _mm256_movemask_epi8(t(merged));
            mask.count_ones() as usize / 2
        }
    }

    pub fn find_splat64_last(&self, q_simd: Simd<u32, 8>) -> usize {
        let low: Simd<u32, 8> = Simd::from_slice(&self.data[0..N / 2]);
        let high: Simd<u32, 8> = Simd::from_slice(&self.data[N / 2..N]);
        unsafe {
            let q_simd: Simd<i32, 8> = t(q_simd);
            let mask_low = q_simd.simd_ge(t(low));
            let mask_high = q_simd.simd_ge(t(high));
            use std::mem::transmute as t;
            let merged = _mm256_packs_epi32(t(mask_low), t(mask_high));
            let mask = _mm256_movemask_epi8(t(merged));
            mask.count_ones() as usize * 32
        }
    }

    /// Return the index of the first element >=q.
    /// This first does a single comparison to choose the left or right half of the array,
    /// and then uses SIMD on that half.
    /// This may reduce the pressure on SIMD registers.
    pub fn find_split(&self, q: u32) -> usize {
        let idx;
        if q <= self.data[B / 2] {
            idx = 0;
        } else {
            idx = B / 2;
        }
        let half_simd = Simd::<u32, 8>::from_slice(&self.data[idx..idx + B / 2]);
        let q_simd = Simd::splat(q);
        let mask = q_simd.simd_le(half_simd);
        idx + mask.first_set().unwrap_or(8)
    }
}
