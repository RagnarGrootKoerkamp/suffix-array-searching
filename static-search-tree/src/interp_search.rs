use itertools::Itertools;

use crate::binary_search::SortedVec;
use std::simd::{cmp::SimdOrd, isizex8, LaneCount, Simd};

impl SortedVec {
    /// Return the value of the first value >= query.
    pub fn interpolation_search(&self, q: u32) -> u32 {
        let mut l: usize = 0;
        // FIXME: is this inclusive?
        let mut r: usize = self.vals.len() - 1;
        let mut l_val: usize = self.get(l).try_into().unwrap();
        let mut r_val: usize = self.get(r).try_into().unwrap();
        let q_val = q.try_into().unwrap();
        if q_val <= l_val {
            return self.get(l);
        }
        assert!(
            r_val.checked_mul(r).is_some(),
            "Too large K causes integer overflow."
        );
        // n = 10^9
        // lg n = 30
        // lg lg n = 5
        //
        // 1111111111111111111111112 9999
        // *.  --------------------     *
        //  *.                          *
        //   *                          *
        //    *                         *
        //                         *    *
        //
        while l < r {
            // The +1 and +2 ensure l<m<r.
            // HOT: The division is slow.
            // OMFG is there really not a better way to do this? Multiple integer types are a nightmare
            let mut m: usize = l + (r - l) * (q_val - l_val + 1) / (r_val - l_val + 2);
            let low = l + (r - l) / 16;
            let high = l + 15 * (r - l) / 16;
            m = m.clamp(low, high);
            let m_val: usize = self.get(m).try_into().unwrap();
            if m_val < q_val {
                l = m + 1;
                l_val = m_val;
            } else {
                r = m;
                r_val = m_val;
            }
        }
        self.get(l)
    }

    pub fn interp_search_batched_simd<const P: usize>(&self, qs: &[u32; P]) -> [u32; P] {
        let mut ls: [isize; P] = [0isize; P];
        let mut rs: [isize; P] = [(self.vals.len() - 1).try_into().unwrap(); P];
        let mut l_vals: [isize; P] = ls.map(|i| self.get(i as usize).try_into().unwrap());
        let mut r_vals: [isize; P] = rs.map(|i| self.get(i as usize).try_into().unwrap());
        let qs: [isize; P] = qs
            .iter()
            .map(|&x| x as isize)
            .collect_vec()
            .try_into()
            .unwrap();
        let mut retvals = [0u32; P];
        // an accumulator value going from 0 to P
        let mut done = 0;
        assert!(P % 8 == 0);
        let ones = isizex8::splat(1);
        let twos = isizex8::splat(2);
        while done < P {
            // println!("Iteration! {:?} {:?}", ls, rs);
            done = 0;
            // this is subject to change, maybe we want completely size-agnostic simd
            let simd_iters = P / 8;
            for i in 0..simd_iters {
                let q_vec = isizex8::from_slice(&qs[i * 8..(i + 1) * 8]);
                let l_vec = isizex8::from_slice(&ls[i * 8..(i + 1) * 8]);
                let r_vec = isizex8::from_slice(&rs[i * 8..(i + 1) * 8]);

                let l_vals_vec = isizex8::from_slice(&l_vals[i * 8..(i + 1) * 8]);
                let r_vals_vec = isizex8::from_slice(&r_vals[i * 8..(i + 1) * 8]);

                let mut m = l_vec
                    + (r_vec - l_vec) * (q_vec - l_vals_vec + ones)
                        / (r_vals_vec - l_vals_vec + twos);
                let low = l_vec + (r_vec - l_vec) / isizex8::splat(16);
                let high = l_vec + isizex8::splat(15) * (r_vec - l_vec) / isizex8::splat(16);
                // equivalent to clamp
                m = m.simd_min(high);
                m = m.simd_max(low);
                let m_arr = m.as_array();
                let mut m_val = [0isize; 8];
                for j in 0..8 {
                    if ls[i * 8 + j] < rs[i * 8 + j] {
                        m_val[j] = self.get(m_arr[j] as usize).try_into().unwrap();
                        if m_val[j] < qs[i * 8 + j] {
                            ls[i * 8 + j] = m_arr[j] + 1;
                            l_vals[i * 8 + j] = m_val[j];
                        } else {
                            rs[i * 8 + j] = m_arr[j];
                            r_vals[i * 8 + j] = m_val[j];
                        }
                    } else {
                        retvals[i * 8 + j] = self.get(ls[i * 8 + j] as usize);
                        done += 1;
                    }
                }
            }
        }

        retvals
    }

    pub fn interp_search_batched<const P: usize>(&self, qs: &[u32; P]) -> [u32; P] {
        let mut ls = [0usize; P];
        let mut rs = [self.vals.len() - 1; P];
        let mut l_vals: [usize; P] = ls.map(|i| self.get(i).try_into().unwrap());
        let mut r_vals: [usize; P] = rs.map(|i| self.get(i).try_into().unwrap());
        let mut retvals = [0u32; P];
        // TODO: do this smarter
        let mut done = [false; P];
        // do we need this edge case?

        // inkit to avoid negative values
        for i in 0..P {
            let q_val: usize = qs[i].try_into().unwrap();
            if q_val <= l_vals[i] {
                retvals[i] = self.get(ls[i]);
                done[i] = true;
            }
        }

        // this loop should not be difficult to make a fixed number of iterations, or just not have to check
        // the condition every time
        while !done.iter().all(|&x| x) {
            // println!("Iteration! {:?} {:?}", ls, rs);
            for i in 0..P {
                // this might be a hard-to-predict branch
                if done[i] {
                    continue;
                }

                let q_val = qs[i].try_into().unwrap();
                let l = ls[i];
                let r = rs[i];
                let l_val = l_vals[i];
                let r_val = r_vals[i];

                if l >= r {
                    retvals[i] = self.get(l);
                    done[i] = true;
                    continue;
                }

                let mut m: usize = l + (r - l) * (q_val - l_val + 1) / (r_val - l_val + 2);
                let low = l + (r - l) / 16;
                let high = l + 15 * (r - l) / 16;
                m = m.clamp(low, high);
                let m_val = self.get(m).try_into().unwrap();
                if m_val < q_val {
                    ls[i] = m + 1;
                    l_vals[i] = m_val;
                } else {
                    rs[i] = m;
                    r_vals[i] = m_val;
                }
            }
        }

        retvals
    }
}

#[cfg(test)]
mod tests {
    use crate::{binary_search::SortedVec, SearchIndex};

    #[test]
    fn interppolation_vs_binsearch() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let q = 5;
        let i_res = SortedVec::new(&input).interpolation_search(q);
        let bin_res = SortedVec::new(&input).binary_search(q);
        println!("{i_res}, {bin_res}");
        assert!(i_res == bin_res);
    }

    #[test]
    fn normal_vs_batched() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let qs = [0, 1, 2, 3, 4, 5, 6, 9];
        let batch_res = SortedVec::new(&input).interp_search_batched::<8>(&qs);
        let i_res = qs.map(|q| SortedVec::new(&input).interpolation_search(q));
        println!("{:?}, {:?}", i_res, batch_res);
        assert!(i_res == batch_res);
    }

    #[test]
    fn interpolation_batched_simd() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let qs = [0, 1, 2, 3, 4, 5, 6, 9];
        let bin_res = SortedVec::new(&input).interp_search_batched::<8>(&qs);
        let i_res = SortedVec::new(&input).interp_search_batched_simd::<8>(&qs);
        println!("{:?}, {:?}", i_res, bin_res);
        assert!(i_res == bin_res);
    }
}
