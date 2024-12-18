use crate::{binary_search::SortedVec, SearchScheme};

// completely basic binsearch
pub struct InterpolationSearch;

impl SearchScheme for InterpolationSearch {
    type INDEX = SortedVec;

    /// Return the value of the first value >= query.
    fn query_one(&self, index: &SortedVec, q: u32) -> u32 {
        let mut l: usize = 0;
        // FIXME: is this inclusive?
        let mut r: usize = index.vals.len() - 1;
        let mut l_val: usize = index.get(l).try_into().unwrap();
        let mut r_val: usize = index.get(r).try_into().unwrap();
        let q_val = q.try_into().unwrap();
        if q_val <= l_val {
            return index.get(l);
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
            let m_val: usize = index.get(m).try_into().unwrap();
            if m_val < q_val {
                l = m + 1;
                l_val = m_val;
            } else {
                r = m;
                r_val = m_val;
            }
        }
        index.get(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        binary_search::{BinarySearch, SortedVec},
        SearchIndex,
    };

    #[test]
    fn interppolation_vs_binsearch() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let q = 5;
        let i_res = SortedVec::new(&input).query_one(q, InterpolationSearch);
        let bin_res = SortedVec::new(&input).query_one(q, BinarySearch);
        assert!(i_res == bin_res);
        println!("{i_res}, {bin_res}");
    }
}
