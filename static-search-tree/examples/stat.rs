#![allow(unused)]
use bench::{bench_batch, BFn};
use bplustree::{BpTree16, BpTree16R};
use static_search_tree::*;

fn main() {
    const TEST_START_POW2: usize = 31;
    const TEST_END_POW2: usize = 31;
    const TEST_QUERIES: usize = 10000000;

    eprintln!("Gen queries..");
    let queries = &gen_queries(TEST_QUERIES);
    eprintln!("Gen queries DONE");

    for pow2 in TEST_START_POW2..TEST_END_POW2 + 1 {
        eprintln!("Testing size: {}", pow2);
        let size = 2usize.pow(pow2 as u32);
        eprintln!("Gen vals..");
        let vals = gen_vals(size, true);
        eprintln!("Gen vals DONE");

        eprintln!("Building B+Tree..");
        let bp = &mut BpTree16R::new_fwd(vals.clone(), true);
        eprintln!("Building B+Tree DONE");

        // let f: BFn<128, _> = ("bp_batch_prefetch", BpTree16::batch_prefetch::<128>);
        // let t = bench_batch(bp, f, queries);
        // eprintln!("{}: {}", f.0, t);
        // let f: BFn<128, _> = ("bp_batch_ptr", BpTree16::batch_ptr::<128>);
        // let t = bench_batch(bp, f, queries);
        // eprintln!("{}: {}", f.0, t);
        // let f: BFn<128, _> = ("bp_batch_ptr2", BpTree16::batch_ptr3::<128, true>);
        // for _ in 0..10 {
        //     let t = bench_batch(bp, f, queries);
        //     eprintln!("{}: {}", f.0, t);
        // }

        for _ in 0..1000 {
            let f: BFn<128, _> = (
                "bpf_batch_pf_2",
                BpTree16R::batch_no_prefetch::<128, false, 2>,
            );
            bench_batch(bp, f, queries);

            // let f: IFn<_> = ("bpf_interleave_64", BpTree16R::interleave::<64, false>);
            // bench_all(bp, f, queries);
        }
    }
}
