use clap::Parser;
use sa_layout::*;

#[derive(Parser)]
struct Args {
    k: usize,
}

fn main() {
    let seq = read_human_genome();

    let args = Args::parse();
    let k = args.k;

    let mut counts = vec![0u32; 1 << (2 * k)];
    let mut key = 0;
    for i in 0..k - 1 {
        key = key << 2 | seq[i] as usize;
    }
    for i in k - 1..seq.len() {
        key = key << 2 | seq[i] as usize;
        key &= (1 << (2 * k)) - 1;
        counts[key] += 1;
    }

    // histogram of sizes
    let mut hist = vec![0; 30];
    let mut sum = vec![0; 30];
    for &c in &counts {
        if c > 0 {
            let x = c.ilog2() as usize;
            hist[x] += 1;
            sum[x] += c;
        }
    }

    let kmers = seq.len() - k + 1;
    let mut acc = 0;

    for i in 0..hist.len() {
        let h = hist[i];
        let s = sum[i];
        acc += s;
        eprintln!(
            "2^{i:>2} {h:>10} {s:>10} {acc:>10} {:>10.4} {:>10.4}",
            acc as f64 / kmers as f64,
            1. - acc as f64 / kmers as f64
        );
    }

    // let out = serde_json::to_string(&counts).unwrap();
    // eprintln!("{}", out);
}
