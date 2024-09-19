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

    let mut counts = vec![0; 1 << (2 * k)];
    for i in 0..seq.len() - k + 1 {
        let mut key = 0;
        for j in 0..k {
            key = key << 2 | seq[i + j] as usize;
        }
        counts[key] += 1;
    }

    let out = serde_json::to_string(&counts).unwrap();
    eprintln!("{}", out);
}
