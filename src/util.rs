use rand::Rng;
use std::{path::Path, sync::LazyLock};

pub type Seq = [u8];
pub type Sequence = Vec<u8>;

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

    let mut reader = needletail::parse_fastx_file(path).unwrap();
    let mut out = vec![];
    while let Some(record) = reader.next() {
        out.extend(
            record
                .unwrap()
                .seq()
                .into_iter()
                .map(|c| unsafe { map.get_unchecked(*c as usize) }),
        );
        // break;
    }
    out
}

pub fn read_human_genome() -> Sequence {
    read_fasta_file(&Path::new("human-genome.fa"))
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

/// Prefetch the given cacheline into L1 cache.
pub fn prefetch_index<T>(s: &[T], index: usize) {
    let ptr = unsafe { s.as_ptr().add(index) as *const u64 };
    prefetch_ptr(ptr);
}

/// Prefetch the given cacheline into L1 cache.
pub fn prefetch_ptr<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // TODO: Put this behind a feature flag.
        // std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_LOCALITY3);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        // Do nothing.
    }
}

pub fn time<T>(t: &str, f: impl FnOnce() -> T) -> T {
    eprintln!("{t}: Starting");
    let start = std::time::Instant::now();
    let r = f();
    let elapsed = start.elapsed();
    eprintln!("{t}: Elapsed: {:?}", elapsed);
    r
}

fn init_trace() {
    // use tracing::level_filters::LevelFilter;
    // use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    // tracing_subscriber::registry()
    //     .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
    //     .with(
    //         tracing_subscriber::EnvFilter::builder()
    //             .with_default_directive(LevelFilter::TRACE.into())
    //             .from_env_lossy(),
    //     )
    //     .init();

    stderrlog::new()
        .color(stderrlog::ColorChoice::Always)
        .verbosity(5)
        .show_level(true)
        .init()
        .unwrap();
}

pub static INIT_TRACE: LazyLock<()> = LazyLock::new(init_trace);
