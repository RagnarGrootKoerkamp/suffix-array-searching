use itertools::Itertools;
use std::array::from_fn;
use std::{fmt::Debug, simd::Simd};

use crate::node::{BTreeNode, MAX};
use crate::{prefetch_ptr, vec_on_hugepages, SearchIndex};

/// N total elements in a node.
/// B branching factor.
/// B-1 actual elements in a node.
/// COMPACT: instead of a single tree with the first few layers removed,
///          store many small packed trees.
#[derive(Debug)]
pub struct PartitionedSTree<const B: usize, const N: usize, const COMPACT: bool, const L1: bool> {
    tree: Vec<BTreeNode<N>>,
    offsets: Vec<usize>,
    /// Amount to shift values/queries to the right to get their part.
    shift: usize,
    /// blocks per part
    bpp: usize,
    /// Number of nodes in layer 1.
    /// Number of values in the root is l1-1.
    l1: usize,
}

impl<const B: usize, const N: usize, const COMPACT: bool, const L1: bool> SearchIndex
    for PartitionedSTree<B, N, COMPACT, L1>
{
    fn size(&self) -> usize {
        std::mem::size_of_val(self.tree.as_slice())
    }
}

pub type PartitionedSTree16 = PartitionedSTree<16, 16, false, false>;
pub type PartitionedSTree15 = PartitionedSTree<15, 16, false, false>;
pub type PartitionedSTree16C = PartitionedSTree<16, 16, true, false>;
pub type PartitionedSTree15C = PartitionedSTree<15, 16, true, false>;
pub type PartitionedSTree16L = PartitionedSTree<16, 16, false, true>;
pub type PartitionedSTree15L = PartitionedSTree<15, 16, false, true>;

impl<const B: usize, const N: usize, const COMPACT: bool, const L1: bool>
    PartitionedSTree<B, N, COMPACT, L1>
{
    // Helper functions for construction.
    fn blocks(n: usize) -> usize {
        n.div_ceil(B)
    }
    fn prev_keys(n: usize) -> usize {
        Self::blocks(n).div_ceil(B + 1) * B
    }
    fn height(n: usize) -> usize {
        if n <= B {
            1
        } else {
            Self::height(Self::prev_keys(n)) + 1
        }
    }
    fn layer_size(mut n: usize, h: usize, height: usize) -> usize {
        for _ in h..height - 1 {
            n = Self::prev_keys(n);
        }
        n
    }

    /// Partition on the first `b` bits of each key before building the tree.
    /// Any bits beyond the maximum value are skipped.
    /// - uses hugepages.
    /// - uses forward layout.
    /// - uses 'rev' bucket order.
    /// - uses the full array.
    pub fn new(vals: &[u32], b: usize) -> Self {
        assert!(vals.is_sorted());
        assert!(*vals.last().unwrap() <= MAX);

        let n = vals.len();
        assert!(n > 0);

        let bits = 1 + vals.last().unwrap().ilog2() as usize;
        let mut shift = bits.saturating_sub(b);

        let mut parts = 1 << (bits - shift);

        // Compute bucket sizes.
        // For compact case, we need to store one sentinel of padding at the end of each part.
        let mut bucket_sizes = vec![if COMPACT { 1 } else { 0 }; parts];
        for &val in vals {
            let bucket = (val >> shift) as usize;
            bucket_sizes[bucket] += 1;
        }

        // Find largest bucket.
        let mut max_bucket = *bucket_sizes.iter().max().unwrap();
        // Number of layers for largest bucket.
        let height = Self::height(max_bucket);

        // Try reducing the number of parts if the height stays the same.
        let mut b2 = b;
        loop {
            if b2 == 0 {
                break;
            }
            b2 -= 1;
            if b2 > bits {
                break;
            }
            let shift2 = bits.saturating_sub(b2);
            let parts2 = 1 << (bits - shift2);
            let mut bucket_sizes2 = vec![if COMPACT { 1 } else { 0 }; parts2];
            for &val in vals {
                let bucket = (val >> shift2) as usize;
                bucket_sizes2[bucket] += 1;
            }
            let max_bucket2 = *bucket_sizes2.iter().max().unwrap();
            let height2 = Self::height(max_bucket2);
            if height2 > height {
                // eprintln!("{bucket_sizes2:?}");
                break;
            }
            shift = shift2;
            parts = parts2;
            max_bucket = max_bucket2;
        }

        let layer_sizes;
        let offsets;
        let mut tree;
        let bpp;
        let mut l1 = 0;
        if COMPACT {
            // In this case, L1 is meaningless!
            assert!(
                !L1,
                "In the compact case, a separate level 0 size doesn't do anything."
            );
            assert!(height > 0);
            // All layers are full, for indexing purposes.
            // TODO: Layer sizes given by max_bucket_size.
            layer_sizes = (0..height)
                .map(|h| Self::layer_size(max_bucket, h, height).div_ceil(B))
                .collect_vec();
            assert!(layer_sizes[0] == 1);
            bpp = layer_sizes.iter().sum::<usize>();
            let n_blocks = parts * bpp;

            // Offsets within a part.
            offsets = layer_sizes
                .iter()
                .scan(0, |sum, sz| {
                    let offset = *sum;
                    *sum += sz;
                    Some(offset)
                })
                .collect_vec();

            tree = vec_on_hugepages::<BTreeNode<N>>(n_blocks);

            // First initialize all nodes in the layer with MAX.
            for node in &mut tree {
                node.data.fill(MAX);
            }

            // Initialize the last layer.
            let mut prev_part = 0;
            let mut idx = 0;

            let h = height - 1;
            let ol = offsets[h]; // last-level offset
            for &val in vals {
                let part = (val >> shift) as usize;

                // For each completed part (typically just 1), append the current val.
                while prev_part < part {
                    if idx / B < layer_sizes[h] {
                        tree[prev_part * bpp + ol + idx / B].data[idx % B..].fill(val);
                    }
                    prev_part += 1;
                    idx = 0;
                }

                tree[part * bpp + ol + idx / B].data[idx % B] = val;
                // If B<N and there is some buffer space in each node,
                // put us also in the last element of the previous node.
                if B < N && idx % B == 0 && idx > 0 {
                    tree[part * bpp + ol + idx / B - 1].data[B] = val;
                }
                idx += 1;
            }

            // Initialize the inner layers.
            for h in (0..height - 1).rev() {
                let oh = offsets[h];

                for part in 0..parts {
                    for i in 0..B * layer_sizes[h] {
                        let mut k = i / B;
                        let j = i % B;
                        k = k * (B + 1) + j + 1;
                        for _l in h..height - 2 {
                            k *= B + 1;
                        }
                        tree[part * bpp + oh + i / B].data[i % B] = if k * B < max_bucket {
                            tree[part * bpp + ol + k - 1].data[B - 1]
                        } else {
                            MAX
                        };
                    }
                }
            }
        } else {
            assert!(height > 0);
            // All layers are full, for indexing purposes.
            // Unless, L1 is set. Then, the first layer below the root (ie the number of entries in the root) can be smaller.
            layer_sizes = if !L1 {
                (0..height).map(|h| (B + 1).pow(h as u32)).collect_vec()
            } else {
                // let normal_layer_sizes = (0..height)
                //     .map(|h| Self::layer_size(max_bucket, h, height))
                //     .collect_vec();
                // eprintln!("normal_layer_sizes={:?}", normal_layer_sizes);
                // let normal_layer_sizes = (0..height)
                //     .map(|h| Self::layer_size(max_bucket, h, height).div_ceil(B))
                //     .collect_vec();
                // eprintln!("normal_layer_sizes={:?}", normal_layer_sizes);

                // Number of nodes in level 1 is number of values in level 0 + 1.
                l1 = Self::layer_size(max_bucket, 1, height).div_ceil(B);
                eprintln!("size: {n} l1 {l1}");
                (0..height)
                    .map(|h| ((B + 1).pow(h as u32) * l1).div_ceil(B + 1))
                    .collect_vec()
            };
            eprintln!("layer_sizes={:?}", layer_sizes);

            assert!(
                layer_sizes[0] == 1,
                "layer_sizes={:?} has unexpected root size?",
                layer_sizes
            );

            bpp = layer_sizes.iter().sum::<usize>();
            let n_blocks = parts * bpp;

            offsets = layer_sizes
                .iter()
                .scan(0, |sum, sz| {
                    let offset = *sum;
                    *sum += parts * sz;
                    Some(offset)
                })
                .collect_vec();

            tree = vec_on_hugepages::<BTreeNode<N>>(n_blocks);

            // First initialize all nodes in the layer with MAX.
            // TODO: Maybe we can omit this and avoid mapping some of the pages of the tree?
            for node in &mut tree {
                node.data.fill(MAX);
            }

            // Initialize the last layer.
            let ol = offsets[height - 1];
            let mut prev_part = 0;
            let mut idx = 0;

            for &val in vals {
                let part = (val >> shift) as usize;

                // For each completed part (typically just 1), append the current val.
                while prev_part < part {
                    if idx < (prev_part + 1) * B * layer_sizes[height - 1] {
                        tree[ol + idx / B].data[idx % B..].fill(val);
                    }
                    prev_part += 1;
                    idx = prev_part * B * layer_sizes[height - 1];
                }

                tree[ol + idx / B].data[idx % B] = val;
                // If B<N and there is some buffer space in each node,
                // put us also in the last element of the previous node.
                if B < N && idx % B == 0 && idx > 0 {
                    tree[ol + idx / B - 1].data[B] = val;
                }
                idx += 1;
            }

            // Initialize the inner layers.
            for h in (0..height - 1).rev() {
                let oh = offsets[h];

                let l = layer_sizes[h];
                let ll = layer_sizes[height - 1];
                for p in 0..parts {
                    for i in 0..B * layer_sizes[h] {
                        let mut k = i / B;
                        let j = i % B;
                        k = k * (B + 1) + j + 1;
                        for _l in h..height - 2 {
                            k *= B + 1;
                        }
                        tree[oh + l * p + i / B].data[i % B] = if k * B < max_bucket {
                            tree[ol + ll * p + k - 1].data[B - 1]
                        } else {
                            MAX
                        };
                    }
                }
            }
        }
        assert!(offsets.len() > 0);

        eprintln!(
            "PartitionedSTree: n={} b={} shift={} parts={} height={} layer_sizes={:?} offsets={:?} compact={COMPACT} max_bucket={max_bucket}",
            n, b, shift, parts, height, layer_sizes, offsets
        );
        // for (i, node) in tree.iter().enumerate() {
        //     eprintln!("{i:>2} {:?}", node);
        // }

        Self {
            tree,
            offsets,
            shift,
            bpp,
            l1,
        }
    }
}

/// Partitions, full
/// First layer 0 of all parts, then layer 1 of all parts, ...
/// Inefficient, because layers much have their 'full' size and grown by B+1 each level.
impl<const B: usize, const N: usize> PartitionedSTree<B, N, false, false> {
    pub fn search<const P: usize, const PF: bool>(&self, qb: &[u32; P]) -> [u32; P] {
        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
            .collect_vec();

        // Initial parts, and prefetch them.
        let o0 = offsets[0];
        let mut k: [usize; P] = qb.map(|q| {
            let k = (q as usize >> self.shift) * 64;
            if PF {
                prefetch_ptr(unsafe { o0.byte_add(k) });
            }
            k
        });
        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

        for [o, o2] in offsets.array_windows() {
            for i in 0..P {
                let jump_to = unsafe { *o.byte_add(k[i]) }.find_splat64(q_simd[i]);
                k[i] = k[i] * (B + 1) + jump_to;
                prefetch_ptr(unsafe { o2.byte_add(k[i]) });
            }
        }

        let o = offsets.last().unwrap();
        from_fn(|i| {
            let idx = unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i]);

            unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
        })
    }
}

/// Partitions, compact.
/// Layout: tree for part 1, tree for part 2, ...
/// More efficient because each part is stored compactly.
/// Slightly slower though, because we must explicitly track to which part each query belongs.
impl<const B: usize, const N: usize> PartitionedSTree<B, N, true, false> {
    pub fn search<const P: usize, const PF: bool>(&self, qb: &[u32; P]) -> [u32; P] {
        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
            .collect_vec();

        // Initial parts, and prefetch them.
        let o0 = offsets[0];
        let mut k: [usize; P] = [0; P];
        let parts: [usize; P] = qb.map(|q| {
            // byte offset of the part.
            let p = (q as usize >> self.shift) * self.bpp * 64;
            if PF {
                prefetch_ptr(unsafe { o0.byte_add(p) });
            }
            p
        });

        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

        for [o, o2] in offsets.array_windows() {
            for i in 0..P {
                let jump_to = unsafe { *o.byte_add(parts[i] + k[i]) }.find_splat64(q_simd[i]);
                k[i] = k[i] * (B + 1) + jump_to;
                prefetch_ptr(unsafe { o2.byte_add(parts[i] + k[i]) });
            }
        }

        let o = offsets.last().unwrap();
        from_fn(|i| {
            let idx = unsafe { *o.byte_add(parts[i] + k[i]) }.find_splat(q_simd[i]);

            unsafe { (o.byte_add(parts[i] + k[i]) as *const u32).add(idx).read() }
        })
    }
}

/// Partitions, full with level1 compression.
/// First layer 0 of all parts, then layer 1 of all parts, ...
/// The number of children of each level0 node (root) is only `self.l1`, instead of the full `B+1`, saving some memory.
impl<const B: usize, const N: usize> PartitionedSTree<B, N, false, true> {
    pub fn search<const P: usize, const PF: bool>(&self, qb: &[u32; P]) -> [u32; P] {
        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
            .collect_vec();

        // Initial parts, and prefetch them.
        let o0 = offsets[0];
        let mut k: [usize; P] = qb.map(|q| {
            let k = (q as usize >> self.shift) * 64;
            if PF {
                prefetch_ptr(unsafe { o0.byte_add(k) });
            }
            k
        });
        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

        if offsets.len() >= 2 {
            // Extract the first iteration over levels, because we multiply by l1 instead of B+1 here.
            let o = offsets[0];
            let o2 = offsets[1];
            for i in 0..P {
                let jump_to = unsafe { *o.byte_add(k[i]) }.find_splat64(q_simd[i]);
                //            vvvvvvv
                k[i] = k[i] * self.l1 + jump_to;
                prefetch_ptr(unsafe { o2.byte_add(k[i]) });
            }
        }

        for [o, o2] in offsets[1..].array_windows() {
            for i in 0..P {
                let jump_to = unsafe { *o.byte_add(k[i]) }.find_splat64(q_simd[i]);
                //            vvvvvvv
                k[i] = k[i] * (B + 1) + jump_to;
                prefetch_ptr(unsafe { o2.byte_add(k[i]) });
            }
        }

        let o = offsets.last().unwrap();
        from_fn(|i| {
            let idx = unsafe { *o.byte_add(k[i]) }.find_splat(q_simd[i]);

            unsafe { (o.byte_add(k[i]) as *const u32).add(idx).read() }
        })
    }
}
