use itertools::Itertools;
use std::array::from_fn;
use std::{fmt::Debug, simd::Simd};

use crate::node::{BTreeNode, MAX};
use crate::{prefetch_ptr, vec_on_hugepages};

/// N total elements in a node.
/// B branching factor.
/// B-1 actual elements in a node.
#[derive(Debug)]
pub struct PartitionedSTree<const B: usize, const N: usize> {
    tree: Vec<BTreeNode<N>>,
    offsets: Vec<usize>,
    /// Amount to shift values/queries to the right to get their part.
    shift: usize,
}

pub type PartitionedSTree16 = PartitionedSTree<16, 16>;
pub type PartitionedSTree15 = PartitionedSTree<15, 16>;

impl<const B: usize, const N: usize> PartitionedSTree<B, N> {
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
        let shift = bits.saturating_sub(b);

        let parts = 1 << (bits - shift);

        // Compute bucket sizes.
        let mut bucket_sizes = vec![0; parts];
        for &val in vals {
            let bucket = (val >> shift) as usize;
            bucket_sizes[bucket] += 1;
        }

        // Find largest bucket.
        let max_bucket = *bucket_sizes.iter().max().unwrap();
        // Number of layers for largest bucket.
        let height = Self::height(max_bucket);

        assert!(height > 0);
        // All layers are full, for indexing purposes.
        // TODO: Layer sizes given by max_bucket_size.
        let layer_sizes = (0..height).map(|h| (B + 1).pow(h as u32)).collect_vec();
        assert!(layer_sizes[0] > 0);
        let n_blocks = parts * layer_sizes.iter().sum::<usize>();

        let offsets = layer_sizes
            .iter()
            .scan(0, |sum, sz| {
                let offset = *sum;
                *sum += parts * sz;
                Some(offset)
            })
            .collect_vec();

        let mut tree = vec_on_hugepages::<BTreeNode<N>>(n_blocks);

        // First initialize all nodes in the layer with MAX.
        for node in &mut tree {
            node.data.fill(MAX);
        }

        // Copy the input values to the parts in the last layer.
        // offset of last layer
        let ol = offsets[height - 1];
        let mut prev_part = 0;
        let mut idx = 0;

        // Initialize the last layer.
        for &val in vals {
            let part = (val >> shift) as usize;

            // For each completed part (typically just 1), append us.
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

        // Pad the last node in the initial layer with MAX.
        if idx / B < parts * layer_sizes[height - 1] {
            tree[ol + idx / B].data[idx % B..].fill(MAX);
        }

        // Initialize the inner layers.
        for h in (0..height - 1).rev() {
            let oh = offsets[h];
            // First initialize all nodes in the layer with MAX.
            tree[oh..oh + parts * layer_sizes[h]]
                .iter_mut()
                .for_each(|node| {
                    node.data.fill(MAX);
                });

            for i in 0..B * parts * layer_sizes[h] {
                let mut k = i / B;
                let j = i % B;
                k = k * (B + 1) + j + 1;
                for _l in h..height - 2 {
                    k *= B + 1;
                }
                // TODO: Enable this again when using smaller layers.
                tree[oh + i / B].data[i % B] = if k * B < n || true {
                    tree[ol + k - 1].data[B - 1]
                } else {
                    MAX
                };
            }
        }
        assert!(offsets.len() > 0);

        Self {
            tree,
            offsets,
            shift,
        }
    }

    pub fn search<const P: usize>(&self, qb: &[u32; P]) -> [u32; P] {
        let mut k: [usize; P] = qb.map(|q| (q as usize >> self.shift) * 64);
        let q_simd = qb.map(|q| Simd::<u32, 8>::splat(q));

        let offsets = self
            .offsets
            .iter()
            .map(|o| unsafe { self.tree.as_ptr().add(*o) })
            .collect_vec();

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
