//! DXBC container checksum — the 16-byte digest fxc stores in the container
//! header (bytes 4..20). It is a *modified* MD5 over the container bytes that
//! follow the hash field (offset 20 to end): standard MD5 transform, but the
//! message length is placed at the **start** of the final block and a
//! `(numBits >> 2) | 1` marker at the **end**, rather than MD5's tail padding.
//!
//! Recomputing this lets reassembled (possibly edited) containers carry a
//! valid checksum instead of a stale one.

/// Per-round left-rotation amounts (standard MD5).
#[rustfmt::skip]
const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// Per-round additive constants (standard MD5: floor(2^32 * abs(sin(i+1)))).
#[rustfmt::skip]
const K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

fn transform(state: &mut [u32; 4], block: &[u8; 64]) {
    let mut m = [0u32; 16];
    for (i, word) in m.iter_mut().enumerate() {
        *word = u32::from_le_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }

    let (mut a, mut b, mut c, mut d) = (state[0], state[1], state[2], state[3]);
    for i in 0..64 {
        let (f, g) = if i < 16 {
            ((b & c) | (!b & d), i)
        } else if i < 32 {
            ((d & b) | (!d & c), (5 * i + 1) % 16)
        } else if i < 48 {
            (b ^ c ^ d, (3 * i + 5) % 16)
        } else {
            (c ^ (b | !d), (7 * i) % 16)
        };
        let f = f
            .wrapping_add(a)
            .wrapping_add(K[i])
            .wrapping_add(m[g]);
        a = d;
        d = c;
        c = b;
        b = b.wrapping_add(f.rotate_left(S[i]));
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
}

/// Compute the DXBC container checksum over `data` (the bytes after the header
/// hash field, i.e. container offset 20 to end).
pub fn dxbc_checksum(data: &[u8]) -> [u8; 16] {
    let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let num_bytes = data.len();
    let num_bits = (num_bytes as u32).wrapping_mul(8);

    let whole = num_bytes / 64;
    for i in 0..whole {
        let mut block = [0u8; 64];
        block.copy_from_slice(&data[i * 64..i * 64 + 64]);
        transform(&mut state, &block);
    }

    let left = num_bytes % 64;
    let tail = &data[whole * 64..];

    if left >= 56 {
        // Not enough room: a padded block, then a length/marker block.
        let mut block = [0u8; 64];
        block[..left].copy_from_slice(tail);
        block[left] = 0x80;
        transform(&mut state, &block);

        let mut last = [0u8; 64];
        last[0..4].copy_from_slice(&num_bits.to_le_bytes());
        last[60..64].copy_from_slice(&((num_bits >> 2) | 1).to_le_bytes());
        transform(&mut state, &last);
    } else {
        // One final block: length at the front, data, 0x80, marker at the end.
        let mut block = [0u8; 64];
        block[0..4].copy_from_slice(&num_bits.to_le_bytes());
        block[4..4 + left].copy_from_slice(tail);
        block[4 + left] = 0x80;
        block[60..64].copy_from_slice(&((num_bits >> 2) | 1).to_le_bytes());
        transform(&mut state, &block);
    }

    let mut out = [0u8; 16];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::dxbc_checksum;

    // Regression lock. Correctness of the algorithm itself is proven externally
    // by reproducing all 1,152 real container hashes (examples/hash_check.rs);
    // this guards against accidental changes to the transform.
    #[test]
    fn known_answer() {
        let input = b"the quick brown fox jumps over the lazy dxbc container hash";
        assert_eq!(
            dxbc_checksum(input),
            [
                0xa1, 0xb2, 0xd9, 0xd5, 0x15, 0x9d, 0x4d, 0xca, 0x96, 0x58, 0x58, 0x3e, 0x71, 0x02,
                0xff, 0x1e,
            ]
        );
    }

    #[test]
    fn deterministic_and_input_sensitive() {
        let a = dxbc_checksum(b"content-a");
        assert_eq!(a, dxbc_checksum(b"content-a"));
        assert_ne!(a, dxbc_checksum(b"content-b"));
    }

    #[test]
    fn handles_block_boundaries() {
        // Exercise the < 56, >= 56, and exact-multiple-of-64 tail paths.
        for len in [0usize, 55, 56, 63, 64, 119, 120, 128] {
            let data: alloc::vec::Vec<u8> = (0..len).map(|i| i as u8).collect();
            let h = dxbc_checksum(&data);
            assert_eq!(h, dxbc_checksum(&data), "len {len} not deterministic");
        }
    }
}
