use ark_ff::PrimeField;
use rayon::prelude::*;
use std::{cell::RefCell, sync::OnceLock};

use crate::algebra::fields::Field192;

pub type Digest = [u8; 32];

const FIELD_LEAF_DOMAIN: &[u8; 19] = b"LiLAC/field-leaf/v1";
const FIELD_NODE_DOMAIN: &[u8; 19] = b"LiLAC/field-node/v1";
const BLAKE3_IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];
const BLAKE3_CHUNK_START: u8 = 1 << 0;
const BLAKE3_CHUNK_END: u8 = 1 << 1;
const BLAKE3_ROOT: u8 = 1 << 3;
const PARALLEL_MIN_FIELDS: usize = 1 << 18;
const CHUNK_FIELDS: usize = 1 << 12;
#[cfg(target_arch = "aarch64")]
const BLAKE3_MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];
#[cfg(target_arch = "aarch64")]
const _: () = {
    let mut round = 0;
    while round < BLAKE3_MSG_SCHEDULE.len() {
        let mut word = 0;
        while word < BLAKE3_MSG_SCHEDULE[round].len() {
            assert!(BLAKE3_MSG_SCHEDULE[round][word] < 16);
            word += 1;
        }
        round += 1;
    }
};

#[cfg(target_arch = "aarch64")]
#[allow(clippy::inline_always)] // Preserve one vectorized compression body in this hot path.
mod neon4 {
    use super::{Digest, BLAKE3_IV, BLAKE3_MSG_SCHEDULE};
    use core::arch::aarch64::{
        uint32x4_t, vaddq_u32, vdupq_n_u32, veorq_u32, vld1q_u32, vorrq_u32, vreinterpretq_u32_u64,
        vreinterpretq_u64_u32, vshlq_n_u32, vshrq_n_u32, vst1q_u32, vtrn1q_u32, vtrn1q_u64,
        vtrn2q_u32, vtrn2q_u64, vuzp1q_u32, vuzp2q_u32,
    };

    #[inline(always)]
    unsafe fn set4(a: u32, b: u32, c: u32, d: u32) -> uint32x4_t {
        let words = [a, b, c, d];
        unsafe { vld1q_u32(words.as_ptr()) }
    }

    #[inline(always)]
    unsafe fn rotr16(value: uint32x4_t) -> uint32x4_t {
        unsafe { vorrq_u32(vshrq_n_u32::<16>(value), vshlq_n_u32::<16>(value)) }
    }

    #[inline(always)]
    unsafe fn rotr12(value: uint32x4_t) -> uint32x4_t {
        unsafe { vorrq_u32(vshrq_n_u32::<12>(value), vshlq_n_u32::<20>(value)) }
    }

    #[inline(always)]
    unsafe fn rotr8(value: uint32x4_t) -> uint32x4_t {
        unsafe { vorrq_u32(vshrq_n_u32::<8>(value), vshlq_n_u32::<24>(value)) }
    }

    #[inline(always)]
    unsafe fn rotr7(value: uint32x4_t) -> uint32x4_t {
        unsafe { vorrq_u32(vshrq_n_u32::<7>(value), vshlq_n_u32::<25>(value)) }
    }

    #[inline(always)]
    unsafe fn g(
        state: &mut [uint32x4_t; 16],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        x: uint32x4_t,
        y: uint32x4_t,
    ) {
        let mut va = state[a];
        let mut vb = state[b];
        let mut vc = state[c];
        let mut vd = state[d];
        unsafe {
            va = vaddq_u32(vaddq_u32(va, vb), x);
            vd = rotr16(veorq_u32(vd, va));
            vc = vaddq_u32(vc, vd);
            vb = rotr12(veorq_u32(vb, vc));
            va = vaddq_u32(vaddq_u32(va, vb), y);
            vd = rotr8(veorq_u32(vd, va));
            vc = vaddq_u32(vc, vd);
            vb = rotr7(veorq_u32(vb, vc));
        }
        state[a] = va;
        state[b] = vb;
        state[c] = vc;
        state[d] = vd;
    }

    #[inline(always)]
    unsafe fn round(state: &mut [uint32x4_t; 16], message: &[uint32x4_t; 16], round: usize) {
        let s = BLAKE3_MSG_SCHEDULE[round];
        unsafe {
            g(state, 0, 4, 8, 12, message[s[0]], message[s[1]]);
            g(state, 1, 5, 9, 13, message[s[2]], message[s[3]]);
            g(state, 2, 6, 10, 14, message[s[4]], message[s[5]]);
            g(state, 3, 7, 11, 15, message[s[6]], message[s[7]]);
            g(state, 0, 5, 10, 15, message[s[8]], message[s[9]]);
            g(state, 1, 6, 11, 12, message[s[10]], message[s[11]]);
            g(state, 2, 7, 8, 13, message[s[12]], message[s[13]]);
            g(state, 3, 4, 9, 14, message[s[14]], message[s[15]]);
        }
    }

    // Two independent four-lane states are kept in one value so LLVM can
    // schedule across their dependency chains. This is instruction-level
    // interleaving, not a wider-than-NEON vector type.
    #[derive(Clone, Copy)]
    struct Packed8([uint32x4_t; 2]);

    #[inline(always)]
    unsafe fn set8(words: [u32; 8]) -> Packed8 {
        unsafe {
            Packed8([
                set4(words[0], words[1], words[2], words[3]),
                set4(words[4], words[5], words[6], words[7]),
            ])
        }
    }

    #[inline(always)]
    unsafe fn add8(left: Packed8, right: Packed8) -> Packed8 {
        unsafe {
            Packed8([
                vaddq_u32(left.0[0], right.0[0]),
                vaddq_u32(left.0[1], right.0[1]),
            ])
        }
    }

    #[inline(always)]
    unsafe fn xor8(left: Packed8, right: Packed8) -> Packed8 {
        unsafe {
            Packed8([
                veorq_u32(left.0[0], right.0[0]),
                veorq_u32(left.0[1], right.0[1]),
            ])
        }
    }

    #[inline(always)]
    unsafe fn or8(left: Packed8, right: Packed8) -> Packed8 {
        unsafe {
            Packed8([
                vorrq_u32(left.0[0], right.0[0]),
                vorrq_u32(left.0[1], right.0[1]),
            ])
        }
    }

    #[inline(always)]
    unsafe fn shr8_8(value: Packed8) -> Packed8 {
        unsafe { Packed8([vshrq_n_u32::<8>(value.0[0]), vshrq_n_u32::<8>(value.0[1])]) }
    }

    #[inline(always)]
    unsafe fn shl24_8(value: Packed8) -> Packed8 {
        unsafe { Packed8([vshlq_n_u32::<24>(value.0[0]), vshlq_n_u32::<24>(value.0[1])]) }
    }

    #[inline(always)]
    unsafe fn rotr16_8(value: Packed8) -> Packed8 {
        unsafe { Packed8([rotr16(value.0[0]), rotr16(value.0[1])]) }
    }

    #[inline(always)]
    unsafe fn rotr12_8(value: Packed8) -> Packed8 {
        unsafe { Packed8([rotr12(value.0[0]), rotr12(value.0[1])]) }
    }

    #[inline(always)]
    unsafe fn rotr8_8(value: Packed8) -> Packed8 {
        unsafe { Packed8([rotr8(value.0[0]), rotr8(value.0[1])]) }
    }

    #[inline(always)]
    unsafe fn rotr7_8(value: Packed8) -> Packed8 {
        unsafe { Packed8([rotr7(value.0[0]), rotr7(value.0[1])]) }
    }

    #[inline(always)]
    unsafe fn g8(
        state: &mut [Packed8; 16],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        x: Packed8,
        y: Packed8,
    ) {
        let mut va = state[a];
        let mut vb = state[b];
        let mut vc = state[c];
        let mut vd = state[d];
        unsafe {
            va = add8(add8(va, vb), x);
            vd = rotr16_8(xor8(vd, va));
            vc = add8(vc, vd);
            vb = rotr12_8(xor8(vb, vc));
            va = add8(add8(va, vb), y);
            vd = rotr8_8(xor8(vd, va));
            vc = add8(vc, vd);
            vb = rotr7_8(xor8(vb, vc));
        }
        state[a] = va;
        state[b] = vb;
        state[c] = vc;
        state[d] = vd;
    }

    #[inline(always)]
    unsafe fn round8(state: &mut [Packed8; 16], message: &[Packed8; 16], round: usize) {
        let s = BLAKE3_MSG_SCHEDULE[round];
        unsafe {
            // Every compile-time schedule entry is in 0..16.  Keeping the
            // compact runtime-round loop avoids the instruction-footprint
            // regression of full unrolling, while unchecked message reads
            // remove sixteen redundant bounds checks per round.
            g8(
                state,
                0,
                4,
                8,
                12,
                *message.get_unchecked(s[0]),
                *message.get_unchecked(s[1]),
            );
            g8(
                state,
                1,
                5,
                9,
                13,
                *message.get_unchecked(s[2]),
                *message.get_unchecked(s[3]),
            );
            g8(
                state,
                2,
                6,
                10,
                14,
                *message.get_unchecked(s[4]),
                *message.get_unchecked(s[5]),
            );
            g8(
                state,
                3,
                7,
                11,
                15,
                *message.get_unchecked(s[6]),
                *message.get_unchecked(s[7]),
            );
            g8(
                state,
                0,
                5,
                10,
                15,
                *message.get_unchecked(s[8]),
                *message.get_unchecked(s[9]),
            );
            g8(
                state,
                1,
                6,
                11,
                12,
                *message.get_unchecked(s[10]),
                *message.get_unchecked(s[11]),
            );
            g8(
                state,
                2,
                7,
                8,
                13,
                *message.get_unchecked(s[12]),
                *message.get_unchecked(s[13]),
            );
            g8(
                state,
                3,
                4,
                9,
                14,
                *message.get_unchecked(s[14]),
                *message.get_unchecked(s[15]),
            );
        }
    }

    #[inline(always)]
    unsafe fn compress_message_state8_packed(
        cvs: &[Packed8; 8],
        message: &[Packed8; 16],
        block_len: u8,
        flags: u8,
    ) -> [Packed8; 16] {
        let zero = Packed8([vdupq_n_u32(0), vdupq_n_u32(0)]);
        let mut state = [zero; 16];
        state[..8].copy_from_slice(cvs);
        for word in 0..4 {
            state[8 + word] = Packed8([vdupq_n_u32(BLAKE3_IV[word]), vdupq_n_u32(BLAKE3_IV[word])]);
        }
        state[12] = zero;
        state[13] = zero;
        state[14] = Packed8([
            vdupq_n_u32(u32::from(block_len)),
            vdupq_n_u32(u32::from(block_len)),
        ]);
        state[15] = Packed8([vdupq_n_u32(u32::from(flags)), vdupq_n_u32(u32::from(flags))]);
        for index in 0..7 {
            unsafe { round8(&mut state, message, index) };
        }
        state
    }

    #[inline(always)]
    unsafe fn compress_message_state8(
        cvs: &[[u32; 8]; 8],
        message: &[Packed8; 16],
        block_len: u8,
        flags: u8,
    ) -> [Packed8; 16] {
        let packed = std::array::from_fn(|word| unsafe {
            set8(std::array::from_fn(|lane| cvs[lane][word]))
        });
        unsafe { compress_message_state8_packed(&packed, message, block_len, flags) }
    }

    #[inline(always)]
    unsafe fn transpose4x4(rows: [uint32x4_t; 4]) -> [uint32x4_t; 4] {
        let ab_even = unsafe { vtrn1q_u32(rows[0], rows[1]) };
        let ab_odd = unsafe { vtrn2q_u32(rows[0], rows[1]) };
        let cd_even = unsafe { vtrn1q_u32(rows[2], rows[3]) };
        let cd_odd = unsafe { vtrn2q_u32(rows[2], rows[3]) };
        unsafe {
            [
                vreinterpretq_u32_u64(vtrn1q_u64(
                    vreinterpretq_u64_u32(ab_even),
                    vreinterpretq_u64_u32(cd_even),
                )),
                vreinterpretq_u32_u64(vtrn1q_u64(
                    vreinterpretq_u64_u32(ab_odd),
                    vreinterpretq_u64_u32(cd_odd),
                )),
                vreinterpretq_u32_u64(vtrn2q_u64(
                    vreinterpretq_u64_u32(ab_even),
                    vreinterpretq_u64_u32(cd_even),
                )),
                vreinterpretq_u32_u64(vtrn2q_u64(
                    vreinterpretq_u64_u32(ab_odd),
                    vreinterpretq_u64_u32(cd_odd),
                )),
            ]
        }
    }

    /// Load either the left or right digest of eight parent lanes and
    /// transpose its words into the word-major representation used by the
    /// compression kernel.
    #[inline(always)]
    unsafe fn load_digest_words8(children: &[Digest; 16], parity: usize) -> [Packed8; 8] {
        let zero = Packed8([vdupq_n_u32(0), vdupq_n_u32(0)]);
        let mut output = [zero; 8];
        for group in 0..2 {
            for word_block in 0..2 {
                let rows = std::array::from_fn(|lane| unsafe {
                    vld1q_u32(
                        children[2 * (4 * group + lane) + parity]
                            .as_ptr()
                            .add(16 * word_block)
                            .cast(),
                    )
                });
                let words = unsafe { transpose4x4(rows) };
                for word in 0..4 {
                    output[4 * word_block + word].0[group] = words[word];
                }
            }
        }
        output
    }

    #[inline(always)]
    unsafe fn compress_message_xof32_8_packed<const TRANSPOSED_OUTPUT: bool>(
        cvs: &[Packed8; 8],
        message: &[Packed8; 16],
        block_len: u8,
        flags: u8,
    ) -> [Digest; 8] {
        let state = unsafe { compress_message_state8_packed(cvs, message, block_len, flags) };
        let mut output = [[0_u8; 32]; 8];
        if TRANSPOSED_OUTPUT && cfg!(target_endian = "little") {
            let values = std::array::from_fn::<_, 8, _>(|word| unsafe {
                xor8(state[word], state[word + 8])
            });
            for group in 0..2 {
                for word_block in 0..2 {
                    let words = unsafe {
                        transpose4x4(std::array::from_fn(|word| {
                            values[4 * word_block + word].0[group]
                        }))
                    };
                    for (lane, words) in words.into_iter().enumerate() {
                        unsafe {
                            vst1q_u32(
                                output[4 * group + lane]
                                    .as_mut_ptr()
                                    .add(16 * word_block)
                                    .cast(),
                                words,
                            );
                        };
                    }
                }
            }
        } else {
            for word in 0..8 {
                let value = unsafe { xor8(state[word], state[word + 8]) };
                for group in 0..2 {
                    let mut lanes = [0_u32; 4];
                    unsafe { vst1q_u32(lanes.as_mut_ptr(), value.0[group]) };
                    for (lane, value) in lanes.iter().enumerate() {
                        output[4 * group + lane][4 * word..4 * word + 4]
                            .copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
        output
    }

    #[inline(always)]
    unsafe fn compress_message_xof32_8<const TRANSPOSED_OUTPUT: bool>(
        cvs: &[[u32; 8]; 8],
        message: &[Packed8; 16],
        block_len: u8,
        flags: u8,
    ) -> [Digest; 8] {
        let packed = std::array::from_fn(|word| unsafe {
            set8(std::array::from_fn(|lane| cvs[lane][word]))
        });
        unsafe {
            compress_message_xof32_8_packed::<TRANSPOSED_OUTPUT>(&packed, message, block_len, flags)
        }
    }

    #[inline(always)]
    unsafe fn compress_message_xof32_words8_packed(
        cvs: &[Packed8; 8],
        message: &[Packed8; 16],
        block_len: u8,
        flags: u8,
    ) -> [Packed8; 8] {
        let state = unsafe { compress_message_state8_packed(cvs, message, block_len, flags) };
        std::array::from_fn(|word| unsafe { xor8(state[word], state[word + 8]) })
    }

    #[inline(always)]
    unsafe fn compress_message_xof32_words8(
        cvs: &[[u32; 8]; 8],
        message: &[Packed8; 16],
        block_len: u8,
        flags: u8,
    ) -> [Packed8; 8] {
        let packed = std::array::from_fn(|word| unsafe {
            set8(std::array::from_fn(|lane| cvs[lane][word]))
        });
        unsafe { compress_message_xof32_words8_packed(&packed, message, block_len, flags) }
    }

    #[inline(always)]
    unsafe fn compress_message_cv32_8<const TRANSPOSED_OUTPUT: bool>(
        cvs: &[[u32; 8]; 8],
        message: &[Packed8; 16],
        block_len: u8,
        flags: u8,
    ) -> [[u32; 8]; 8] {
        let state = unsafe { compress_message_state8(cvs, message, block_len, flags) };
        let mut output = [[0_u32; 8]; 8];
        if TRANSPOSED_OUTPUT && cfg!(target_endian = "little") {
            let values = std::array::from_fn::<_, 8, _>(|word| unsafe {
                xor8(state[word], state[word + 8])
            });
            for group in 0..2 {
                for word_block in 0..2 {
                    let words = unsafe {
                        transpose4x4(std::array::from_fn(|word| {
                            values[4 * word_block + word].0[group]
                        }))
                    };
                    for (lane, words) in words.into_iter().enumerate() {
                        unsafe {
                            vst1q_u32(
                                output[4 * group + lane].as_mut_ptr().add(4 * word_block),
                                words,
                            );
                        };
                    }
                }
            }
        } else {
            for word in 0..8 {
                let value = unsafe { xor8(state[word], state[word + 8]) };
                for group in 0..2 {
                    let mut lanes = [0_u32; 4];
                    unsafe { vst1q_u32(lanes.as_mut_ptr(), value.0[group]) };
                    for (lane, value) in lanes.into_iter().enumerate() {
                        output[4 * group + lane][word] = value;
                    }
                }
            }
        }
        output
    }

    #[inline(always)]
    const fn canonical_word(limbs: &[u64; 3], byte_offset: usize) -> u32 {
        let limb = byte_offset / 8;
        let shift = 8 * (byte_offset % 8);
        let low = limbs[limb] >> shift;
        let high = if shift != 0 && limb + 1 < limbs.len() {
            limbs[limb + 1] << (64 - shift)
        } else {
            0
        };
        (low | high) as u32
    }

    #[inline(always)]
    unsafe fn field_message8_scalar(canonical: &[[u64; 3]; 8]) -> [Packed8; 16] {
        let zero = Packed8([vdupq_n_u32(0), vdupq_n_u32(0)]);
        let mut message = [zero; 16];
        for (word, value) in [0x414c_694c, 0x6966_2f43, 0x2d64_6c65, 0x6661_656c]
            .into_iter()
            .enumerate()
        {
            message[word] = Packed8([vdupq_n_u32(value), vdupq_n_u32(value)]);
        }
        message[4] = unsafe {
            set8(std::array::from_fn(|lane| {
                0x0031_762f | ((canonical[lane][0] as u32 & 0xff) << 24)
            }))
        };
        for (word, byte_offset) in (5..=10).zip((1..=21).step_by(4)) {
            message[word] = unsafe {
                set8(std::array::from_fn(|lane| {
                    canonical_word(&canonical[lane], byte_offset)
                }))
            };
        }
        message
    }

    #[inline(always)]
    unsafe fn field_message8_vector(canonical: &[[u64; 3]; 8]) -> [Packed8; 16] {
        let zero = Packed8([vdupq_n_u32(0), vdupq_n_u32(0)]);
        let mut canonical_words = [zero; 6];
        for group in 0..2 {
            let rows = std::array::from_fn(|lane| unsafe {
                vld1q_u32(canonical[4 * group + lane].as_ptr().cast())
            });
            let words = unsafe { transpose4x4(rows) };
            for word in 0..4 {
                canonical_words[word].0[group] = words[word];
            }
        }
        canonical_words[4] = unsafe { set8(std::array::from_fn(|lane| canonical[lane][2] as u32)) };
        canonical_words[5] = unsafe {
            set8(std::array::from_fn(|lane| {
                (canonical[lane][2] >> 32) as u32
            }))
        };

        let mut message = [zero; 16];
        for (word, value) in [0x414c_694c, 0x6966_2f43, 0x2d64_6c65, 0x6661_656c]
            .into_iter()
            .enumerate()
        {
            message[word] = Packed8([vdupq_n_u32(value), vdupq_n_u32(value)]);
        }
        let domain_tail = Packed8([vdupq_n_u32(0x0031_762f), vdupq_n_u32(0x0031_762f)]);
        message[4] = unsafe { or8(domain_tail, shl24_8(canonical_words[0])) };
        for word in 0..5 {
            message[5 + word] = unsafe {
                or8(
                    shr8_8(canonical_words[word]),
                    shl24_8(canonical_words[word + 1]),
                )
            };
        }
        message[10] = unsafe { shr8_8(canonical_words[5]) };
        message
    }

    #[inline(always)]
    unsafe fn field_digest_words8(canonical: &[[u64; 3]; 8]) -> [Packed8; 8] {
        let message = unsafe { field_message8_vector(canonical) };
        unsafe {
            compress_message_xof32_words8(
                &[BLAKE3_IV; 8],
                &message,
                43,
                super::BLAKE3_CHUNK_START | super::BLAKE3_CHUNK_END | super::BLAKE3_ROOT,
            )
        }
    }

    /// Hash eight canonical 192-bit field encodings without first writing and
    /// then reparsing eight zero-padded 64-byte blocks.
    #[target_feature(enable = "neon")]
    unsafe fn compress_field_leaves8_impl<
        const TRANSPOSED_OUTPUT: bool,
        const VECTOR_MESSAGES: bool,
    >(
        canonical: &[[u64; 3]; 8],
    ) -> [Digest; 8] {
        let message = if VECTOR_MESSAGES && cfg!(target_endian = "little") {
            unsafe { field_message8_vector(canonical) }
        } else {
            unsafe { field_message8_scalar(canonical) }
        };
        unsafe {
            compress_message_xof32_8::<TRANSPOSED_OUTPUT>(
                &[BLAKE3_IV; 8],
                &message,
                43,
                super::BLAKE3_CHUNK_START | super::BLAKE3_CHUNK_END | super::BLAKE3_ROOT,
            )
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn compress_field_leaves8(canonical: &[[u64; 3]; 8]) -> [Digest; 8] {
        unsafe { compress_field_leaves8_impl::<true, true>(canonical) }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn compress_field_leaves8_scalar_messages(canonical: &[[u64; 3]; 8]) -> [Digest; 8] {
        unsafe { compress_field_leaves8_impl::<true, false>(canonical) }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn compress_field_leaves8_scatter(canonical: &[[u64; 3]; 8]) -> [Digest; 8] {
        unsafe { compress_field_leaves8_impl::<false, true>(canonical) }
    }

    #[inline(always)]
    fn digest_word(digest: &Digest, byte_offset: usize) -> u32 {
        u32::from_le_bytes(digest[byte_offset..byte_offset + 4].try_into().unwrap())
    }

    #[inline(always)]
    unsafe fn parent_messages8_scalar(children: &[Digest; 16]) -> ([Packed8; 16], [Packed8; 16]) {
        let zero = Packed8([vdupq_n_u32(0), vdupq_n_u32(0)]);
        let mut first = [zero; 16];
        for (word, value) in [0x414c_694c, 0x6966_2f43, 0x2d64_6c65, 0x6564_6f6e]
            .into_iter()
            .enumerate()
        {
            first[word] = Packed8([vdupq_n_u32(value), vdupq_n_u32(value)]);
        }
        first[4] = unsafe {
            set8(std::array::from_fn(|lane| {
                0x0031_762f | (u32::from(children[2 * lane][0]) << 24)
            }))
        };
        for (word, byte_offset) in (5..=11).zip((1..=25).step_by(4)) {
            first[word] = unsafe {
                set8(std::array::from_fn(|lane| {
                    digest_word(&children[2 * lane], byte_offset)
                }))
            };
        }
        first[12] = unsafe {
            set8(std::array::from_fn(|lane| {
                u32::from(children[2 * lane][29])
                    | (u32::from(children[2 * lane][30]) << 8)
                    | (u32::from(children[2 * lane][31]) << 16)
                    | (u32::from(children[2 * lane + 1][0]) << 24)
            }))
        };
        for (word, byte_offset) in (13..=15).zip((1..=9).step_by(4)) {
            first[word] = unsafe {
                set8(std::array::from_fn(|lane| {
                    digest_word(&children[2 * lane + 1], byte_offset)
                }))
            };
        }

        let mut final_block = [zero; 16];
        for (word, byte_offset) in (0..=3).zip((13..=25).step_by(4)) {
            final_block[word] = unsafe {
                set8(std::array::from_fn(|lane| {
                    digest_word(&children[2 * lane + 1], byte_offset)
                }))
            };
        }
        final_block[4] = unsafe {
            set8(std::array::from_fn(|lane| {
                u32::from(children[2 * lane + 1][29])
                    | (u32::from(children[2 * lane + 1][30]) << 8)
                    | (u32::from(children[2 * lane + 1][31]) << 16)
            }))
        };
        (first, final_block)
    }

    #[inline(always)]
    unsafe fn parent_messages8_from_words(
        left: &[Packed8; 8],
        right: &[Packed8; 8],
    ) -> ([Packed8; 16], [Packed8; 16]) {
        let zero = Packed8([vdupq_n_u32(0), vdupq_n_u32(0)]);
        let mut first = [zero; 16];
        for (word, value) in [0x414c_694c, 0x6966_2f43, 0x2d64_6c65, 0x6564_6f6e]
            .into_iter()
            .enumerate()
        {
            first[word] = Packed8([vdupq_n_u32(value), vdupq_n_u32(value)]);
        }
        let domain_tail = Packed8([vdupq_n_u32(0x0031_762f), vdupq_n_u32(0x0031_762f)]);
        first[4] = unsafe { or8(domain_tail, shl24_8(left[0])) };
        for word in 0..7 {
            first[5 + word] = unsafe { or8(shr8_8(left[word]), shl24_8(left[word + 1])) };
        }
        first[12] = unsafe { or8(shr8_8(left[7]), shl24_8(right[0])) };
        for word in 0..3 {
            first[13 + word] = unsafe { or8(shr8_8(right[word]), shl24_8(right[word + 1])) };
        }

        let mut final_block = [zero; 16];
        for word in 0..4 {
            final_block[word] = unsafe { or8(shr8_8(right[word + 3]), shl24_8(right[word + 4])) };
        }
        final_block[4] = unsafe { shr8_8(right[7]) };
        (first, final_block)
    }

    #[inline(always)]
    unsafe fn parent_messages8_vector(children: &[Digest; 16]) -> ([Packed8; 16], [Packed8; 16]) {
        let left = unsafe { load_digest_words8(children, 0) };
        let right = unsafe { load_digest_words8(children, 1) };
        unsafe { parent_messages8_from_words(&left, &right) }
    }

    #[inline(always)]
    unsafe fn pair_child_words8(
        first: &[Packed8; 8],
        second: &[Packed8; 8],
    ) -> ([Packed8; 8], [Packed8; 8]) {
        let left = std::array::from_fn(|word| {
            Packed8([
                unsafe { vuzp1q_u32(first[word].0[0], first[word].0[1]) },
                unsafe { vuzp1q_u32(second[word].0[0], second[word].0[1]) },
            ])
        });
        let right = std::array::from_fn(|word| {
            Packed8([
                unsafe { vuzp2q_u32(first[word].0[0], first[word].0[1]) },
                unsafe { vuzp2q_u32(second[word].0[0], second[word].0[1]) },
            ])
        });
        (left, right)
    }

    #[inline(always)]
    unsafe fn compress_parent_words8(left: &[Packed8; 8], right: &[Packed8; 8]) -> [Packed8; 8] {
        let (first_block, final_block) = unsafe { parent_messages8_from_words(left, right) };
        let state = unsafe {
            compress_message_state8(&[BLAKE3_IV; 8], &first_block, 64, super::BLAKE3_CHUNK_START)
        };
        let cvs = std::array::from_fn(|word| unsafe { xor8(state[word], state[word + 8]) });
        unsafe {
            compress_message_xof32_words8_packed(
                &cvs,
                &final_block,
                19,
                super::BLAKE3_CHUNK_END | super::BLAKE3_ROOT,
            )
        }
    }

    #[inline(always)]
    unsafe fn compress_parent_digests8(left: &[Packed8; 8], right: &[Packed8; 8]) -> [Digest; 8] {
        let (first_block, final_block) = unsafe { parent_messages8_from_words(left, right) };
        let state = unsafe {
            compress_message_state8(&[BLAKE3_IV; 8], &first_block, 64, super::BLAKE3_CHUNK_START)
        };
        let cvs = std::array::from_fn(|word| unsafe { xor8(state[word], state[word + 8]) });
        unsafe {
            compress_message_xof32_8_packed::<true>(
                &cvs,
                &final_block,
                19,
                super::BLAKE3_CHUNK_END | super::BLAKE3_ROOT,
            )
        }
    }

    #[inline(always)]
    unsafe fn compress_parents8_words(children: &[Digest; 16]) -> [Packed8; 8] {
        let left = unsafe { load_digest_words8(children, 0) };
        let right = unsafe { load_digest_words8(children, 1) };
        unsafe { compress_parent_words8(&left, &right) }
    }

    /// Hash sixteen field leaves and their eight immediate parents without
    /// materializing the sixteen intermediate digests in lane-major memory.
    #[target_feature(enable = "neon")]
    pub unsafe fn compress_field_level1_8(
        first: &[[u64; 3]; 8],
        second: &[[u64; 3]; 8],
    ) -> [Digest; 8] {
        let first_words = unsafe { field_digest_words8(first) };
        let second_words = unsafe { field_digest_words8(second) };
        let (left, right) = unsafe { pair_child_words8(&first_words, &second_words) };
        unsafe { compress_parent_digests8(&left, &right) }
    }

    /// Hash two groups of eight parents and their eight immediate parents
    /// without materializing the sixteen intermediate digests.
    #[target_feature(enable = "neon")]
    pub unsafe fn compress_parent_level2_8(
        first: &[Digest; 16],
        second: &[Digest; 16],
    ) -> [Digest; 8] {
        let first_words = unsafe { compress_parents8_words(first) };
        let second_words = unsafe { compress_parents8_words(second) };
        let (left, right) = unsafe { pair_child_words8(&first_words, &second_words) };
        unsafe { compress_parent_digests8(&left, &right) }
    }

    /// Hash eight 83-byte nodes directly from their child digests, avoiding
    /// sixteen temporary 64-byte blocks and their subsequent word parsing.
    #[target_feature(enable = "neon")]
    unsafe fn compress_parents8_impl<
        const TRANSPOSED_OUTPUT: bool,
        const MATERIALIZED_CV: bool,
        const VECTOR_MESSAGES: bool,
    >(
        children: &[Digest; 16],
    ) -> [Digest; 8] {
        let (first, final_block) = if VECTOR_MESSAGES && cfg!(target_endian = "little") {
            unsafe { parent_messages8_vector(children) }
        } else {
            unsafe { parent_messages8_scalar(children) }
        };
        if MATERIALIZED_CV {
            let cvs = unsafe {
                compress_message_cv32_8::<TRANSPOSED_OUTPUT>(
                    &[BLAKE3_IV; 8],
                    &first,
                    64,
                    super::BLAKE3_CHUNK_START,
                )
            };
            unsafe {
                compress_message_xof32_8::<TRANSPOSED_OUTPUT>(
                    &cvs,
                    &final_block,
                    19,
                    super::BLAKE3_CHUNK_END | super::BLAKE3_ROOT,
                )
            }
        } else {
            let state = unsafe {
                compress_message_state8(&[BLAKE3_IV; 8], &first, 64, super::BLAKE3_CHUNK_START)
            };
            let cvs = std::array::from_fn(|word| unsafe { xor8(state[word], state[word + 8]) });
            unsafe {
                compress_message_xof32_8_packed::<TRANSPOSED_OUTPUT>(
                    &cvs,
                    &final_block,
                    19,
                    super::BLAKE3_CHUNK_END | super::BLAKE3_ROOT,
                )
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn compress_parents8(children: &[Digest; 16]) -> [Digest; 8] {
        unsafe { compress_parents8_impl::<true, false, true>(children) }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn compress_parents8_scalar_messages(children: &[Digest; 16]) -> [Digest; 8] {
        unsafe { compress_parents8_impl::<true, false, false>(children) }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn compress_parents8_materialized_cv(children: &[Digest; 16]) -> [Digest; 8] {
        unsafe { compress_parents8_impl::<true, true, false>(children) }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn compress_parents8_scatter(children: &[Digest; 16]) -> [Digest; 8] {
        unsafe { compress_parents8_impl::<false, true, false>(children) }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn compress_xof32(
        cvs: &[[u32; 8]; 4],
        blocks: &[[u8; 64]; 4],
        block_len: u8,
        flags: u8,
    ) -> [Digest; 4] {
        let mut message = [vdupq_n_u32(0); 16];
        for (word, message_word) in message.iter_mut().enumerate() {
            let offset = 4 * word;
            *message_word = unsafe {
                set4(
                    u32::from_le_bytes(blocks[0][offset..offset + 4].try_into().unwrap()),
                    u32::from_le_bytes(blocks[1][offset..offset + 4].try_into().unwrap()),
                    u32::from_le_bytes(blocks[2][offset..offset + 4].try_into().unwrap()),
                    u32::from_le_bytes(blocks[3][offset..offset + 4].try_into().unwrap()),
                )
            };
        }
        let mut state = [vdupq_n_u32(0); 16];
        for word in 0..8 {
            state[word] = unsafe { set4(cvs[0][word], cvs[1][word], cvs[2][word], cvs[3][word]) };
        }
        for word in 0..4 {
            state[8 + word] = vdupq_n_u32(BLAKE3_IV[word]);
        }
        state[12] = vdupq_n_u32(0);
        state[13] = vdupq_n_u32(0);
        state[14] = vdupq_n_u32(u32::from(block_len));
        state[15] = vdupq_n_u32(u32::from(flags));
        for index in 0..7 {
            unsafe { round(&mut state, &message, index) };
        }
        let mut output = [[0_u8; 32]; 4];
        for word in 0..8 {
            let value = veorq_u32(state[word], state[word + 8]);
            let mut lanes = [0_u32; 4];
            unsafe { vst1q_u32(lanes.as_mut_ptr(), value) };
            for lane in 0..4 {
                output[lane][4 * word..4 * word + 4].copy_from_slice(&lanes[lane].to_le_bytes());
            }
        }
        output
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn compress_xof32_8(
        cvs: &[[u32; 8]; 8],
        blocks: &[[u8; 64]; 8],
        block_len: u8,
        flags: u8,
    ) -> [Digest; 8] {
        let zero = Packed8([vdupq_n_u32(0), vdupq_n_u32(0)]);
        let mut message = [zero; 16];
        for (word, message_word) in message.iter_mut().enumerate() {
            let offset = 4 * word;
            let words = std::array::from_fn(|lane| {
                u32::from_le_bytes(blocks[lane][offset..offset + 4].try_into().unwrap())
            });
            *message_word = unsafe { set8(words) };
        }
        unsafe { compress_message_xof32_8::<true>(cvs, &message, block_len, flags) }
    }
}

fn blake3_platform() -> blake3::platform::Platform {
    // `platform` is a doc-hidden benchmark API. Cargo.toml pins Blake3 1.8.3,
    // and the tests below compare both fixed transcript lengths against the
    // stable `blake3::hash` entry point over thousands of inputs.
    static PLATFORM: OnceLock<blake3::platform::Platform> = OnceLock::new();
    *PLATFORM.get_or_init(blake3::platform::Platform::detect)
}

#[inline]
fn fixed_blake3_xof4(
    cvs: &[[u32; 8]; 4],
    blocks: &[[u8; 64]; 4],
    block_len: u8,
    flags: u8,
) -> [Digest; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory in AArch64. This implements four independent
        // compression calls across vector lanes, including short final blocks
        // that Blake3's public batch API does not expose.
        unsafe { neon4::compress_xof32(cvs, blocks, block_len, flags) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let platform = blake3_platform();
        std::array::from_fn(|index| {
            let output = platform.compress_xof(&cvs[index], &blocks[index], block_len, 0, flags);
            output[..32].try_into().unwrap()
        })
    }
}

#[inline]
fn fixed_blake3_xof8(
    cvs: &[[u32; 8]; 8],
    blocks: &[[u8; 64]; 8],
    block_len: u8,
    flags: u8,
) -> [Digest; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory in AArch64; the implementation carries two
        // independent four-lane groups through the same Blake3 rounds.
        unsafe { neon4::compress_xof32_8(cvs, blocks, block_len, flags) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let first_cvs = std::array::from_fn(|index| cvs[index]);
        let first_blocks = std::array::from_fn(|index| blocks[index]);
        let second_cvs = std::array::from_fn(|index| cvs[index + 4]);
        let second_blocks = std::array::from_fn(|index| blocks[index + 4]);
        let first = fixed_blake3_xof4(&first_cvs, &first_blocks, block_len, flags);
        let second = fixed_blake3_xof4(&second_cvs, &second_blocks, block_len, flags);
        std::array::from_fn(|index| {
            if index < 4 {
                first[index]
            } else {
                second[index - 4]
            }
        })
    }
}

fn fixed_blake3_hash_43(block: &[u8; 64]) -> Digest {
    let output = blake3_platform().compress_xof(
        &BLAKE3_IV,
        block,
        43,
        0,
        BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT,
    );
    output[..32].try_into().unwrap()
}

fn fixed_blake3_hash_83(first: &[u8; 64], second: &[u8; 64]) -> Digest {
    let platform = blake3_platform();
    let mut cv = BLAKE3_IV;
    platform.compress_in_place(&mut cv, first, 64, 0, BLAKE3_CHUNK_START);
    let output = platform.compress_xof(&cv, second, 19, 0, BLAKE3_CHUNK_END | BLAKE3_ROOT);
    output[..32].try_into().unwrap()
}

fn node_blocks(left: Digest, right: Digest) -> ([u8; 64], [u8; 64]) {
    let mut first = [0_u8; 64];
    first[..FIELD_NODE_DOMAIN.len()].copy_from_slice(FIELD_NODE_DOMAIN);
    first[FIELD_NODE_DOMAIN.len()..FIELD_NODE_DOMAIN.len() + 32].copy_from_slice(&left);
    first[FIELD_NODE_DOMAIN.len() + 32..].copy_from_slice(&right[..13]);
    let mut second = [0_u8; 64];
    second[..19].copy_from_slice(&right[13..]);
    (first, second)
}

fn field_leaf_block(value: Field192) -> [u8; 64] {
    let mut input = [0_u8; 64];
    input[..FIELD_LEAF_DOMAIN.len()].copy_from_slice(FIELD_LEAF_DOMAIN);
    let bigint = value.into_bigint();
    for (index, limb) in bigint.as_ref().iter().enumerate() {
        let start = FIELD_LEAF_DOMAIN.len() + index * 8;
        input[start..start + 8].copy_from_slice(&limb.to_le_bytes());
    }
    input
}

fn field_leaves8_materialized(values: &[Field192; 8]) -> [Digest; 8] {
    let blocks = std::array::from_fn(|index| field_leaf_block(values[index]));
    fixed_blake3_xof8(
        &[BLAKE3_IV; 8],
        &blocks,
        43,
        BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT,
    )
}

#[inline]
fn field_leaves8(values: &[Field192; 8]) -> [Digest; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        let canonical = std::array::from_fn(|index| values[index].into_bigint().0);
        // AArch64 requires NEON, and the specialized entry point preserves the
        // exact 19-byte domain plus 24-byte canonical field encoding.
        unsafe { neon4::compress_field_leaves8(&canonical) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        field_leaves8_materialized(values)
    }
}

fn field_leaves8_scatter(values: &[Field192; 8]) -> [Digest; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        let canonical = std::array::from_fn(|index| values[index].into_bigint().0);
        unsafe { neon4::compress_field_leaves8_scatter(&canonical) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        field_leaves8_materialized(values)
    }
}

fn field_leaves8_scalar_messages(values: &[Field192; 8]) -> [Digest; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        let canonical = std::array::from_fn(|index| values[index].into_bigint().0);
        unsafe { neon4::compress_field_leaves8_scalar_messages(&canonical) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        field_leaves8_materialized(values)
    }
}

fn field_level1_8_unfused(values: &[Field192; 16]) -> [Digest; 8] {
    let left_values = std::array::from_fn(|index| values[index]);
    let right_values = std::array::from_fn(|index| values[8 + index]);
    let left = field_leaves8(&left_values);
    let right = field_leaves8(&right_values);
    let children = std::array::from_fn(|index| {
        if index < 8 {
            left[index]
        } else {
            right[index - 8]
        }
    });
    parent8(&children)
}

fn field_level1_8(values: &[Field192; 16]) -> [Digest; 8] {
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        let first = std::array::from_fn(|index| values[index].into_bigint().0);
        let second = std::array::from_fn(|index| values[8 + index].into_bigint().0);
        unsafe { neon4::compress_field_level1_8(&first, &second) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    {
        field_level1_8_unfused(values)
    }
}

fn parent4(children: &[Digest; 8]) -> [Digest; 4] {
    let platform = blake3_platform();
    if platform.simd_degree() < 4 {
        return std::array::from_fn(|index| parent(children[2 * index], children[2 * index + 1]));
    }
    let blocks = std::array::from_fn::<_, 4, _>(|index| {
        node_blocks(children[2 * index], children[2 * index + 1])
    });
    let first = [&blocks[0].0, &blocks[1].0, &blocks[2].0, &blocks[3].0];
    let mut chaining_values = [0_u8; 4 * 32];
    platform.hash_many(
        &first,
        &BLAKE3_IV,
        0,
        blake3::IncrementCounter::No,
        0,
        BLAKE3_CHUNK_START,
        0,
        &mut chaining_values,
    );
    let cvs = std::array::from_fn(|index| {
        let bytes = &chaining_values[index * 32..(index + 1) * 32];
        std::array::from_fn(|word| {
            u32::from_le_bytes(bytes[word * 4..(word + 1) * 4].try_into().unwrap())
        })
    });
    let final_blocks = std::array::from_fn(|index| blocks[index].1);
    fixed_blake3_xof4(&cvs, &final_blocks, 19, BLAKE3_CHUNK_END | BLAKE3_ROOT)
}

fn parent8_materialized_fixed_first_compression(children: &[Digest; 16]) -> [Digest; 8] {
    let platform = blake3_platform();
    if platform.simd_degree() < 4 {
        return std::array::from_fn(|index| parent(children[2 * index], children[2 * index + 1]));
    }
    let blocks = std::array::from_fn::<_, 8, _>(|index| {
        node_blocks(children[2 * index], children[2 * index + 1])
    });
    #[cfg(target_arch = "aarch64")]
    let cvs = {
        let first_blocks = std::array::from_fn(|index| blocks[index].0);
        let chaining_bytes =
            fixed_blake3_xof8(&[BLAKE3_IV; 8], &first_blocks, 64, BLAKE3_CHUNK_START);
        std::array::from_fn(|index| {
            std::array::from_fn(|word| {
                u32::from_le_bytes(
                    chaining_bytes[index][word * 4..(word + 1) * 4]
                        .try_into()
                        .unwrap(),
                )
            })
        })
    };
    #[cfg(not(target_arch = "aarch64"))]
    let cvs = {
        let first: [&[u8; 64]; 8] = std::array::from_fn(|index| &blocks[index].0);
        let mut chaining_values = [0_u8; 8 * 32];
        platform.hash_many(
            &first,
            &BLAKE3_IV,
            0,
            blake3::IncrementCounter::No,
            0,
            BLAKE3_CHUNK_START,
            0,
            &mut chaining_values,
        );
        std::array::from_fn(|index| {
            let bytes = &chaining_values[index * 32..(index + 1) * 32];
            std::array::from_fn(|word| {
                u32::from_le_bytes(bytes[word * 4..(word + 1) * 4].try_into().unwrap())
            })
        })
    };
    let final_blocks = std::array::from_fn(|index| blocks[index].1);
    fixed_blake3_xof8(&cvs, &final_blocks, 19, BLAKE3_CHUNK_END | BLAKE3_ROOT)
}

fn parent8(children: &[Digest; 16]) -> [Digest; 8] {
    let platform = blake3_platform();
    if platform.simd_degree() < 4 {
        return std::array::from_fn(|index| parent(children[2 * index], children[2 * index + 1]));
    }
    #[cfg(target_arch = "aarch64")]
    {
        // AArch64 requires NEON. Construct the two fixed node blocks directly
        // in vector words rather than materializing and reparsing byte arrays.
        unsafe { neon4::compress_parents8(children) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        parent8_materialized_fixed_first_compression(children)
    }
}

#[allow(dead_code)] // Retained as the exact AArch64 benchmark oracle.
fn parent_level2_8_unfused(children: &[Digest; 32]) -> [Digest; 8] {
    let first_children: &[Digest; 16] = children[..16].try_into().unwrap();
    let second_children: &[Digest; 16] = children[16..].try_into().unwrap();
    let first = parent8(first_children);
    let second = parent8(second_children);
    let intermediate = std::array::from_fn(|index| {
        if index < 8 {
            first[index]
        } else {
            second[index - 8]
        }
    });
    parent8(&intermediate)
}

fn parent_level2_8(children: &[Digest; 32]) -> [Digest; 8] {
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        let first: &[Digest; 16] = children[..16].try_into().unwrap();
        let second: &[Digest; 16] = children[16..].try_into().unwrap();
        unsafe { neon4::compress_parent_level2_8(first, second) }
    }
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    {
        parent_level2_8_unfused(children)
    }
}

fn parent8_scatter(children: &[Digest; 16]) -> [Digest; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon4::compress_parents8_scatter(children) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        parent8_materialized_fixed_first_compression(children)
    }
}

/// Reproduce the scalar-gather message-packing schedule while retaining the
/// register-resident chaining-value handoff.
fn parent8_scalar_messages(children: &[Digest; 16]) -> [Digest; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon4::compress_parents8_scalar_messages(children) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        parent8_materialized_fixed_first_compression(children)
    }
}

/// Reproduce the former AArch64 schedule that transposed the first-block
/// chaining values to lane-major memory before gathering them for block two.
fn parent8_materialized_cv(children: &[Digest; 16]) -> [Digest; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon4::compress_parents8_materialized_cv(children) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        parent8_materialized_fixed_first_compression(children)
    }
}

fn parent8_platform_first_compression(children: &[Digest; 16]) -> [Digest; 8] {
    let platform = blake3_platform();
    let blocks = std::array::from_fn::<_, 8, _>(|index| {
        node_blocks(children[2 * index], children[2 * index + 1])
    });
    let first: [&[u8; 64]; 8] = std::array::from_fn(|index| &blocks[index].0);
    let mut chaining_values = [0_u8; 8 * 32];
    platform.hash_many(
        &first,
        &BLAKE3_IV,
        0,
        blake3::IncrementCounter::No,
        0,
        BLAKE3_CHUNK_START,
        0,
        &mut chaining_values,
    );
    let cvs = std::array::from_fn(|index| {
        let bytes = &chaining_values[index * 32..(index + 1) * 32];
        std::array::from_fn(|word| {
            u32::from_le_bytes(bytes[word * 4..(word + 1) * 4].try_into().unwrap())
        })
    });
    let final_blocks = std::array::from_fn(|index| blocks[index].1);
    fixed_blake3_xof8(&cvs, &final_blocks, 19, BLAKE3_CHUNK_END | BLAKE3_ROOT)
}

pub fn field_leaf(value: Field192) -> Digest {
    fixed_blake3_hash_43(&field_leaf_block(value))
}

pub fn parent(left: Digest, right: Digest) -> Digest {
    let (first, second) = node_blocks(left, right);
    fixed_blake3_hash_83(&first, &second)
}

thread_local! {
    static EXACT_ROOT_SCRATCH: RefCell<Vec<Digest>> = const { RefCell::new(Vec::new()) };
}

fn reduce_exact_digests_unfused_parent_levels(scratch: &mut [Digest]) -> Digest {
    assert!(scratch.len() >= 8 && scratch.len().is_power_of_two());
    let mut active = scratch.len();
    while active > 1 {
        let pairs = active / 2;
        let batched8 = pairs / 8 * 8;
        for pair in (0..batched8).step_by(8) {
            let child = 2 * pair;
            let roots = {
                let children: &[Digest; 16] = scratch[child..child + 16].try_into().unwrap();
                parent8(children)
            };
            scratch[pair..pair + 8].copy_from_slice(&roots);
        }
        let batched4 = batched8 + (pairs - batched8) / 4 * 4;
        for pair in (batched8..batched4).step_by(4) {
            let child = 2 * pair;
            let roots = {
                let children: &[Digest; 8] = scratch[child..child + 8].try_into().unwrap();
                parent4(children)
            };
            scratch[pair..pair + 4].copy_from_slice(&roots);
        }
        for pair in batched4..pairs {
            scratch[pair] = parent(scratch[2 * pair], scratch[2 * pair + 1]);
        }
        active = pairs;
    }
    scratch[0]
}

fn reduce_exact_digests(scratch: &mut [Digest]) -> Digest {
    assert!(scratch.len() >= 8 && scratch.len().is_power_of_two());
    let mut active = scratch.len();
    while active >= 32 {
        let outputs = active / 4;
        for output in (0..outputs).step_by(8) {
            let child = 4 * output;
            let roots = {
                let children: &[Digest; 32] = scratch[child..child + 32].try_into().unwrap();
                parent_level2_8(children)
            };
            scratch[output..output + 8].copy_from_slice(&roots);
        }
        active = outputs;
    }
    reduce_exact_digests_unfused_parent_levels(&mut scratch[..active])
}

fn reduce_exact_digests_copied(scratch: &mut [Digest]) -> Digest {
    assert!(scratch.len() >= 8 && scratch.len().is_power_of_two());
    let mut active = scratch.len();
    while active > 1 {
        let pairs = active / 2;
        let batched8 = pairs / 8 * 8;
        for pair in (0..batched8).step_by(8) {
            let child = 2 * pair;
            let children = std::array::from_fn(|index| scratch[child + index]);
            let roots = parent8(&children);
            scratch[pair..pair + 8].copy_from_slice(&roots);
        }
        let batched4 = batched8 + (pairs - batched8) / 4 * 4;
        for pair in (batched8..batched4).step_by(4) {
            let child = 2 * pair;
            let children = std::array::from_fn(|index| scratch[child + index]);
            let roots = parent4(&children);
            scratch[pair..pair + 4].copy_from_slice(&roots);
        }
        for pair in batched4..pairs {
            scratch[pair] = parent(scratch[2 * pair], scratch[2 * pair + 1]);
        }
        active = pairs;
    }
    scratch[0]
}

fn reduce_exact_digests_materialized_cv(scratch: &mut [Digest]) -> Digest {
    assert!(scratch.len() >= 8 && scratch.len().is_power_of_two());
    let mut active = scratch.len();
    while active > 1 {
        let pairs = active / 2;
        let batched8 = pairs / 8 * 8;
        for pair in (0..batched8).step_by(8) {
            let child = 2 * pair;
            let roots = {
                let children: &[Digest; 16] = scratch[child..child + 16].try_into().unwrap();
                parent8_materialized_cv(children)
            };
            scratch[pair..pair + 8].copy_from_slice(&roots);
        }
        let batched4 = batched8 + (pairs - batched8) / 4 * 4;
        for pair in (batched8..batched4).step_by(4) {
            let child = 2 * pair;
            let roots = {
                let children: &[Digest; 8] = scratch[child..child + 8].try_into().unwrap();
                parent4(children)
            };
            scratch[pair..pair + 4].copy_from_slice(&roots);
        }
        for pair in batched4..pairs {
            scratch[pair] = parent(scratch[2 * pair], scratch[2 * pair + 1]);
        }
        active = pairs;
    }
    scratch[0]
}

fn reduce_exact_digests_scalar_messages(scratch: &mut [Digest]) -> Digest {
    assert!(scratch.len() >= 8 && scratch.len().is_power_of_two());
    let mut active = scratch.len();
    while active > 1 {
        let pairs = active / 2;
        let batched8 = pairs / 8 * 8;
        for pair in (0..batched8).step_by(8) {
            let child = 2 * pair;
            let roots = {
                let children: &[Digest; 16] = scratch[child..child + 16].try_into().unwrap();
                parent8_scalar_messages(children)
            };
            scratch[pair..pair + 8].copy_from_slice(&roots);
        }
        let batched4 = batched8 + (pairs - batched8) / 4 * 4;
        for pair in (batched8..batched4).step_by(4) {
            let child = 2 * pair;
            let roots = {
                let children: &[Digest; 8] = scratch[child..child + 8].try_into().unwrap();
                parent4(children)
            };
            scratch[pair..pair + 4].copy_from_slice(&roots);
        }
        for pair in batched4..pairs {
            scratch[pair] = parent(scratch[2 * pair], scratch[2 * pair + 1]);
        }
        active = pairs;
    }
    scratch[0]
}

fn reduce_exact_digests_scatter(scratch: &mut [Digest]) -> Digest {
    assert!(scratch.len() >= 8 && scratch.len().is_power_of_two());
    let mut active = scratch.len();
    while active > 1 {
        let pairs = active / 2;
        let batched8 = pairs / 8 * 8;
        for pair in (0..batched8).step_by(8) {
            let child = 2 * pair;
            let children = std::array::from_fn(|index| scratch[child + index]);
            let roots = parent8_scatter(&children);
            scratch[pair..pair + 8].copy_from_slice(&roots);
        }
        let batched4 = batched8 + (pairs - batched8) / 4 * 4;
        for pair in (batched8..batched4).step_by(4) {
            let child = 2 * pair;
            let children = std::array::from_fn(|index| scratch[child + index]);
            let roots = parent4(&children);
            scratch[pair..pair + 4].copy_from_slice(&roots);
        }
        for pair in batched4..pairs {
            scratch[pair] = parent(scratch[2 * pair], scratch[2 * pair + 1]);
        }
        active = pairs;
    }
    scratch[0]
}

fn reduce_exact_digests_platform_first(scratch: &mut [Digest]) -> Digest {
    assert!(scratch.len() >= 8 && scratch.len().is_power_of_two());
    let mut active = scratch.len();
    while active > 1 {
        let pairs = active / 2;
        let batched8 = pairs / 8 * 8;
        for pair in (0..batched8).step_by(8) {
            let child = 2 * pair;
            let children = std::array::from_fn(|index| scratch[child + index]);
            let roots = parent8_platform_first_compression(&children);
            scratch[pair..pair + 8].copy_from_slice(&roots);
        }
        let batched4 = batched8 + (pairs - batched8) / 4 * 4;
        for pair in (batched8..batched4).step_by(4) {
            let child = 2 * pair;
            let children = std::array::from_fn(|index| scratch[child + index]);
            let roots = parent4(&children);
            scratch[pair..pair + 4].copy_from_slice(&roots);
        }
        for pair in batched4..pairs {
            scratch[pair] = parent(scratch[2 * pair], scratch[2 * pair + 1]);
        }
        active = pairs;
    }
    scratch[0]
}

fn reduce_exact_digests_materialized_fixed_first(scratch: &mut [Digest]) -> Digest {
    assert!(scratch.len() >= 8 && scratch.len().is_power_of_two());
    let mut active = scratch.len();
    while active > 1 {
        let pairs = active / 2;
        let batched8 = pairs / 8 * 8;
        for pair in (0..batched8).step_by(8) {
            let child = 2 * pair;
            let children = std::array::from_fn(|index| scratch[child + index]);
            let roots = parent8_materialized_fixed_first_compression(&children);
            scratch[pair..pair + 8].copy_from_slice(&roots);
        }
        let batched4 = batched8 + (pairs - batched8) / 4 * 4;
        for pair in (batched8..batched4).step_by(4) {
            let child = 2 * pair;
            let children = std::array::from_fn(|index| scratch[child + index]);
            let roots = parent4(&children);
            scratch[pair..pair + 4].copy_from_slice(&roots);
        }
        for pair in batched4..pairs {
            scratch[pair] = parent(scratch[2 * pair], scratch[2 * pair + 1]);
        }
        active = pairs;
    }
    scratch[0]
}

fn extend_field_leaves_batched<F>(values: &[Field192], scratch: &mut Vec<Digest>, transform: F)
where
    F: Fn(Field192) -> Field192,
{
    let mut chunks8 = values.chunks_exact(8);
    for chunk in &mut chunks8 {
        let transformed = std::array::from_fn(|index| transform(chunk[index]));
        let hashes = field_leaves8(&transformed);
        scratch.extend_from_slice(&hashes);
    }
    let mut chunks4 = chunks8.remainder().chunks_exact(4);
    for chunk in &mut chunks4 {
        let blocks = std::array::from_fn(|index| field_leaf_block(transform(chunk[index])));
        let hashes = fixed_blake3_xof4(
            &[BLAKE3_IV; 4],
            &blocks,
            43,
            BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT,
        );
        scratch.extend_from_slice(&hashes);
    }
    scratch.extend(
        chunks4
            .remainder()
            .iter()
            .map(|value| field_leaf(transform(*value))),
    );
}

fn extend_field_level1_batched(values: &[Field192], scratch: &mut Vec<Digest>) {
    assert!(values.len() >= 16 && values.len().is_power_of_two());
    for chunk in values.chunks_exact(16) {
        let values: &[Field192; 16] = chunk.try_into().unwrap();
        scratch.extend_from_slice(&field_level1_8(values));
    }
}

fn extend_field_level1_batched_unfused_io(values: &[Field192], scratch: &mut Vec<Digest>) {
    assert!(values.len() >= 16 && values.len().is_power_of_two());
    for chunk in values.chunks_exact(16) {
        let values: &[Field192; 16] = chunk.try_into().unwrap();
        scratch.extend_from_slice(&field_level1_8_unfused(values));
    }
}

fn extend_field_level1_batched_scatter(values: &[Field192], scratch: &mut Vec<Digest>) {
    assert!(values.len() >= 16 && values.len().is_power_of_two());
    for chunk in values.chunks_exact(16) {
        let left_values = std::array::from_fn(|index| chunk[index]);
        let right_values = std::array::from_fn(|index| chunk[8 + index]);
        let left = field_leaves8_scatter(&left_values);
        let right = field_leaves8_scatter(&right_values);
        let children = std::array::from_fn(|index| {
            if index < 8 {
                left[index]
            } else {
                right[index - 8]
            }
        });
        scratch.extend_from_slice(&parent8_scatter(&children));
    }
}

fn extend_field_level1_batched_scalar_leaf_messages(
    values: &[Field192],
    scratch: &mut Vec<Digest>,
) {
    assert!(values.len() >= 16 && values.len().is_power_of_two());
    for chunk in values.chunks_exact(16) {
        let left_values = std::array::from_fn(|index| chunk[index]);
        let right_values = std::array::from_fn(|index| chunk[8 + index]);
        let left = field_leaves8_scalar_messages(&left_values);
        let right = field_leaves8_scalar_messages(&right_values);
        let children = std::array::from_fn(|index| {
            if index < 8 {
                left[index]
            } else {
                right[index - 8]
            }
        });
        scratch.extend_from_slice(&parent8(&children));
    }
}

fn extend_field_leaves_batched_materialized<F>(
    values: &[Field192],
    scratch: &mut Vec<Digest>,
    transform: F,
) where
    F: Fn(Field192) -> Field192,
{
    let mut chunks8 = values.chunks_exact(8);
    for chunk in &mut chunks8 {
        let transformed = std::array::from_fn(|index| transform(chunk[index]));
        scratch.extend_from_slice(&field_leaves8_materialized(&transformed));
    }
    let mut chunks4 = chunks8.remainder().chunks_exact(4);
    for chunk in &mut chunks4 {
        let blocks = std::array::from_fn(|index| field_leaf_block(transform(chunk[index])));
        scratch.extend_from_slice(&fixed_blake3_xof4(
            &[BLAKE3_IV; 4],
            &blocks,
            43,
            BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT,
        ));
    }
    scratch.extend(
        chunks4
            .remainder()
            .iter()
            .map(|value| field_leaf(transform(*value))),
    );
}

fn exact_prefix_root_batched(values: &[Field192]) -> Digest {
    assert!(values.len() >= 8 && values.len().is_power_of_two());
    EXACT_ROOT_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        exact_prefix_root_with_scratch(values, &mut scratch)
    })
}

/// Compute an exact field-Merkle root with caller-owned digest scratch.
///
/// This is transcript-identical to [`prefix_root`] and lets a Rayon worker
/// reuse one buffer without repeated TLS lookup and `RefCell` borrows.
pub fn exact_prefix_root_with_scratch(values: &[Field192], scratch: &mut Vec<Digest>) -> Digest {
    assert!(values.len() >= 8 && values.len().is_power_of_two());
    scratch.clear();
    if values.len() >= 16 {
        extend_field_level1_batched(values, scratch);
    } else {
        extend_field_leaves_batched(values, scratch, |value| value);
    }
    reduce_exact_digests(scratch)
}

/// Reproduce the unfused exact-root schedule for crossed artifact benchmarks.
#[doc(hidden)]
pub fn prefix_root_unfused_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 8 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_leaves_batched(values, &mut scratch, |value| value);
            reduce_exact_digests(&mut scratch)
        });
    }
    sequential_prefix_root(values, capacity, zeros)
}

/// Reproduce the pre-transpose AArch64 output-scatter schedule for crossed
/// artifact benchmarks. Other architectures use the ordinary exact path.
#[doc(hidden)]
pub fn prefix_root_scatter_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 16 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_level1_batched_scatter(values, &mut scratch);
            reduce_exact_digests_scatter(&mut scratch)
        });
    }
    prefix_root(values, capacity, zeros)
}

/// Reproduce the pre-borrowed parent-input schedule for crossed artifact
/// benchmarks. This is transcript-identical to [`prefix_root`].
#[doc(hidden)]
pub fn prefix_root_copied_parents_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 16 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_level1_batched(values, &mut scratch);
            reduce_exact_digests_copied(&mut scratch)
        });
    }
    prefix_root(values, capacity, zeros)
}

/// Reproduce the pre-register-resident chaining-value parent schedule for
/// crossed artifact benchmarks. This is transcript-identical to
/// [`prefix_root`].
#[doc(hidden)]
pub fn prefix_root_materialized_cv_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 16 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_level1_batched(values, &mut scratch);
            reduce_exact_digests_materialized_cv(&mut scratch)
        });
    }
    prefix_root(values, capacity, zeros)
}

/// Reproduce the pre-vector-load parent message-packing schedule for crossed
/// artifact benchmarks. This is transcript-identical to [`prefix_root`].
#[doc(hidden)]
pub fn prefix_root_scalar_parent_messages_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 16 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_level1_batched(values, &mut scratch);
            reduce_exact_digests_scalar_messages(&mut scratch)
        });
    }
    prefix_root(values, capacity, zeros)
}

/// Reproduce the pre-vector-load field-leaf message-packing schedule for
/// crossed artifact benchmarks. This is transcript-identical to
/// [`prefix_root`].
#[doc(hidden)]
pub fn prefix_root_scalar_leaf_messages_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 16 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_level1_batched_scalar_leaf_messages(values, &mut scratch);
            reduce_exact_digests(&mut scratch)
        });
    }
    prefix_root(values, capacity, zeros)
}

/// Reproduce the pre-fused leaf-to-parent AArch64 level-1 schedule for
/// crossed artifact benchmarks. This is transcript-identical to
/// [`prefix_root`].
#[doc(hidden)]
pub fn prefix_root_unfused_leaf_parent_io_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 16 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_level1_batched_unfused_io(values, &mut scratch);
            reduce_exact_digests(&mut scratch)
        });
    }
    prefix_root(values, capacity, zeros)
}

/// Reproduce the pre-fused upper-parent reduction schedule for crossed
/// artifact benchmarks. This is transcript-identical to [`prefix_root`].
#[doc(hidden)]
pub fn prefix_root_unfused_parent_levels_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 16 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_level1_batched(values, &mut scratch);
            reduce_exact_digests_unfused_parent_levels(&mut scratch)
        });
    }
    prefix_root(values, capacity, zeros)
}

/// Reproduce the pre-optimization AArch64 exact-root path for crossed
/// artifact benchmarks. This is transcript-identical to [`prefix_root`].
#[doc(hidden)]
pub fn prefix_root_platform_first_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 8 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_leaves_batched(values, &mut scratch, |value| value);
            reduce_exact_digests_platform_first(&mut scratch)
        });
    }
    sequential_prefix_root(values, capacity, zeros)
}

/// Reproduce the pre-optimization AArch64 field-leaf block-materialization
/// path while retaining the current parent compression for crossed artifact
/// benchmarks. This is transcript-identical to [`prefix_root`].
#[doc(hidden)]
pub fn prefix_root_materialized_leaves_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 8 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_leaves_batched_materialized(values, &mut scratch, |value| value);
            reduce_exact_digests(&mut scratch)
        });
    }
    sequential_prefix_root(values, capacity, zeros)
}

/// Reproduce the byte-materialized parent path while retaining direct field
/// leaves, for crossed artifact benchmarks.
#[doc(hidden)]
pub fn prefix_root_materialized_parents_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 8 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_leaves_batched(values, &mut scratch, |value| value);
            reduce_exact_digests_materialized_fixed_first(&mut scratch)
        });
    }
    sequential_prefix_root(values, capacity, zeros)
}

/// Reproduce the complete pre-direct-message path: materialized leaf blocks
/// and materialized parent blocks with fixed first compression.
#[doc(hidden)]
pub fn prefix_root_materialized_blocks_for_benchmark(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 8 {
        return EXACT_ROOT_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.clear();
            extend_field_leaves_batched_materialized(values, &mut scratch, |value| value);
            reduce_exact_digests_materialized_fixed_first(&mut scratch)
        });
    }
    sequential_prefix_root(values, capacity, zeros)
}

fn exact_scaled_prefix_root_batched(values: &[Field192], scale: Field192) -> Digest {
    assert!(values.len() >= 8 && values.len().is_power_of_two());
    EXACT_ROOT_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.clear();
        extend_field_leaves_batched(values, &mut scratch, |value| scale * value);
        reduce_exact_digests(&mut scratch)
    })
}

/// Compute a padded scaled prefix from exact aligned dyadic subtrees.  This
/// keeps the canonical zero-padding semantics while allowing non-power-of-two
/// prefixes to use the same batched leaf and parent kernels as exact roots.
fn dyadic_scaled_prefix_root(
    values: &[Field192],
    scale: Field192,
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() >= 8 && values.len() < capacity && capacity.is_power_of_two());
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    let mut position = 0;
    while values.len() - position >= 8 {
        let remaining = values.len() - position;
        let block = 1usize << (usize::BITS - 1 - remaining.leading_zeros());
        let root = exact_scaled_prefix_root_batched(&values[position..position + block], scale);
        accumulator.append_subtree(root, block.trailing_zeros() as usize);
        position += block;
    }
    for value in &values[position..] {
        accumulator.append_leaf(field_leaf(scale * *value));
    }
    accumulator.finish(capacity, zeros)
}

pub fn zero_roots(max_height: usize) -> Vec<Digest> {
    let mut roots = Vec::with_capacity(max_height + 1);
    roots.push(field_leaf(Field192::from(0_u64)));
    for height in 0..max_height {
        roots.push(parent(roots[height], roots[height]));
    }
    roots
}

#[derive(Debug)]
pub struct MerkleAccumulator {
    stack: Vec<Option<Digest>>,
    leaves: usize,
}

impl MerkleAccumulator {
    pub fn new(max_height: usize) -> Self {
        Self {
            stack: vec![None; max_height + 1],
            leaves: 0,
        }
    }

    pub fn append_subtree(&mut self, mut root: Digest, mut height: usize) {
        let span = 1usize << height;
        assert_eq!(self.leaves % span, 0, "unaligned appended subtree");
        self.leaves += span;
        loop {
            if self.stack[height].is_none() {
                self.stack[height] = Some(root);
                return;
            }
            let left = self.stack[height].take().unwrap();
            root = parent(left, root);
            height += 1;
        }
    }

    pub fn append_leaf(&mut self, leaf: Digest) {
        self.append_subtree(leaf, 0);
    }

    pub fn fill_zeros_to(&mut self, capacity: usize, zeros: &[Digest]) {
        assert!(capacity.is_power_of_two() && self.leaves <= capacity);
        while self.leaves < capacity {
            let remaining = capacity - self.leaves;
            let alignment = if self.leaves == 0 {
                capacity
            } else {
                1usize << self.leaves.trailing_zeros()
            };
            let span = alignment.min(1usize << (usize::BITS - 1 - remaining.leading_zeros()));
            let height = span.trailing_zeros() as usize;
            self.append_subtree(zeros[height], height);
        }
    }

    pub fn finish(mut self, capacity: usize, zeros: &[Digest]) -> Digest {
        self.fill_zeros_to(capacity, zeros);
        let height = capacity.trailing_zeros() as usize;
        assert_eq!(self.leaves, capacity);
        assert!(self
            .stack
            .iter()
            .enumerate()
            .all(|(index, value)| index == height || value.is_none()));
        self.stack[height].take().unwrap()
    }
}

pub fn prefix_root(values: &[Field192], capacity: usize, zeros: &[Digest]) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() >= PARALLEL_MIN_FIELDS {
        return chunked_prefix_root(values, capacity, zeros, CHUNK_FIELDS);
    }
    if values.len() == capacity && values.len() >= 8 {
        return exact_prefix_root_batched(values);
    }
    sequential_prefix_root(values, capacity, zeros)
}

fn sequential_prefix_root(values: &[Field192], capacity: usize, zeros: &[Digest]) -> Digest {
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    for value in values {
        accumulator.append_leaf(field_leaf(*value));
    }
    accumulator.finish(capacity, zeros)
}

fn chunked_prefix_root(
    values: &[Field192],
    capacity: usize,
    zeros: &[Digest],
    chunk_fields: usize,
) -> Digest {
    assert!(
        values.len() <= capacity
            && capacity.is_power_of_two()
            && chunk_fields.is_power_of_two()
            && chunk_fields <= capacity
    );
    let chunk_roots = values
        .par_chunks(chunk_fields)
        .map(|chunk| {
            if chunk.len() == chunk_fields && chunk.len() >= 8 {
                exact_prefix_root_batched(chunk)
            } else {
                sequential_prefix_root(chunk, chunk_fields, zeros)
            }
        })
        .collect::<Vec<_>>();
    assert!(chunk_roots.len() * chunk_fields <= capacity);
    let chunk_height = chunk_fields.trailing_zeros() as usize;
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    for root in chunk_roots {
        accumulator.append_subtree(root, chunk_height);
    }
    accumulator.finish(capacity, zeros)
}

/// Compute the ordinary field-Merkle root of `scale * values` without
/// materializing the scaled row. This preserves the exact leaf and node
/// format used by [`prefix_root`].
pub fn scaled_prefix_root(
    values: &[Field192],
    scale: Field192,
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    if values.len() == capacity && values.len() >= 8 {
        return exact_scaled_prefix_root_batched(values, scale);
    }
    if values.len() >= 8 {
        return dyadic_scaled_prefix_root(values, scale, capacity, zeros);
    }
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    for value in values {
        accumulator.append_leaf(field_leaf(scale * *value));
    }
    accumulator.finish(capacity, zeros)
}

/// Reproduce the former scalar padded-prefix path for crossed benchmarks.
#[doc(hidden)]
pub fn scaled_prefix_root_sequential_for_benchmark(
    values: &[Field192],
    scale: Field192,
    capacity: usize,
    zeros: &[Digest],
) -> Digest {
    assert!(values.len() <= capacity && capacity.is_power_of_two());
    let mut accumulator = MerkleAccumulator::new(capacity.trailing_zeros() as usize);
    for value in values {
        accumulator.append_leaf(field_leaf(scale * *value));
    }
    accumulator.finish(capacity, zeros)
}

pub fn combine_equal_subtrees(roots: &[Digest]) -> Digest {
    assert!(!roots.is_empty() && roots.len().is_power_of_two());
    match roots {
        [root] => return *root,
        [left, right] => return parent(*left, *right),
        [a, b, c, d] => return parent(parent(*a, *b), parent(*c, *d)),
        _ => {}
    }
    combine_equal_subtrees_parallel(roots)
}

fn combine_equal_subtrees_parallel(roots: &[Digest]) -> Digest {
    let mut level = roots.to_vec();
    while level.len() > 1 {
        level = level
            .par_chunks_exact(2)
            .map(|pair| parent(pair[0], pair[1]))
            .collect();
    }
    level[0]
}

/// Reproduce the allocation-heavy parallel reducer for crossed artifact
/// benchmarks. This is transcript-identical to [`combine_equal_subtrees`].
#[doc(hidden)]
pub fn combine_equal_subtrees_parallel_for_benchmark(roots: &[Digest]) -> Digest {
    assert!(!roots.is_empty() && roots.len().is_power_of_two());
    combine_equal_subtrees_parallel(roots)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_field_leaf(value: Field192) -> Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(FIELD_LEAF_DOMAIN);
        let bigint = value.into_bigint();
        for limb in bigint.as_ref() {
            hasher.update(&limb.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    #[test]
    fn small_equal_subtree_fast_paths_match_parallel_reduction() {
        let roots = (0..64)
            .map(|index| field_leaf(Field192::from(index as u64)))
            .collect::<Vec<_>>();
        for size in [1, 2, 4, 8, 16, 64] {
            assert_eq!(
                combine_equal_subtrees(&roots[..size]),
                combine_equal_subtrees_parallel(&roots[..size])
            );
        }
    }

    #[test]
    fn caller_owned_exact_root_scratch_matches_tls_path() {
        let zeros = zero_roots(12);
        let values = (0..4096)
            .map(|index| Field192::from((19 * index + 7) as u64))
            .collect::<Vec<_>>();
        let mut scratch = Vec::new();
        for size in [8, 16, 64, 256, 1024, 4096] {
            assert_eq!(
                exact_prefix_root_with_scratch(&values[..size], &mut scratch),
                prefix_root(&values[..size], size, &zeros)
            );
        }
    }

    #[test]
    fn fused_first_parent_level_matches_unfused_exact_root() {
        let zeros = zero_roots(12);
        let values = (0..4096)
            .map(|index| Field192::from((23 * index + 11) as u64))
            .collect::<Vec<_>>();
        for size in [8, 16, 64, 256, 1024, 4096] {
            assert_eq!(
                prefix_root(&values[..size], size, &zeros),
                prefix_root_unfused_for_benchmark(&values[..size], size, &zeros)
            );
        }
    }

    #[test]
    fn transposed_output_matches_scatter_exact_root() {
        let zeros = zero_roots(12);
        let values = (0..4096)
            .map(|index| Field192::from((29 * index + 13) as u64))
            .collect::<Vec<_>>();
        for size in [8, 16, 64, 256, 1024, 4096] {
            assert_eq!(
                prefix_root(&values[..size], size, &zeros),
                prefix_root_scatter_for_benchmark(&values[..size], size, &zeros)
            );
        }
    }

    #[test]
    fn borrowed_parent_inputs_match_copied_exact_root() {
        let zeros = zero_roots(12);
        let values = (0..4096)
            .map(|index| Field192::from((31 * index + 17) as u64))
            .collect::<Vec<_>>();
        for size in [16, 64, 256, 1024, 4096] {
            assert_eq!(
                prefix_root(&values[..size], size, &zeros),
                prefix_root_copied_parents_for_benchmark(&values[..size], size, &zeros)
            );
        }
    }

    #[test]
    #[ignore = "borrowed versus copied parent-input exact-root benchmark"]
    fn borrowed_parent_inputs_benchmark_copied_exact_root() {
        use std::hint::black_box;
        use std::time::Instant;

        let values = (0..1024_u64)
            .map(|index| {
                let seed = blake3::hash(&(index + 89).to_le_bytes());
                Field192::from_le_bytes_mod_order(seed.as_bytes())
            })
            .collect::<Vec<_>>();
        let zeros = zero_roots(10);
        assert_eq!(
            prefix_root(&values, values.len(), &zeros),
            prefix_root_copied_parents_for_benchmark(&values, values.len(), &zeros)
        );
        let iterations = 10_000;
        let run = |root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(root(black_box(&values), values.len(), &zeros));
            }
            start.elapsed()
        };
        let borrowed_first = run(prefix_root);
        let copied_first = run(prefix_root_copied_parents_for_benchmark);
        let copied_second = run(prefix_root_copied_parents_for_benchmark);
        let borrowed_second = run(prefix_root);
        eprintln!(
            "exact_root_1024 iterations={} borrowed-parent-inputs={:.3}/{:.3} ms copied-parent-inputs={:.3}/{:.3} ms",
            iterations,
            borrowed_first.as_secs_f64() * 1_000.0,
            borrowed_second.as_secs_f64() * 1_000.0,
            copied_first.as_secs_f64() * 1_000.0,
            copied_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "NEON transposed-output versus scalar-scatter exact-root benchmark"]
    fn transposed_output_benchmarks_scatter_exact_root() {
        use std::hint::black_box;
        use std::time::Instant;

        let values = (0..1024_u64)
            .map(|index| {
                let seed = blake3::hash(&(index + 73).to_le_bytes());
                Field192::from_le_bytes_mod_order(seed.as_bytes())
            })
            .collect::<Vec<_>>();
        let zeros = zero_roots(10);
        assert_eq!(
            prefix_root(&values, values.len(), &zeros),
            prefix_root_scatter_for_benchmark(&values, values.len(), &zeros)
        );
        let iterations = 10_000;
        let run = |root: fn(&[Field192], usize, &[Digest]) -> Digest| {
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(root(black_box(&values), values.len(), &zeros));
            }
            start.elapsed()
        };
        let transposed_first = run(prefix_root);
        let scatter_first = run(prefix_root_scatter_for_benchmark);
        let scatter_second = run(prefix_root_scatter_for_benchmark);
        let transposed_second = run(prefix_root);
        eprintln!(
            "exact_root_1024 iterations={} transposed={:.3}/{:.3} ms scatter={:.3}/{:.3} ms",
            iterations,
            transposed_first.as_secs_f64() * 1_000.0,
            transposed_second.as_secs_f64() * 1_000.0,
            scatter_first.as_secs_f64() * 1_000.0,
            scatter_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "fused versus unfused 1,024-field exact-root benchmark"]
    fn fused_first_parent_level_benchmarks_unfused_root() {
        use std::hint::black_box;
        use std::time::Instant;

        let values = (0..1024_u64)
            .map(|index| {
                let seed = blake3::hash(&(index + 41).to_le_bytes());
                Field192::from_le_bytes_mod_order(seed.as_bytes())
            })
            .collect::<Vec<_>>();
        let zeros = zero_roots(10);
        assert_eq!(
            prefix_root(&values, values.len(), &zeros),
            prefix_root_unfused_for_benchmark(&values, values.len(), &zeros)
        );
        let iterations = 10_000;
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(prefix_root(black_box(&values), values.len(), &zeros));
        }
        let fused_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(prefix_root_unfused_for_benchmark(
                black_box(&values),
                values.len(),
                &zeros,
            ));
        }
        let unfused_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(prefix_root_unfused_for_benchmark(
                black_box(&values),
                values.len(),
                &zeros,
            ));
        }
        let unfused_second = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(prefix_root(black_box(&values), values.len(), &zeros));
        }
        let fused_second = start.elapsed();
        eprintln!(
            "fused-level1={:.3}/{:.3} ms unfused={:.3}/{:.3} ms",
            fused_first.as_secs_f64() * 1_000.0,
            fused_second.as_secs_f64() * 1_000.0,
            unfused_first.as_secs_f64() * 1_000.0,
            unfused_second.as_secs_f64() * 1_000.0,
        );
    }

    fn legacy_parent(left: Digest, right: Digest) -> Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(FIELD_NODE_DOMAIN);
        hasher.update(&left);
        hasher.update(&right);
        *hasher.finalize().as_bytes()
    }

    #[test]
    fn one_shot_hashes_match_legacy_transcript_bytes() {
        let left = legacy_field_leaf(Field192::from(17_u64));
        let right = legacy_field_leaf(Field192::from(29_u64));
        assert_eq!(field_leaf(Field192::from(17_u64)), left);
        assert_eq!(field_leaf(Field192::from(29_u64)), right);
        assert_eq!(parent(left, right), legacy_parent(left, right));
    }

    #[test]
    fn fixed_length_hashes_match_blake3_over_many_inputs() {
        for index in 0..4096_u64 {
            let seed = blake3::hash(&index.to_le_bytes());
            let value = Field192::from_le_bytes_mod_order(seed.as_bytes());
            assert_eq!(field_leaf(value), legacy_field_leaf(value));

            let mut left_input = [0_u8; 16];
            left_input[..8].copy_from_slice(&index.to_le_bytes());
            left_input[8..].copy_from_slice(&index.wrapping_mul(3).to_le_bytes());
            let left = *blake3::hash(&left_input).as_bytes();
            let mut right_input = [0_u8; 16];
            right_input[..8].copy_from_slice(&index.wrapping_add(1).to_le_bytes());
            right_input[8..].copy_from_slice(&index.wrapping_mul(7).to_le_bytes());
            let right = *blake3::hash(&right_input).as_bytes();
            assert_eq!(parent(left, right), legacy_parent(left, right));
        }
    }

    #[test]
    fn four_way_short_compressions_match_scalar_blake3() {
        let platform = blake3_platform();
        for batch in 0..1024_u64 {
            let cvs = std::array::from_fn(|lane| {
                std::array::from_fn(|word| {
                    (batch as u32)
                        .wrapping_mul(0x9e37_79b9)
                        .wrapping_add((lane as u32) << 16)
                        .wrapping_add(word as u32)
                })
            });
            let blocks = std::array::from_fn(|lane| {
                std::array::from_fn(|byte| {
                    batch
                        .wrapping_mul(17)
                        .wrapping_add((lane * 64 + byte) as u64) as u8
                })
            });
            for (block_len, flags) in [
                (19, BLAKE3_CHUNK_END | BLAKE3_ROOT),
                (43, BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT),
                (64, BLAKE3_CHUNK_START),
            ] {
                let batched = fixed_blake3_xof4(&cvs, &blocks, block_len, flags);
                let scalar: [Digest; 4] = std::array::from_fn(|lane| {
                    let output =
                        platform.compress_xof(&cvs[lane], &blocks[lane], block_len, 0, flags);
                    output[..32].try_into().unwrap()
                });
                assert_eq!(batched, scalar);
            }
        }
    }

    #[test]
    fn eight_way_short_compressions_match_scalar_blake3() {
        let platform = blake3_platform();
        for batch in 0..512_u64 {
            let cvs = std::array::from_fn(|lane| {
                std::array::from_fn(|word| {
                    (batch as u32)
                        .wrapping_mul(0x85eb_ca6b)
                        .wrapping_add((lane as u32) << 16)
                        .wrapping_add(word as u32)
                })
            });
            let blocks = std::array::from_fn(|lane| {
                std::array::from_fn(|byte| {
                    batch
                        .wrapping_mul(29)
                        .wrapping_add((lane * 64 + byte) as u64) as u8
                })
            });
            for (block_len, flags) in [
                (19, BLAKE3_CHUNK_END | BLAKE3_ROOT),
                (43, BLAKE3_CHUNK_START | BLAKE3_CHUNK_END | BLAKE3_ROOT),
                (64, BLAKE3_CHUNK_START),
            ] {
                let batched = fixed_blake3_xof8(&cvs, &blocks, block_len, flags);
                let scalar: [Digest; 8] = std::array::from_fn(|lane| {
                    let output =
                        platform.compress_xof(&cvs[lane], &blocks[lane], block_len, 0, flags);
                    output[..32].try_into().unwrap()
                });
                assert_eq!(batched, scalar);
            }
        }
    }

    #[test]
    fn batched_field_leaves_match_scalar_leaves() {
        let values = (0..4099_u64)
            .map(|index| {
                let seed = blake3::hash(&index.to_le_bytes());
                Field192::from_le_bytes_mod_order(seed.as_bytes())
            })
            .collect::<Vec<_>>();
        let mut batched = Vec::new();
        extend_field_leaves_batched(&values, &mut batched, |value| value);
        let scalar = values
            .iter()
            .map(|value| field_leaf(*value))
            .collect::<Vec<_>>();
        assert_eq!(batched, scalar);
        for chunk in values[..4096].chunks_exact(8) {
            let values = std::array::from_fn(|index| chunk[index]);
            assert_eq!(
                field_leaves8_scalar_messages(&values),
                field_leaves8(&values)
            );
        }
    }

    #[test]
    #[ignore = "vector-load versus scalar-gather field-leaf message benchmark"]
    fn vector_field_messages_benchmark_scalar_gather() {
        use std::hint::black_box;
        use std::time::Instant;

        let values = std::array::from_fn(|index| {
            let seed = blake3::hash(&(index as u64 + 0x4c45_4146).to_le_bytes());
            Field192::from_le_bytes_mod_order(seed.as_bytes())
        });
        assert_eq!(
            field_leaves8(&values),
            field_leaves8_scalar_messages(&values)
        );
        let iterations = 1_000_000;
        let run = |candidate: fn(&[Field192; 8]) -> [Digest; 8]| {
            let mut values = values;
            let start = Instant::now();
            for iteration in 0..iterations {
                let hashes = candidate(black_box(&values));
                values[iteration & 7] = Field192::from_le_bytes_mod_order(&hashes[iteration & 7]);
            }
            black_box(values);
            start.elapsed()
        };

        let vector_first = run(field_leaves8);
        let scalar_first = run(field_leaves8_scalar_messages);
        let scalar_second = run(field_leaves8_scalar_messages);
        let vector_second = run(field_leaves8);
        eprintln!(
            "field-leaf8-messages iterations={iterations} vector-load={:.3}/{:.3} ms scalar-gather={:.3}/{:.3} ms",
            vector_first.as_secs_f64() * 1_000.0,
            vector_second.as_secs_f64() * 1_000.0,
            scalar_first.as_secs_f64() * 1_000.0,
            scalar_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    fn fused_field_level1_matches_unfused() {
        for batch in 0..256_u64 {
            let values = std::array::from_fn(|index| {
                let seed = blake3::hash(
                    &(batch.wrapping_mul(16).wrapping_add(index as u64)).to_le_bytes(),
                );
                Field192::from_le_bytes_mod_order(seed.as_bytes())
            });
            assert_eq!(field_level1_8(&values), field_level1_8_unfused(&values));
        }
    }

    #[test]
    #[ignore = "register-fused versus materialized leaf-to-parent benchmark"]
    fn fused_field_level1_benchmark_unfused_io() {
        use std::hint::black_box;
        use std::time::Instant;

        let values = std::array::from_fn(|index| {
            let seed = blake3::hash(&(index as u64 + 0x4c31_4655).to_le_bytes());
            Field192::from_le_bytes_mod_order(seed.as_bytes())
        });
        assert_eq!(field_level1_8(&values), field_level1_8_unfused(&values));
        let iterations = 250_000;
        let run = |candidate: fn(&[Field192; 16]) -> [Digest; 8]| {
            let mut values = values;
            let start = Instant::now();
            for iteration in 0..iterations {
                let hashes = candidate(black_box(&values));
                values[iteration & 15] = Field192::from_le_bytes_mod_order(&hashes[iteration & 7]);
            }
            black_box(values);
            start.elapsed()
        };

        let fused_first = run(field_level1_8);
        let unfused_first = run(field_level1_8_unfused);
        let unfused_second = run(field_level1_8_unfused);
        let fused_second = run(field_level1_8);
        eprintln!(
            "field-level1 iterations={iterations} fused={:.3}/{:.3} ms unfused={:.3}/{:.3} ms",
            fused_first.as_secs_f64() * 1_000.0,
            fused_second.as_secs_f64() * 1_000.0,
            unfused_first.as_secs_f64() * 1_000.0,
            unfused_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    fn fused_parent_level2_matches_unfused() {
        for batch in 0..256_u64 {
            let children = std::array::from_fn(|index| {
                *blake3::hash(&(batch.wrapping_mul(32).wrapping_add(index as u64)).to_le_bytes())
                    .as_bytes()
            });
            assert_eq!(
                parent_level2_8(&children),
                parent_level2_8_unfused(&children)
            );
        }
    }

    #[test]
    #[ignore = "register-fused versus materialized two-parent-level benchmark"]
    fn fused_parent_level2_benchmark_unfused_io() {
        use std::hint::black_box;
        use std::time::Instant;

        let children = std::array::from_fn(|index| {
            *blake3::hash(&(index as u64 + 0x5032_4655).to_le_bytes()).as_bytes()
        });
        assert_eq!(
            parent_level2_8(&children),
            parent_level2_8_unfused(&children)
        );
        let iterations = 100_000;
        let run = |candidate: fn(&[Digest; 32]) -> [Digest; 8]| {
            let mut children = children;
            let start = Instant::now();
            for iteration in 0..iterations {
                let hashes = candidate(black_box(&children));
                children[iteration & 31] = hashes[iteration & 7];
            }
            black_box(children);
            start.elapsed()
        };

        let fused_first = run(parent_level2_8);
        let unfused_first = run(parent_level2_8_unfused);
        let unfused_second = run(parent_level2_8_unfused);
        let fused_second = run(parent_level2_8);
        eprintln!(
            "parent-level2 iterations={iterations} fused={:.3}/{:.3} ms unfused={:.3}/{:.3} ms",
            fused_first.as_secs_f64() * 1_000.0,
            fused_second.as_secs_f64() * 1_000.0,
            unfused_first.as_secs_f64() * 1_000.0,
            unfused_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "direct canonical-limb field-leaf microbenchmark"]
    fn direct_field_leaves_benchmark_materialized_blocks() {
        use std::hint::black_box;
        use std::time::Instant;

        let values = std::array::from_fn(|index| {
            let seed = blake3::hash(&(index as u64 + 17).to_le_bytes());
            Field192::from_le_bytes_mod_order(seed.as_bytes())
        });
        assert_eq!(field_leaves8(&values), field_leaves8_materialized(&values));
        let iterations = 1_000_000;

        let start = Instant::now();
        for _ in 0..iterations {
            black_box(field_leaves8_materialized(black_box(&values)));
        }
        let materialized_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(field_leaves8(black_box(&values)));
        }
        let direct_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(field_leaves8(black_box(&values)));
        }
        let direct_second = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(field_leaves8_materialized(black_box(&values)));
        }
        let materialized_second = start.elapsed();

        eprintln!(
            "field_leaf8 iterations={} direct={:.3}/{:.3} ms materialized={:.3}/{:.3} ms",
            iterations,
            direct_first.as_secs_f64() * 1_000.0,
            direct_second.as_secs_f64() * 1_000.0,
            materialized_first.as_secs_f64() * 1_000.0,
            materialized_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    fn four_way_parents_match_scalar_parents() {
        for batch in 0..1024_u64 {
            let children = std::array::from_fn(|index| {
                let value = batch * 8 + index as u64;
                *blake3::hash(&value.to_le_bytes()).as_bytes()
            });
            let batched = parent4(&children);
            let scalar =
                std::array::from_fn(|index| parent(children[2 * index], children[2 * index + 1]));
            assert_eq!(batched, scalar);
        }
    }

    #[test]
    fn eight_way_parents_match_scalar_parents() {
        for batch in 0..512_u64 {
            let children = std::array::from_fn(|index| {
                let value = batch * 16 + index as u64;
                *blake3::hash(&value.to_le_bytes()).as_bytes()
            });
            let batched = parent8(&children);
            let scalar =
                std::array::from_fn(|index| parent(children[2 * index], children[2 * index + 1]));
            assert_eq!(batched, scalar);
            assert_eq!(parent8_scalar_messages(&children), scalar);
            assert_eq!(parent8_materialized_cv(&children), scalar);
        }
    }

    #[test]
    #[ignore = "vector-load versus scalar-gather parent-message benchmark"]
    fn vector_parent_messages_benchmark_scalar_gather() {
        use std::hint::black_box;
        use std::time::Instant;

        let seed_children = std::array::from_fn(|index| {
            *blake3::hash(&(index as u64 + 0x4d53_47).to_le_bytes()).as_bytes()
        });
        assert_eq!(
            parent8(&seed_children),
            parent8_scalar_messages(&seed_children)
        );
        let iterations = 1_000_000;
        let run = |candidate: fn(&[Digest; 16]) -> [Digest; 8]| {
            let mut children = seed_children;
            let start = Instant::now();
            for iteration in 0..iterations {
                let roots = candidate(black_box(&children));
                children[iteration & 15] = roots[iteration & 7];
            }
            black_box(children);
            start.elapsed()
        };

        let vector_first = run(parent8);
        let scalar_first = run(parent8_scalar_messages);
        let scalar_second = run(parent8_scalar_messages);
        let vector_second = run(parent8);
        eprintln!(
            "parent8-messages iterations={iterations} vector-load={:.3}/{:.3} ms scalar-gather={:.3}/{:.3} ms",
            vector_first.as_secs_f64() * 1_000.0,
            vector_second.as_secs_f64() * 1_000.0,
            scalar_first.as_secs_f64() * 1_000.0,
            scalar_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "register-resident versus materialized chaining-value parent benchmark"]
    fn register_resident_parent_cv_benchmarks_materialized_cv() {
        use std::hint::black_box;
        use std::time::Instant;

        let seed_children = std::array::from_fn(|index| {
            *blake3::hash(&(index as u64 + 0x4356).to_le_bytes()).as_bytes()
        });
        assert_eq!(
            parent8(&seed_children),
            parent8_materialized_cv(&seed_children)
        );
        let iterations = 1_000_000;
        let run = |candidate: fn(&[Digest; 16]) -> [Digest; 8]| {
            let mut children = seed_children;
            let start = Instant::now();
            for iteration in 0..iterations {
                let roots = candidate(black_box(&children));
                children[iteration & 15] = roots[iteration & 7];
            }
            black_box(children);
            start.elapsed()
        };

        let packed_first = run(parent8);
        let materialized_first = run(parent8_materialized_cv);
        let materialized_second = run(parent8_materialized_cv);
        let packed_second = run(parent8);
        eprintln!(
            "parent8-cv iterations={iterations} register-resident={:.3}/{:.3} ms materialized={:.3}/{:.3} ms",
            packed_first.as_secs_f64() * 1_000.0,
            packed_second.as_secs_f64() * 1_000.0,
            materialized_first.as_secs_f64() * 1_000.0,
            materialized_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "fixed-length node-compression microbenchmark"]
    fn fixed_first_compression_benchmarks_platform_hash_many() {
        use std::hint::black_box;
        use std::time::Instant;

        let mut children = std::array::from_fn(|index| {
            *blake3::hash(&(index as u64 + 1).to_le_bytes()).as_bytes()
        });
        assert_eq!(
            parent8(&children),
            parent8_platform_first_compression(&children)
        );
        assert_eq!(
            parent8(&children),
            parent8_materialized_fixed_first_compression(&children)
        );

        let iterations = 1_000_000;
        let start = Instant::now();
        for iteration in 0..iterations {
            let roots = parent8_platform_first_compression(black_box(&children));
            children[iteration & 15] = roots[iteration & 7];
        }
        let platform_first = start.elapsed();
        let start = Instant::now();
        for iteration in 0..iterations {
            let roots = parent8_materialized_fixed_first_compression(black_box(&children));
            children[iteration & 15] = roots[iteration & 7];
        }
        let materialized_first = start.elapsed();
        let start = Instant::now();
        for iteration in 0..iterations {
            let roots = parent8(black_box(&children));
            children[iteration & 15] = roots[iteration & 7];
        }
        let fixed_first = start.elapsed();
        let start = Instant::now();
        for iteration in 0..iterations {
            let roots = parent8(black_box(&children));
            children[iteration & 15] = roots[iteration & 7];
        }
        let fixed_second = start.elapsed();
        let start = Instant::now();
        for iteration in 0..iterations {
            let roots = parent8_materialized_fixed_first_compression(black_box(&children));
            children[iteration & 15] = roots[iteration & 7];
        }
        let materialized_second = start.elapsed();
        let start = Instant::now();
        for iteration in 0..iterations {
            let roots = parent8_platform_first_compression(black_box(&children));
            children[iteration & 15] = roots[iteration & 7];
        }
        let platform_second = start.elapsed();

        black_box(children);
        eprintln!(
            "parent8 iterations={} direct_messages={:.3}/{:.3} ms materialized_fixed_first={:.3}/{:.3} ms platform_hash_many={:.3}/{:.3} ms",
            iterations,
            fixed_first.as_secs_f64() * 1_000.0,
            fixed_second.as_secs_f64() * 1_000.0,
            materialized_first.as_secs_f64() * 1_000.0,
            materialized_second.as_secs_f64() * 1_000.0,
            platform_first.as_secs_f64() * 1_000.0,
            platform_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    fn exact_batched_roots_match_sequential_roots() {
        let zeros = zero_roots(12);
        let values = (0..4096)
            .map(|index| Field192::from((17 * index + 11) as u64))
            .collect::<Vec<_>>();
        for length in [8, 16, 64, 256, 1024, 4096] {
            assert_eq!(
                exact_prefix_root_batched(&values[..length]),
                sequential_prefix_root(&values[..length], length, &zeros)
            );
        }
    }

    #[test]
    #[ignore = "1,024-field exact-root reduction benchmark"]
    fn fixed_first_compression_benchmarks_complete_exact_root() {
        use std::hint::black_box;
        use std::time::Instant;

        let values = (0..1024_u64)
            .map(|index| {
                let seed = blake3::hash(&index.to_le_bytes());
                Field192::from_le_bytes_mod_order(seed.as_bytes())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            exact_prefix_root_batched(&values),
            prefix_root_platform_first_for_benchmark(&values, values.len(), &zero_roots(10))
        );
        let iterations = 10_000;
        let mut scratch = Vec::with_capacity(values.len());

        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched(black_box(&values), &mut scratch, |value| value);
            black_box(reduce_exact_digests_platform_first(&mut scratch));
        }
        let platform_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched(black_box(&values), &mut scratch, |value| value);
            black_box(reduce_exact_digests_materialized_fixed_first(&mut scratch));
        }
        let materialized_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched(black_box(&values), &mut scratch, |value| value);
            black_box(reduce_exact_digests(&mut scratch));
        }
        let fixed_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched(black_box(&values), &mut scratch, |value| value);
            black_box(reduce_exact_digests(&mut scratch));
        }
        let fixed_second = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched(black_box(&values), &mut scratch, |value| value);
            black_box(reduce_exact_digests_materialized_fixed_first(&mut scratch));
        }
        let materialized_second = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched(black_box(&values), &mut scratch, |value| value);
            black_box(reduce_exact_digests_platform_first(&mut scratch));
        }
        let platform_second = start.elapsed();

        eprintln!(
            "exact_root_1024 iterations={} direct_messages={:.3}/{:.3} ms materialized_fixed_first={:.3}/{:.3} ms platform_hash_many={:.3}/{:.3} ms",
            iterations,
            fixed_first.as_secs_f64() * 1_000.0,
            fixed_second.as_secs_f64() * 1_000.0,
            materialized_first.as_secs_f64() * 1_000.0,
            materialized_second.as_secs_f64() * 1_000.0,
            platform_first.as_secs_f64() * 1_000.0,
            platform_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    #[ignore = "direct field-leaf 1,024-field exact-root benchmark"]
    fn direct_field_leaves_benchmark_complete_exact_root() {
        use std::hint::black_box;
        use std::time::Instant;

        let values = (0..1024_u64)
            .map(|index| {
                let seed = blake3::hash(&index.to_le_bytes());
                Field192::from_le_bytes_mod_order(seed.as_bytes())
            })
            .collect::<Vec<_>>();
        let zeros = zero_roots(10);
        assert_eq!(
            exact_prefix_root_batched(&values),
            prefix_root_materialized_leaves_for_benchmark(&values, values.len(), &zeros)
        );
        assert_eq!(
            exact_prefix_root_batched(&values),
            prefix_root_materialized_blocks_for_benchmark(&values, values.len(), &zeros)
        );
        let iterations = 10_000;
        let mut scratch = Vec::with_capacity(values.len());

        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched_materialized(black_box(&values), &mut scratch, |value| {
                value
            });
            black_box(reduce_exact_digests_materialized_fixed_first(&mut scratch));
        }
        let baseline_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched_materialized(black_box(&values), &mut scratch, |value| {
                value
            });
            black_box(reduce_exact_digests(&mut scratch));
        }
        let materialized_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched(black_box(&values), &mut scratch, |value| value);
            black_box(reduce_exact_digests(&mut scratch));
        }
        let direct_first = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched(black_box(&values), &mut scratch, |value| value);
            black_box(reduce_exact_digests(&mut scratch));
        }
        let direct_second = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched_materialized(black_box(&values), &mut scratch, |value| {
                value
            });
            black_box(reduce_exact_digests(&mut scratch));
        }
        let materialized_second = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            scratch.clear();
            extend_field_leaves_batched_materialized(black_box(&values), &mut scratch, |value| {
                value
            });
            black_box(reduce_exact_digests_materialized_fixed_first(&mut scratch));
        }
        let baseline_second = start.elapsed();

        eprintln!(
            "exact_root_1024 iterations={} direct_all={:.3}/{:.3} ms materialized_leaves={:.3}/{:.3} ms materialized_all={:.3}/{:.3} ms",
            iterations,
            direct_first.as_secs_f64() * 1_000.0,
            direct_second.as_secs_f64() * 1_000.0,
            materialized_first.as_secs_f64() * 1_000.0,
            materialized_second.as_secs_f64() * 1_000.0,
            baseline_first.as_secs_f64() * 1_000.0,
            baseline_second.as_secs_f64() * 1_000.0,
        );
    }

    #[test]
    fn chunked_prefix_root_matches_sequential_root() {
        let zeros = zero_roots(8);
        let values = (0..128)
            .map(|index| Field192::from((index + 1) as u64))
            .collect::<Vec<_>>();
        for length in [0, 1, 7, 8, 9, 63, 64, 65, 127, 128] {
            assert_eq!(
                chunked_prefix_root(&values[..length], 128, &zeros, 8),
                sequential_prefix_root(&values[..length], 128, &zeros)
            );
        }
    }

    #[test]
    fn scaled_prefix_root_matches_materialized_row() {
        let zeros = zero_roots(12);
        let values = (0..4096)
            .map(|index| Field192::from((3 * index + 5) as u64))
            .collect::<Vec<_>>();
        let scale = Field192::from(17_u64);
        let scaled = values
            .iter()
            .map(|value| scale * *value)
            .collect::<Vec<_>>();
        for length in [0, 1, 7, 8, 9, 73, 128, 583, 823, 1369, 2438, 4096] {
            assert_eq!(
                scaled_prefix_root(&values[..length], scale, 4096, &zeros),
                sequential_prefix_root(&scaled[..length], 4096, &zeros)
            );
        }
    }
}
