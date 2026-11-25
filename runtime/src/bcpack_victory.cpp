// BCPack: lossless packing of BC1-BC7 block streams, as the console's
// DirectStorage hardware unpacks them. Decode only; BC2 is left out (no title
// data uses it).
//
// Ported from VIctory (github.com/marvisak/victory,
// crates/victory-formats/src/bcpack.rs, GPL-3.0), keeping its structure and
// comments. The format's constant tables (bcpack-tables.bin, 77,982 bytes) are
// not part of this source: they are supplied separately and handed to
// parse_tables at run time.
//
// A packed stream has no header. It is 12 or 16 interleaved rANS *lanes*;
// each lane is a run of fragments (header byte, length, payload) coded with
// one of 128 static models, and a fixed schedule decides which lane takes the
// next input byte. The lanes' symbols form one record per block: two back
// references (to one of the last 256 blocks, or to a per-format dictionary
// that stands in for blocks -1 to -255) and the residuals of the endpoints and
// indices against them. The models, the index rank tables, the BC7 control
// codes and the dictionaries are constants of the format, kept in the table
// file.
//
// Port notes: Rust's u8/u16 arithmetic wraps here through explicit casts back
// to the narrow type (C++ promotes to int); `as` casts are static_casts with
// the same truncation; i16 right shifts are arithmetic (C++20). Where Rust
// would shift a u16 by 16 (only reachable with a malformed control word in the
// table file) this port masks the amount the way a Rust release build does.
// No global mutable state: decode is reentrant for one shared const Tables.

#include "bcpack_victory.h"

#include <algorithm>
#include <array>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace leonida::bcpack_victory {
namespace detail {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i16 = std::int16_t;
using usize = std::size_t;

constexpr usize MODELS = 128;
constexpr usize BUCKETS = 9;
constexpr usize HISTORY = 256;
constexpr usize DICTIONARY = 255;
// rANS: decode while the state is above this, read while below.
constexpr u32 LOW = u32{1} << 24;
// Input bytes per schedule round.
constexpr usize ROUND_BYTES = 8;

// Bytes of the table file (see parse).
constexpr usize TABLES_SIZE = MODELS * BUCKETS * 2 + 256 + 64 + 256 + 64 + 64 + 512 * 2 +
                              DICTIONARY * (6 + 16 + 8 + 32 + 2 + 16 + 4 + 32 + 25 + 33 + 50 + 66);
static_assert(TABLES_SIZE == kTablesSize);

// Endpoints and indices of BC1, BC3, BC4 and BC5: small unsigned values,
// summed per element modulo the element's width.
struct Plain {
    std::array<u8, 8> ep{};
    std::array<u8, 32> ix{};
};

struct Bc6Ep {
    // Endpoint 0 R G B, then endpoints 1-3, left-aligned and biased.
    std::array<std::array<u16, 3>, 4> e{};
};

struct Bc6Ix {
    u8 mode = 0;
    std::array<i16, 16> ix{};
};

struct Bc7Ep {
    u16 ctrl = 0;
    // Per subset: R0 R1 G0 G1 B0 B1 A0 A1, left-aligned.
    std::array<std::array<u16, 8>, 3> rows{};
};

struct Bc7Ix {
    std::array<u8, 2> bits{};
    std::array<std::array<i16, 16>, 2> ix{};
};

// Rust's (Bc6Ep, Bc6Ix) and (Bc7Ep, Bc7Ix) tuples.
struct Bc6Entry {
    Bc6Ep ep;
    Bc6Ix ix;
};
struct Bc7Entry {
    Bc7Ep ep;
    Bc7Ix ix;
};

}  // namespace detail

// The format's constant tables.
struct Tables {
    std::array<std::array<detail::u32, detail::BUCKETS>, detail::MODELS> freq{};
    // Each bucket's first slot; 65536 past the model's last, and one more
    // 65536 at the end so a bucket always has a next start.
    std::array<std::array<detail::u32, detail::BUCKETS + 1>, detail::MODELS> start{};
    // Per model, the bucket of the first slot of each 256-slot window: the
    // search for a slot's bucket starts there and seldom moves on.
    std::array<std::array<detail::u8, 256>, detail::MODELS> bucket{};
    // ceil(2^32 / freq): exact division for the offsets inside a bucket.
    std::array<std::array<detail::u64, detail::BUCKETS>, detail::MODELS> recip{};
    std::array<detail::u8, 256> rank1{};
    std::array<detail::u8, 64> rank3a{};
    std::array<detail::u8, 256> rank3c{};
    std::array<detail::u8, 64> rank4{};
    std::array<detail::u8, 64> rank5{};
    std::array<detail::u16, 512> ctrl7{};
    std::array<detail::Plain, detail::DICTIONARY> bc1{};
    std::array<detail::Plain, detail::DICTIONARY> bc3{};
    std::array<detail::Plain, detail::DICTIONARY> bc4{};
    std::array<detail::Plain, detail::DICTIONARY> bc5{};
    std::array<detail::Bc6Entry, detail::DICTIONARY> bc6{};
    std::array<detail::Bc7Entry, detail::DICTIONARY> bc7{};
};

namespace detail {

struct Reader {
    const u8* p;
    const u8* end;

    // The caller checks the size first; past the end reads zeros.
    u8 byte() { return p < end ? *p++ : u8{0}; }
    template <usize N>
    std::array<u8, N> bytes() {
        std::array<u8, N> a{};
        for (auto& v : a) v = byte();
        return a;
    }
    u16 u16le() {
        const u8 lo = byte();
        const u8 hi = byte();
        return static_cast<u16>(lo | (hi << 8));
    }
};

// Reads the table file: 128 x 9 u16 model frequencies (per bucket, the
// frequency of each of its symbols), the rank tables (BC1 256 B, BC3 alpha
// 64 B and colour 256 B, BC4 64 B, BC5 64 B), 512 u16 BC7 control words, then
// 255 dictionary entries per format in format order: BC1 6 + 16 B, BC3 8 + 32
// B, BC4 2 + 16 B, BC5 4 + 32 B, BC6H (u8 mode, 12 u16, u8 mode, 16 i16), BC7
// (u16 control, 24 u16, 2 u8, 32 i16). Little-endian, no padding. Every model
// must tile 65536.
const char* parse(Tables& t, std::span<const u8> bytes) {
    if (bytes.size() != TABLES_SIZE) return "table file has the wrong size";
    Reader r{bytes.data(), bytes.data() + bytes.size()};
    for (usize m = 0; m < MODELS; ++m) {
        for (auto& f : t.freq[m]) f = r.u16le();
        // Model 16 g + shape has g + 2 buckets; bucket b > 0 holds 2^(b-1) symbols.
        const usize used = m / 16 + 2;
        u32 at = 0;
        for (usize b = 0; b < used; ++b) {
            if (t.freq[m][b] == 0) return "a model has an empty bucket";
            t.start[m][b] = at;
            const u64 f = t.freq[m][b];
            t.recip[m][b] = ((u64{1} << 32) + f - 1) / f;
            at += t.freq[m][b] << (b == 0 ? 0 : b - 1);
        }
        if (at != (u32{1} << 16)) return "a model does not tile 65536";
        for (usize b = used; b < BUCKETS + 1; ++b) t.start[m][b] = u32{1} << 16;
        for (usize window = 0; window < 256; ++window) {
            const u32 first = static_cast<u32>(window) << 8;
            usize count = 0;
            for (usize b = 1; b < BUCKETS + 1; ++b)
                if (t.start[m][b] <= first) ++count;
            t.bucket[m][window] = static_cast<u8>(count);
        }
    }
    t.rank1 = r.bytes<256>();
    t.rank3a = r.bytes<64>();
    t.rank3c = r.bytes<256>();
    t.rank4 = r.bytes<64>();
    t.rank5 = r.bytes<64>();
    for (auto& c : t.ctrl7) c = r.u16le();
    auto plain = [&r](std::array<Plain, DICTIONARY>& dict, usize ep, usize ix) {
        for (auto& p : dict) {
            p = Plain{};
            for (usize k = 0; k < ep; ++k) p.ep[k] = r.byte();
            for (usize k = 0; k < ix; ++k) p.ix[k] = r.byte();
        }
    };
    plain(t.bc1, 6, 16);
    plain(t.bc3, 8, 32);
    plain(t.bc4, 2, 16);
    plain(t.bc5, 4, 32);
    for (auto& entry : t.bc6) {
        // The endpoint entry's mode byte plays no part in decoding.
        r.byte();
        entry = Bc6Entry{};
        for (auto& row : entry.ep.e)
            for (auto& v : row) v = r.u16le();
        entry.ix.mode = r.byte();
        for (auto& v : entry.ix.ix) v = static_cast<i16>(r.u16le());
    }
    for (auto& entry : t.bc7) {
        entry = Bc7Entry{};
        entry.ep.ctrl = r.u16le();
        for (auto& row : entry.ep.rows)
            for (auto& v : row) v = r.u16le();
        entry.ix.bits[0] = r.byte();
        entry.ix.bits[1] = r.byte();
        for (auto& set : entry.ix.ix)
            for (auto& v : set) v = static_cast<i16>(r.u16le());
    }
    return nullptr;
}

// Lanes.

// The bucket holding `slot` (below 65536): the last whose start is at or below
// it. The window table gives the bucket of the window's first slot; later
// buckets that start inside the window are stepped over.
inline usize bucket_of(const Tables& t, u8 model, u32 slot) {
    const auto& start = t.start[model];
    usize b = t.bucket[model][slot >> 8];
    while (start[b + 1] <= slot) ++b;
    return b;
}

struct Lane {
    // Symbol bits - 1 at the full alphabet.
    u8 width = 0;
    // 0 header, 1 length (low byte), 2 length high byte, 3 payload.
    u8 phase = 0;
    bool long_ = false;
    bool reset = false;
    u8 model = 0;
    bool need = false;
    std::optional<u8> latch;
    std::optional<u8> out;
    bool stalled = false;
    u32 x = 0;
    u32 left = 0;

    // One step: take the latched byte (header, length or payload) and decode
    // at most one symbol. (Infallible; VIctory's returns Ok always.)
    void step(const Tables& t) {
        if (phase != 3) {
            if (!latch) return;
            const u8 b = *latch;
            latch.reset();
            need = true;
            if (phase == 0) {
                // The top two bits shrink the alphabet; width >= 3.
                const u8 group = static_cast<u8>(width - (b >> 6));
                model = static_cast<u8>(group * 16 + (b & 15));
                reset = (b & 0x10) != 0;
                long_ = (b & 0x20) != 0;
                left = 0;
                phase = 1;
            } else if (phase == 1 && long_) {
                left = b;
                phase = 2;
            } else if (phase == 1) {
                left = u32{b} + 1;
                phase = 3;
            } else {
                left = ((u32{b} << 8) | left) + 257;
                phase = 3;
            }
            return;
        }
        u32 xs = x;
        if (xs < LOW && latch) {
            const u8 b = *latch;
            latch.reset();
            left = left - 1;  // wrapping
            xs = (xs << 8) | u32{b};
        }
        need = left != 0 && !latch;
        if (xs > LOW) {
            const auto& freq = t.freq[model];
            const auto& start = t.start[model];
            const u32 slot = xs & 0xffff;
            // The bucket: starts are ascending, and 65536 past the model's last.
            const usize b = bucket_of(t, model, slot);
            // Bucket 0 holds symbol 0, which owns slots 1 .. f[0] - 1 (slot 0
            // is never used). Bucket b > 0 holds 2^(b-1) symbols of frequency
            // f[b] from start[b] on; offset and frequency are below 2^16, where
            // the reciprocal divides exactly.
            u32 sym, f, rem;
            if (b == 0) {
                sym = 0;
                f = freq[0] - 1;
                rem = slot - 1;  // wrapping
            } else {
                const u32 offset = slot - start[b];
                const u32 q = static_cast<u32>((u64{offset} * t.recip[model][b]) >> 32);
                sym = (u32{1} << (b - 1)) + q;
                f = freq[b];
                rem = offset - q * freq[b];
            }
            x = (xs >> 16) * f + rem;  // wrapping
            out = static_cast<u8>(sym);
        } else if (left != 0) {
            // At exactly 2^24 the lane neither reads nor decodes again; only a
            // lane that has delivered its symbols may end up there.
            x = xs;
        } else {
            // The fragment ends; a byte read in this step does not reach the carried state.
            phase = 0;
            need = !latch;
            if (reset) x = 0;
        }
    }
};

// Decodes `per` symbols per block on every lane into `records` (one record of
// `widths.size() * per` symbols per block), taking input from `fetch` (given
// the lane that takes the byte; returns std::optional<u8>) in the order of the
// schedule: each round the lanes step in order and take at most 8 bytes
// between them.
// shortcut (VIctory): one lane step at a time (BC1 roughly 30-60 MB/s, BC7
// 15-20 MB/s on one core); if texture streaming needs more, run the schedule
// only to split the input per lane and decode the lanes apart.
template <class Fetch>
const char* lanes(const Tables& t, std::span<const u8> widths, usize per, usize blocks, Fetch&& fetch,
                  u8* records) {
    std::array<Lane, 16> all{};
    const usize n = widths.size();
    for (usize j = 0; j < n; ++j) {
        all[j].width = widths[j];
        all[j].need = true;
    }
    const usize stride = n * per;
    const u32 full = (u32{1} << n) - 1;
    usize pos = 0;
    u32 done = 0;
    int idle = 0;
    // Where symbol `pos` of lane 0 goes; lane j's is `per * j` further on.
    usize at = 0;
    while (pos < blocks * per) {
        usize taken = 0;
        bool moved = false;
        for (usize j = 0; j < n; ++j) {
            Lane& lane = all[j];
            const u32 bit = u32{1} << j;
            const usize before = taken;
            if (!lane.stalled) lane.step(t);
            const bool waiting = lane.out.has_value() && (done & bit) != 0;
            if (lane.need && !waiting && taken < ROUND_BYTES) {
                // Past the end nothing arrives; in a valid stream only finished lanes ask there.
                if (const std::optional<u8> b = fetch(j)) {
                    lane.latch = *b;
                    ++taken;
                }
            }
            lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
            if (!lane.stalled && lane.out) {
                records[at + j * per] = *lane.out;
                lane.out.reset();
                done |= bit;
                moved = true;
            }
        }
        if (done == full) {
            ++pos;
            at += (pos % per == 0) ? stride - per + 1 : 1;
            done = 0;
            moved = true;
        }
        // Nothing read and nothing delivered for a few rounds: no lane can go on.
        idle = (taken > 0 || moved) ? 0 : idle + 1;
        if (idle > 3) return "the stream ends early or is corrupt";
    }
    return nullptr;
}

// Blocks.

// Two 4-bit symbols, the first on the even bits.
inline u8 interleave(u8 a, u8 b) {
    auto spread = [](u8 v) {
        u8 acc = 0;
        for (int k = 0; k < 4; ++k) acc = static_cast<u8>(acc | (((v >> k) & 1) << (2 * k)));
        return acc;
    };
    return static_cast<u8>(spread(a) | (spread(b) << 1));
}

// The record's endpoint and index references.
inline std::pair<usize, usize> references(const u8* s, bool paired) {
    u8 d, ix;
    if (paired) {
        d = interleave(s[0], s[1]);
        ix = interleave(s[2], s[3]);
    } else {
        d = s[0];
        ix = s[1];
    }
    const u8 delta = static_cast<u8>((d >> 1) ^ static_cast<u8>(0u - (d & 1u)));
    return {static_cast<u8>(ix + delta), ix};
}

// Block `i`'s reference `r`: 0 none, else `r` blocks back, before the first
// block into the dictionary.
template <class T>
inline const T& pick(usize i, usize r, const T* history, const T* dictionary, const T& zero) {
    if (r == 0) return zero;
    if (r <= i) return history[(i - r) % HISTORY];
    return dictionary[r - i - 1];
}

// Zigzag residual (0, -1, 1, -2, ...) modulo 2^bits.
inline u8 zz(u8 v, u32 bits) {
    const u8 mask = static_cast<u8>((u32{1} << bits) - 1);
    return static_cast<u8>(((v >> 1) ^ ((v & 1) != 0 ? mask : 0)) & mask);
}

// Element widths of the plain formats' endpoints and indices.
struct Widths {
    std::array<u8, 8> ep;
    std::array<u8, 32> ix;
    bool paired;
};

constexpr Widths widths(std::array<u8, 8> ep, usize ix3, usize ix2, bool paired) {
    std::array<u8, 32> ix{};
    for (usize k = 0; k < ix3 + ix2; ++k) ix[k] = static_cast<u8>(k < ix3 ? 3 : 2);
    return Widths{ep, ix, paired};
}

constexpr Widths BC1 = widths({5, 6, 5, 5, 6, 5, 0, 0}, 0, 16, false);
constexpr Widths BC3 = widths({5, 6, 5, 5, 6, 5, 8, 8}, 16, 16, true);
constexpr Widths BC4 = widths({8, 8, 0, 0, 0, 0, 0, 0}, 16, 0, false);
constexpr Widths BC5 = widths({8, 8, 8, 8, 0, 0, 0, 0}, 32, 0, true);

using ResidualFn = Plain (*)(const Tables&, const u8*);
using PackFn = void (*)(const Plain&, u8*);

const char* plain(const u8* records, usize stride, u8* out, usize block, usize blocks,
                  const std::array<Plain, DICTIONARY>& dictionary, const Widths& w, const Tables& t,
                  ResidualFn residual, PackFn pack) {
    std::vector<Plain> history(HISTORY);
    const Plain zero{};
    for (usize i = 0; i < blocks; ++i) {
        const u8* s = records + i * stride;
        const auto [ep_i, ix_i] = references(s, w.paired);
        const auto& ep_ref = pick(i, ep_i, history.data(), dictionary.data(), zero).ep;
        const auto& ix_ref = pick(i, ix_i, history.data(), dictionary.data(), zero).ix;
        Plain v = residual(t, s);
        for (usize k = 0; k < 8; ++k)
            v.ep[k] = static_cast<u8>(static_cast<u8>(v.ep[k] + ep_ref[k]) &
                                      static_cast<u8>((u32{1} << w.ep[k]) - 1));
        for (usize k = 0; k < 32; ++k)
            v.ix[k] = static_cast<u8>(static_cast<u8>(v.ix[k] + ix_ref[k]) &
                                      static_cast<u8>((u32{1} << w.ix[k]) - 1));
        history[i % HISTORY] = v;
        pack(v, out + i * block);
    }
    return nullptr;
}

// 2-bit index residuals: one symbol per 2x2 quad.
inline void quads(const std::array<u8, 256>& rank, const u8* s, usize count, u8* ix) {
    static constexpr usize P[4] = {0, 1, 4, 5};
    for (usize q = 0; q < count; ++q) {
        const u8 v = rank[s[q]];
        const usize base = (q >> 1) * 8 + (q & 1) * 2;
        for (usize k = 0; k < 4; ++k) ix[base + P[k]] = static_cast<u8>((v >> (2 * k)) & 3);
    }
}

// 3-bit index residuals: one symbol per two pixels in raster order.
inline void pairs(const std::array<u8, 64>& rank, const u8* s, usize count, u8* ix) {
    for (usize k = 0; k < count; ++k) {
        const u8 v = rank[s[k] & 63];
        ix[2 * k] = static_cast<u8>(v & 7);
        ix[2 * k + 1] = static_cast<u8>((v >> 4) & 7);
    }
}

Plain bc1_residual(const Tables& t, const u8* s) {
    Plain v{};
    static constexpr u32 BITS[6] = {5, 6, 5, 5, 6, 5};
    for (usize k = 0; k < 6; ++k) v.ep[k] = zz(s[2 + k], BITS[k]);
    quads(t.rank1, s + 8, 4, v.ix.data());
    return v;
}

Plain bc3_residual(const Tables& t, const u8* s) {
    // Colour lanes carry each channel's two endpoints together.
    Plain v{};
    v.ep = {zz(s[14], 5), zz(s[16], 6), zz(s[18], 5), zz(s[15], 5),
            zz(s[17], 6), zz(s[19], 5), zz(s[4], 8),  zz(s[5], 8)};
    pairs(t.rank3a, s + 6, 8, v.ix.data());
    quads(t.rank3c, s + 20, 4, v.ix.data() + 16);
    return v;
}

Plain bc4_residual(const Tables& t, const u8* s) {
    Plain v{};
    v.ep[0] = zz(s[2], 8);
    v.ep[1] = zz(s[3], 8);
    pairs(t.rank4, s + 4, 8, v.ix.data());
    return v;
}

Plain bc5_residual(const Tables& t, const u8* s) {
    Plain v{};
    for (usize k = 0; k < 4; ++k) v.ep[k] = zz(s[4 + k], 8);
    pairs(t.rank5, s + 8, 16, v.ix.data());
    return v;
}

// Indices along the endpoint line -> BC1 and BC4 codes.
constexpr u32 BC1_CODE[4] = {0, 2, 3, 1};
constexpr u64 BC4_CODE[8] = {0, 2, 3, 4, 5, 6, 7, 1};

inline void put_le(u8* out, u64 v, usize n) {
    for (usize k = 0; k < n; ++k) out[k] = static_cast<u8>(v >> (8 * k));
}

void bc1_color(const u8* ep, const u8* ix, u8* out) {
    const u16 c0 = static_cast<u16>((u32{ep[0]} << 11) | (u32{ep[1]} << 5) | u32{ep[2]});
    const u16 c1 = static_cast<u16>((u32{ep[3]} << 11) | (u32{ep[4]} << 5) | u32{ep[5]});
    u32 bits = 0;
    for (usize p = 0; p < 16; ++p) bits |= BC1_CODE[ix[p] & 3] << (2 * p);
    put_le(out, c0, 2);
    put_le(out + 2, c1, 2);
    put_le(out + 4, bits, 4);
}

// BC4 half: endpoints are kept 128 below their stored value.
void bc4_alpha(u8 e0, u8 e1, const u8* ix, u8* out) {
    u64 bits = 0;
    for (usize p = 0; p < 16; ++p) bits |= BC4_CODE[ix[p] & 7] << (16 + 3 * p);
    const u64 v = u64{static_cast<u8>(e0 ^ 0x80)} | (u64{static_cast<u8>(e1 ^ 0x80)} << 8) | bits;
    put_le(out, v, 8);
}

void bc1_pack(const Plain& v, u8* out) { bc1_color(v.ep.data(), v.ix.data(), out); }

void bc3_pack(const Plain& v, u8* out) {
    bc4_alpha(v.ep[6], v.ep[7], v.ix.data(), out);
    bc1_color(v.ep.data(), v.ix.data() + 16, out + 8);
}

void bc4_pack(const Plain& v, u8* out) { bc4_alpha(v.ep[0], v.ep[1], v.ix.data(), out); }

void bc5_pack(const Plain& v, u8* out) {
    bc4_alpha(v.ep[0], v.ep[1], v.ix.data(), out);
    bc4_alpha(v.ep[2], v.ep[3], v.ix.data() + 16, out + 8);
}

// A byte's even and odd bits as two zigzag-coded 4-bit residuals.
inline std::array<i16, 2> nibbles(u8 b, bool odd_first) {
    auto half = [](u8 v) {
        u8 acc = 0;
        for (int k = 0; k < 4; ++k) acc = static_cast<u8>(acc | (((v >> (2 * k)) & 1) << k));
        return acc;
    };
    auto z = [](u8 u) { return static_cast<i16>(static_cast<i16>(u >> 1) ^ static_cast<i16>(-(u & 1))); };
    const i16 even = z(half(b));
    const i16 odd = z(half(static_cast<u8>(b >> 1)));
    if (odd_first) return {odd, even};
    return {even, odd};
}

// Index residual plus reference, both left-aligned to 16 bits at their own
// widths, back to `bits` bits (signed).
inline i16 add_index(i16 cur, u8 bits, i16 refv, u8 ref_bits) {
    auto up = [](i16 v, u8 b) -> u16 {
        if (b == 0 || b > 16) return 0;
        return static_cast<u16>(static_cast<u16>(v) << (16 - b));
    };
    const i16 sum = static_cast<i16>(static_cast<u16>(up(cur, bits) + up(refv, ref_bits)));
    if (bits == 0) return 0;
    return static_cast<i16>(sum >> (16 - bits));
}

// Little-endian bit writer for one 128-bit block (Rust u128 as two halves).
struct Bits {
    u64 lo = 0;
    u64 hi = 0;
    u32 at = 0;

    void put(u64 value, u32 n) {
        if (n > 0 && at < 128) {
            const u64 m = n >= 64 ? value : (value & ((u64{1} << n) - 1));
            if (at < 64) {
                lo |= m << at;
                if (at > 0) hi |= m >> (64 - at);
            } else {
                hi |= m << (at - 64);
            }
        }
        at += n;
    }

    void write(u8* out) const {
        put_le(out, lo, 8);
        put_le(out + 8, hi, 8);
    }
};

// Rust u16 `>>` as a release build computes it (amount masked to 0-15).
inline u16 shr16(u16 v, u32 n) { return static_cast<u16>(v >> (n & 15)); }

// BC7.

// Public BC7 anchor indices: second subset of two; second and third of three.
constexpr u8 ANCHOR2[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2,  2,  8,  8,  2,  2,  15, 15, 6,  8,  2,  8,  15, 15, 2,  8,  2,  2, 2, 15, 15, 6, 6, 2, 6, 8,
    15, 15, 2,  2,  15, 15, 15, 15, 15, 2,  2,  15,
};
constexpr u8 ANCHOR3A[64] = {
    3, 3, 15, 15, 8, 3,  15, 15, 8,  8, 6,  6,  6,  5,  3,  3,  3, 3,  8,  15, 3, 3,
    6, 10, 5, 8,  8, 6,  8,  5,  15, 15, 8, 15, 3,  5,  6,  10, 8, 15, 15, 3,  15, 5,
    15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3,
};
constexpr u8 ANCHOR3B[64] = {
    15, 8,  8,  3,  15, 15, 3,  8,  15, 15, 15, 15, 15, 15, 15, 8,  15, 8,  15, 3,  15, 8,
    15, 8,  3,  15, 6,  10, 15, 15, 10, 8,  15, 3,  15, 10, 10, 8,  9,  10, 6,  15, 8,  15,
    3,  6,  6,  8,  15, 3,  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3,  15, 15, 8,
};

// Endpoint bits of R G B A per mode (public), before p-bits and rotation.
constexpr u8 BC7_BITS[8][4] = {
    {4, 4, 4, 0}, {6, 6, 6, 0}, {5, 5, 5, 0}, {7, 7, 7, 0},
    {5, 5, 5, 6}, {7, 7, 7, 8}, {7, 7, 7, 7}, {5, 5, 5, 5},
};
// Index bits of the primary and secondary sets per mode.
constexpr u8 BC7_INDEX[8][2] = {
    {3, 0}, {3, 0}, {2, 0}, {2, 0}, {3, 2}, {2, 2}, {4, 0}, {2, 0},
};

inline bool pbits(u32 mode) { return mode == 0 || mode == 1 || mode == 3 || mode == 6 || mode == 7; }

// Channel precisions: the p-bit is red's lowest bit; rotation swaps alpha
// with the rotated channel.
inline std::array<u8, 4> bc7_precision(u16 ctrl) {
    const u32 mode = ctrl & 7u;
    const usize rot = (ctrl >> 3) & 3u;
    std::array<u8, 4> p = {BC7_BITS[mode][0], BC7_BITS[mode][1], BC7_BITS[mode][2], BC7_BITS[mode][3]};
    if (rot != 0) {
        std::swap(p[3], p[rot - 1]);
    } else if (pbits(mode)) {
        p[0] = static_cast<u8>(p[0] + 1);
    }
    return p;
}

void bc7_pack(const Bc7Ep& ep, const Bc7Ix& ix, const std::array<u8, 4>& prec, u8* out) {
    const u16 ctrl = ep.ctrl;
    const u32 mode = ctrl & 7u;
    const usize subsets = (ctrl & 0xffu) >> 6;
    const usize part = (ctrl >> 8) & 63u;
    const usize rot = (ctrl >> 3) & 3u;
    Bits w;
    w.put(u64{1} << mode, mode + 1);
    switch (mode) {
        case 0: w.put(part, 4); break;
        case 1:
        case 2:
        case 3:
        case 7: w.put(part, 6); break;
        case 4:
            w.put(rot, 2);
            w.put((ctrl >> 5) & 1u, 1);
            break;
        case 5: w.put(rot, 2); break;
        default: break;
    }
    const usize ends = 2 * subsets;
    auto field = [&ep](usize row, usize col, u8 p) -> u64 {
        return u64{shr16(ep.rows[row][col], static_cast<u32>(16 - p))};
    };
    std::array<usize, 4> order = {0, 1, 2, 3};
    if (rot != 0) std::swap(order[3], order[rot - 1]);
    for (const usize ch : order) {
        const u8 p = prec[ch];
        if (p == 0) continue;
        const u32 drop = (ch == 0 && pbits(mode)) ? 1u : 0u;
        for (usize e = 0; e < ends; ++e) w.put(field(e / 2, 2 * ch + e % 2, p) >> drop, u32{p} - drop);
    }
    if (mode == 1) {
        for (usize row = 0; row < subsets; ++row) w.put(field(row, 0, prec[0]) & 1, 1);
    } else if (pbits(mode)) {
        for (usize e = 0; e < ends; ++e) w.put(field(e / 2, e % 2, prec[0]) & 1, 1);
    }
    u8 anchors[3] = {0, 0, 0};
    usize nanchors = 1;
    if (subsets == 2) {
        anchors[1] = ANCHOR2[part];
        nanchors = 2;
    } else if (subsets == 3) {
        anchors[1] = ANCHOR3A[part];
        anchors[2] = ANCHOR3B[part];
        nanchors = 3;
    }
    auto indices = [&](usize set) {
        const u8 bits = ix.bits[set];
        for (usize p = 0; p < 16; ++p) {
            bool anchor = false;
            for (usize a = 0; a < nanchors; ++a) anchor = anchor || anchors[a] == static_cast<u8>(p);
            const u32 n = u32{bits} - (anchor ? 1u : 0u);
            w.put(u64{static_cast<u16>(ix.ix[set][p])}, n);
        }
    };
    switch (mode) {
        case 4:
            indices(1);
            indices(0);
            break;
        case 5:
            indices(0);
            indices(1);
            break;
        default: indices(0); break;
    }
    w.write(out);
}

const char* bc7(const u8* records, usize stride, u8* out, usize blocks, const Tables& t) {
    std::vector<Bc7Entry> history(HISTORY);
    const Bc7Entry zero{};
    for (usize i = 0; i < blocks; ++i) {
        const u8* s = records + i * stride;
        const auto [ep_i, ix_i] = references(s, true);
        const u16 ctrl = t.ctrl7[(usize{static_cast<u8>(s[31] & 15)} << 5) | usize{static_cast<u8>(s[30] & 31)}];
        if (ctrl == 0) return "unused BC7 control code";
        const usize mode = ctrl & 7u;
        const usize subsets = (ctrl & 0xffu) >> 6;
        const std::array<u8, 4> prec = bc7_precision(ctrl);
        const bool three = subsets == 3;
        auto or0 = [three](u8 v) -> u8 { return three ? u8{0} : v; };
        auto if3 = [three](u8 v) -> u8 { return three ? v : u8{0}; };
        const u8 stored[3][8] = {
            {s[4], s[5], s[6], s[7], s[8], s[9], or0(s[10]), or0(s[11])},
            {s[12], s[13], s[14], s[15], s[16], s[17], or0(s[18]), or0(s[19])},
            {s[20], s[21], if3(s[10]), if3(s[11]), if3(s[18]), if3(s[19]), 0, 0},
        };
        const Bc7Ep& r = pick(i, ep_i, history.data(), t.bc7.data(), zero).ep;
        Bc7Ep ep{};
        ep.ctrl = ctrl;
        // Rows the reference lacks build on its first row.
        const usize common = std::min(subsets, usize{(r.ctrl & 0xffu) >> 6});
        const usize columns = prec[3] > 0 ? 8 : 6;
        const usize rows = std::min<usize>(subsets, 3);
        for (usize c = 0; c < columns; ++c) {
            const u8 p = prec[c / 2];
            if (p == 0) continue;
            const u16 mask = static_cast<u16>(0xffffu << (16 - p));
            for (usize row = 0; row < rows; ++row) {
                const u8 v = stored[row][c];
                const u16 residual =
                    static_cast<u16>(static_cast<u16>((v >> 1) ^ ((v & 1) != 0 ? 0xffffu : 0u)) << (16 - p));
                const u16 base = static_cast<u16>(r.rows[row < common ? row : 0][c] & mask);
                ep.rows[row][c] = static_cast<u16>(base + residual);
            }
        }
        const u8* bits = BC7_INDEX[mode];
        const Bc7Ix& rx = pick(i, ix_i, history.data(), t.bc7.data(), zero).ix;
        Bc7Ix ix{};
        ix.bits = {bits[0], bits[1]};
        static constexpr usize OFFSETS[2][4] = {{22, 24, 26, 28}, {12, 14, 16, 20}};
        for (usize set = 0; set < 2; ++set) {
            if (bits[set] == 0) continue;
            for (usize k = 0; k < 4; ++k) {
                const usize o = OFFSETS[set][k];
                for (usize b = 0; b < 2; ++b) {
                    const std::array<i16, 2> pair = nibbles(s[o + b], k < 3);
                    for (usize h = 0; h < 2; ++h) {
                        const usize p = 4 * k + 2 * b + h;
                        ix.ix[set][p] = add_index(pair[h], bits[set], rx.ix[set][p], rx.bits[set]);
                    }
                }
            }
        }
        history[i % HISTORY] = Bc7Entry{ep, ix};
        bc7_pack(ep, ix, prec, out + i * 16);
    }
    return nullptr;
}

// BC6H.

// Bits of endpoint 0, then of the other endpoints' R, G, B, per mode 1-14.
constexpr u8 BC6_BITS[15][4] = {
    {0, 0, 0, 0},  {10, 5, 5, 5},   {7, 6, 6, 6},  {11, 5, 4, 4}, {11, 4, 5, 4},
    {11, 4, 4, 5}, {9, 5, 5, 5},    {8, 6, 5, 5},  {8, 5, 6, 5},  {8, 5, 5, 6},
    {6, 6, 6, 6},  {10, 10, 10, 10}, {11, 9, 9, 9}, {12, 8, 8, 8}, {16, 4, 4, 4},
};
// Mode code and its width, modes 1-14.
struct ModeCode {
    u64 code;
    u32 n;
};
constexpr ModeCode BC6_CODE[15] = {
    {0, 0},  {0, 2},  {1, 2},  {2, 5},  {6, 5},  {10, 5}, {14, 5}, {18, 5},
    {22, 5}, {26, 5}, {30, 5}, {3, 5},  {7, 5},  {11, 5}, {15, 5},
};

// A run of the public BC6H layout: `len` bits of channel `ch` (0-2; 3 is the
// partition) of endpoint `ep`, from bit `first` up.
struct Run {
    u8 ch;
    u8 ep;
    u8 first;
    u8 len;
};

namespace bc6layout {
constexpr Run r(u8 ch, u8 ep, u8 first, u8 len) { return Run{ch, ep, first, len}; }
constexpr u8 R = 0;
constexpr u8 G = 1;
constexpr u8 B = 2;
constexpr Run D = r(3, 0, 0, 5);

// The header after the mode bits, per mode 1-14. Modes 13 and 14 write the
// high base bits in ascending order (what the console's packer does).
// clang-format off
constexpr Run M1[] = {
    r(G, 2, 4, 1), r(B, 2, 4, 1), r(B, 3, 4, 1), r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10),
    r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
    r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
    r(B, 3, 3, 1), D,
};
constexpr Run M2[] = {
    r(G, 2, 5, 1), r(G, 3, 4, 1), r(G, 3, 5, 1), r(R, 0, 0, 7), r(B, 3, 0, 1), r(B, 3, 1, 1),
    r(B, 2, 4, 1), r(G, 0, 0, 7), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 7),
    r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
    r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
};
constexpr Run M3[] = {
    r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 5), r(R, 0, 10, 1), r(G, 2, 0, 4),
    r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
    r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
};
constexpr Run M4[] = {
    r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(G, 3, 4, 1),
    r(G, 2, 0, 4), r(G, 1, 0, 5), r(G, 0, 10, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
    r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 0, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
    r(G, 2, 4, 1), r(B, 3, 3, 1), D,
};
constexpr Run M5[] = {
    r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(B, 2, 4, 1),
    r(G, 2, 0, 4), r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5),
    r(B, 0, 10, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 1, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
    r(B, 3, 4, 1), r(B, 3, 3, 1), D,
};
constexpr Run M6[] = {
    r(R, 0, 0, 9), r(B, 2, 4, 1), r(G, 0, 0, 9), r(G, 2, 4, 1), r(B, 0, 0, 9), r(B, 3, 4, 1),
    r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
    r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
    r(B, 3, 3, 1), D,
};
constexpr Run M7[] = {
    r(R, 0, 0, 8), r(G, 3, 4, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 3, 2, 1), r(G, 2, 4, 1),
    r(B, 0, 0, 8), r(B, 3, 3, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 5),
    r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 6),
    r(R, 3, 0, 6), D,
};
constexpr Run M8[] = {
    r(R, 0, 0, 8), r(B, 3, 0, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(G, 2, 5, 1), r(G, 2, 4, 1),
    r(B, 0, 0, 8), r(G, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
    r(G, 1, 0, 6), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5),
    r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
};
constexpr Run M9[] = {
    r(R, 0, 0, 8), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 2, 5, 1), r(G, 2, 4, 1),
    r(B, 0, 0, 8), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
    r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 5),
    r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
};
constexpr Run M10[] = {
    r(R, 0, 0, 6), r(G, 3, 4, 1), r(B, 3, 0, 1), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 6),
    r(G, 2, 5, 1), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 6), r(G, 3, 5, 1),
    r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
    r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
};
constexpr Run M11[] = {r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 10), r(G, 1, 0, 10), r(B, 1, 0, 10)};
constexpr Run M12[] = {
    r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 9), r(R, 0, 10, 1), r(G, 1, 0, 9),
    r(G, 0, 10, 1), r(B, 1, 0, 9), r(B, 0, 10, 1),
};
constexpr Run M13[] = {
    r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 8), r(R, 0, 10, 2), r(G, 1, 0, 8),
    r(G, 0, 10, 2), r(B, 1, 0, 8), r(B, 0, 10, 2),
};
constexpr Run M14[] = {
    r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 6), r(G, 1, 0, 4),
    r(G, 0, 10, 6), r(B, 1, 0, 4), r(B, 0, 10, 6),
};
// clang-format on
}  // namespace bc6layout

constexpr std::span<const Run> BC6_LAYOUT[15] = {
    {},
    bc6layout::M1,  bc6layout::M2,  bc6layout::M3,  bc6layout::M4,  bc6layout::M5,
    bc6layout::M6,  bc6layout::M7,  bc6layout::M8,  bc6layout::M9,  bc6layout::M10,
    bc6layout::M11, bc6layout::M12, bc6layout::M13, bc6layout::M14,
};

inline u8 bc6_index_bits(u8 mode) {
    if (mode >= 1 && mode <= 10) return 3;
    if (mode >= 11 && mode <= 14) return 4;
    return 0;
}

const char* bc6(const u8* records, usize stride, u8* out, usize blocks,
                const std::array<Bc6Entry, DICTIONARY>& dictionary) {
    std::vector<Bc6Entry> history(HISTORY);
    Bc6Entry zero{};
    zero.ix.mode = 11;
    auto z = [](u16 v) -> u16 { return static_cast<u16>((v >> 1) ^ static_cast<u16>(0u - (v & 1u))); };
    auto wide = [](u8 a, u8 b) -> u16 {
        u16 acc = 0;
        for (int k = 0; k < 8; ++k)
            acc = static_cast<u16>(acc | (((a >> k) & 1) << (2 * k)) | (((b >> k) & 1) << (2 * k + 1)));
        return acc;
    };
    for (usize i = 0; i < blocks; ++i) {
        const u8* s = records + i * stride;
        u8* block = out + i * 16;
        const auto [ep_i, ix_i] = references(s, true);
        const u8 mode = static_cast<u8>(s[30] + 1);
        if (!(mode >= 1 && mode <= 14)) return "BC6H mode out of range";
        const bool two = mode < 11;
        u32 shift[4];
        for (usize k = 0; k < 4; ++k) shift[k] = 16 - u32{BC6_BITS[mode][k]};
        const std::array<u16, 3> e0 = {z(wide(s[4], s[5])), z(wide(s[6], s[7])), z(wide(s[8], s[9]))};
        const std::array<u16, 3> e1 = {z(wide(s[10], s[11])), z(wide(s[12], s[13])), z(wide(s[14], s[15]))};
        std::array<u16, 3> e2 = e0;
        std::array<u16, 3> e3 = e1;
        if (two) {
            e2 = {z(s[16]), z(s[18]), z(s[20])};
            e3 = {z(s[17]), z(s[19]), z(s[21])};
        }
        const Bc6Ep& r = pick(i, ep_i, history.data(), dictionary.data(), zero).ep;
        Bc6Ep ep{};
        // Endpoint 0 at the base precision, the others per channel; the
        // reference is cut by the mask after the sum.
        auto mask = [](u32 sh) -> u16 { return static_cast<u16>(0xffffu << sh); };
        for (usize c = 0; c < 3; ++c)
            ep.e[0][c] = static_cast<u16>(static_cast<u16>(r.e[0][c] + static_cast<u16>(e0[c] << shift[0])) &
                                          mask(shift[0]));
        const std::array<u16, 3>* res_of[3] = {&e1, &e2, &e3};
        for (usize k = 0; k < (two ? 3u : 1u); ++k) {
            const std::array<u16, 3>& res = *res_of[k];
            for (usize c = 0; c < 3; ++c)
                ep.e[k + 1][c] = static_cast<u16>(
                    static_cast<u16>(r.e[k + 1][c] + static_cast<u16>(res[c] << shift[1 + c])) & mask(shift[1 + c]));
        }
        if (!two) {
            for (usize c = 0; c < 3; ++c)
                ep.e[2][c] = static_cast<u16>(static_cast<u16>(ep.e[0][c] >> shift[0]) << shift[1]);
            ep.e[3] = ep.e[1];
        }
        const Bc6Ix& rx = pick(i, ix_i, history.data(), dictionary.data(), zero).ix;
        const u8 bits = bc6_index_bits(mode);
        Bc6Ix ix{};
        ix.mode = mode;
        for (usize k = 0; k < 8; ++k) {
            const std::array<i16, 2> pair = nibbles(s[22 + k], k < 6);
            for (usize h = 0; h < 2; ++h) {
                const usize p = 2 * k + h;
                ix.ix[p] = add_index(pair[h], bits, rx.ix[p], bc6_index_bits(rx.mode));
            }
        }
        history[i % HISTORY] = Bc6Entry{ep, ix};
        // Fields: the working value has its top bit flipped.
        auto field = [&ep, &shift](usize k, usize c) -> u64 {
            const u32 sh = k == 0 ? shift[0] : shift[1 + c];
            return u64{static_cast<u16>(static_cast<u16>(ep.e[k][c] ^ 0x8000u) >> sh)};
        };
        Bits w;
        w.put(BC6_CODE[mode].code, BC6_CODE[mode].n);
        const u8 part = static_cast<u8>(s[31] & 31);
        for (const Run& run : BC6_LAYOUT[mode]) {
            if (run.ch == 3)
                w.put(part, 5);
            else
                w.put(field(run.ep, run.ch) >> run.first, run.len);
        }
        const u8 anchor = two ? ANCHOR2[part] : u8{0};
        for (usize p = 0; p < 16; ++p) {
            const bool is_short = p == 0 || p == anchor;
            const u16 v = static_cast<u16>(static_cast<u16>(ix.ix[p]) + static_cast<u16>(1u << (bits - 1)));
            w.put(v, u32{bits} - (is_short ? 1u : 0u));
        }
        w.write(block);
    }
    return nullptr;
}

// Lane widths and symbols per block of each format.
struct Layout {
    std::span<const u8> widths;
    usize per;
};

constexpr u8 L1[] = {7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7};
constexpr u8 L3[] = {3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7};
constexpr u8 L4[] = {7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5};
constexpr u8 L5[] = {3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5};
constexpr u8 L6[] = {3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4};
constexpr u8 L7[] = {3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4};

// Unpacks `src` (BC number `bc`, 1 and 3-7) into `dst`, whose length gives the
// block count. The stream carries no size: bytes after the last one the lanes
// need are ignored, a stream that runs out is an error.
const char* decode(unsigned bc, std::span<const u8> src, std::span<u8> dst, const Tables& t) {
    usize block = 0;
    switch (bc) {
        case 1:
        case 4: block = 8; break;
        case 3:
        case 5:
        case 6:
        case 7: block = 16; break;
        default: return "unsupported BC format";
    }
    if (dst.empty() || dst.size() % block != 0) return "destination is not a whole number of blocks";
    const usize blocks = dst.size() / block;
    Layout layout;
    switch (bc) {
        case 1: layout = {L1, 1}; break;
        case 3: layout = {L3, 2}; break;
        case 4: layout = {L4, 1}; break;
        case 5: layout = {L5, 2}; break;
        case 6: layout = {L6, 2}; break;
        default: layout = {L7, 2}; break;
    }
    const usize stride = layout.widths.size() * layout.per;
    std::vector<u8> records(blocks * stride);
    usize next = 0;
    auto fetch = [&src, &next](usize) -> std::optional<u8> {
        if (next < src.size()) return src[next++];
        return std::nullopt;
    };
    if (const char* e = lanes(t, layout.widths, layout.per, blocks, fetch, records.data())) return e;
    const u8* rec = records.data();
    u8* out = dst.data();
    switch (bc) {
        case 1: return plain(rec, stride, out, block, blocks, t.bc1, BC1, t, bc1_residual, bc1_pack);
        case 3: return plain(rec, stride, out, block, blocks, t.bc3, BC3, t, bc3_residual, bc3_pack);
        case 4: return plain(rec, stride, out, block, blocks, t.bc4, BC4, t, bc4_residual, bc4_pack);
        case 5: return plain(rec, stride, out, block, blocks, t.bc5, BC5, t, bc5_residual, bc5_pack);
        case 6: return bc6(rec, stride, out, blocks, t.bc6);
        default: return bc7(rec, stride, out, blocks, t);
    }
}

}  // namespace detail

void TablesDeleter::operator()(Tables* t) const noexcept { delete t; }

void free_tables(Tables* t) noexcept { delete t; }

TablesPtr parse_tables(std::span<const std::uint8_t> bytes, const char** error) {
    try {
        TablesPtr t(new Tables());
        if (const char* e = detail::parse(*t, bytes)) {
            if (error) *error = e;
            return nullptr;
        }
        return t;
    } catch (const std::bad_alloc&) {
        if (error) *error = "out of memory";
        return nullptr;
    }
}

bool decode(unsigned bc, std::span<const std::uint8_t> src, std::span<std::uint8_t> dst, const Tables& tables,
            const char** error) {
    try {
        if (const char* e = detail::decode(bc, src, dst, tables)) {
            if (error) *error = e;
            return false;
        }
        return true;
    } catch (const std::bad_alloc&) {
        if (error) *error = "out of memory";
        return false;
    } catch (const std::length_error&) {
        if (error) *error = "destination too large";
        return false;
    }
}

}  // namespace leonida::bcpack_victory
