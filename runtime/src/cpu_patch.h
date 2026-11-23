// Ahead-of-time replacement of the Zen 2 instructions a host CPU lacks (docs/cpu_features.md), so the game's code
// runs natively instead of trapping into cpu_emulation.h on every execution.
//
// The executable's sites are found once, at build time (tools/cpu_feature_scan.py --sites; the RVAs stay on the
// player's machine, in cpu_sites.txt beside gta6rt.exe). Before any of the image's code runs, the loader rewrites
// each site whose feature this CPU lacks:
//   MONITORX                 a hint: multi-byte NOP of the same length
//   MWAITX                   PAUSE, then NOPs (an immediate return is one of the exits the APM allows)
//   CLWB m8                  a write-back hint on coherent memory: NOPs (only on CPUs without CLWB)
//   INSERTQ xmm, xmm, i, i   JMP rel32 to a stub that does the same with SSE2 (and NOPs to the old length)
//   EXTRQ   xmm, i, i        the same
// The stubs touch no flags (LEA, not ADD/SUB, moves the stack) and restore any XMM register they borrow, so the code
// around a site sees exactly what the instruction would have left. A site whose decoding does not match one of these
// forms, or is too short for a JMP rel32 (the register forms of INSERTQ/EXTRQ, CLZERO), is left alone: the #UD
// handler in cpu_emulation.h still covers it. cpu_patch_test runs every patched form against that handler.
#pragma once

#include "cpu_emulation.h"

#include <windows.h>
#include <intrin.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace rt::cpu_patch {

struct HostFeatures {
    bool sse4a = false, clzero = false, monitorx = false, clwb = false;
};

inline HostFeatures host_features() {
    HostFeatures f;
    int r[4] = {};
    __cpuid(r, 0x80000000);
    const unsigned max_ext = unsigned(r[0]);
    if (max_ext >= 0x80000001) {
        __cpuid(r, 0x80000001);
        f.sse4a = (r[2] >> 6) & 1;
        f.monitorx = (r[2] >> 29) & 1;
    }
    if (max_ext >= 0x80000008) {
        __cpuid(r, 0x80000008);
        f.clzero = r[1] & 1;
    }
    __cpuid(r, 0);
    if (r[0] >= 7) {
        __cpuidex(r, 7, 0);
        f.clwb = (r[1] >> 24) & 1;
    }
    return f;
}

// cpu_sites.txt: `image <SizeOfImage hex> <TimeDateStamp hex>` (the executable the sites belong to), then one
// hexadecimal RVA per line; # starts a comment. Missing file or no image line: no sites.
struct SitesFile {
    std::uint32_t size_of_image = 0, time_date_stamp = 0;
    std::vector<std::uint32_t> sites;
};

inline SitesFile read_sites(const std::filesystem::path& path) {
    SitesFile out;
    std::ifstream in(path);
    std::string line;
    bool image = false;
    while (std::getline(in, line)) {
        if (const auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        line = line.substr(first);
        try {
            if (line.rfind("image ", 0) == 0) {
                std::size_t at = 0;
                const std::string rest = line.substr(6);
                out.size_of_image = static_cast<std::uint32_t>(std::stoul(rest, &at, 16));
                out.time_date_stamp = static_cast<std::uint32_t>(std::stoul(rest.substr(at), nullptr, 16));
                image = true;
            } else {
                out.sites.push_back(static_cast<std::uint32_t>(std::stoul(line, nullptr, 16)));
            }
        } catch (...) {
            return {};
        }
    }
    if (!image) return {};
    return out;
}

// True when `base` is the image the sites file was made for.
inline bool same_image(const SitesFile& f, const std::uint8_t* base) {
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    return f.size_of_image && nt->OptionalHeader.SizeOfImage == f.size_of_image &&
           nt->FileHeader.TimeDateStamp == f.time_date_stamp;
}

// Code space for the stubs, within a JMP rel32 of the image.
class StubArena {
public:
    StubArena(std::uint8_t* image, std::size_t image_size) : image_(image), image_size_(image_size) {}
    StubArena(const StubArena&) = delete;
    StubArena& operator=(const StubArena&) = delete;
    ~StubArena() { seal(); }  // the stubs stay for the life of the image

    // `bytes` of writable code, 16-byte aligned, reachable from every byte of the image; nullptr when none is.
    std::uint8_t* take(std::size_t bytes) {
        bytes = (bytes + 15) & ~std::size_t(15);
        if (!base_ && !reserve()) return nullptr;
        if (used_ + bytes > size_) return nullptr;
        if (sealed_) {
            DWORD old = 0;
            if (!VirtualProtect(base_, size_, PAGE_READWRITE, &old)) return nullptr;
            sealed_ = false;
        }
        std::uint8_t* p = base_ + used_;
        used_ += bytes;
        return p;
    }

    // Executable and no longer writable: patch_image calls it once its stubs are written.
    void seal() {
        if (!base_ || sealed_) return;
        DWORD old = 0;
        VirtualProtect(base_, size_, PAGE_EXECUTE_READ, &old);
        FlushInstructionCache(GetCurrentProcess(), base_, size_);
        sealed_ = true;
    }

private:
    bool reserve() {
        // The first free 64 KiB-aligned region after the image (then before it) that keeps every distance in the
        // image within +/-2 GiB of the whole arena.
        const std::uintptr_t lo = reinterpret_cast<std::uintptr_t>(image_);
        const std::uintptr_t hi = lo + image_size_;
        const std::uintptr_t reach = 0x7FF00000;  // a little under 2 GiB
        for (std::uintptr_t at = (hi + 0xFFFF) & ~std::uintptr_t(0xFFFF); at + size_ - lo < reach; at += 0x10000) {
            if (void* p = VirtualAlloc(reinterpret_cast<void*>(at), size_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) {
                base_ = static_cast<std::uint8_t*>(p);
                return true;
            }
        }
        for (std::uintptr_t at = (lo - size_) & ~std::uintptr_t(0xFFFF); at > 0x10000 && hi - at < reach;
             at -= 0x10000) {
            if (void* p = VirtualAlloc(reinterpret_cast<void*>(at), size_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) {
                base_ = static_cast<std::uint8_t*>(p);
                return true;
            }
        }
        return false;
    }

    std::uint8_t* image_;
    std::size_t image_size_;
    std::uint8_t* base_ = nullptr;
    std::size_t size_ = 0x10000;  // 64 KiB: ~60 bytes a stub
    std::size_t used_ = 0;
    bool sealed_ = false;
};

namespace detail {

class Emitter {
public:
    explicit Emitter(std::uint8_t* at) : p_(at), start_(at) {}
    void b(std::uint8_t v) { *p_++ = v; }
    void d32(std::uint32_t v) {
        std::memcpy(p_, &v, 4);
        p_ += 4;
    }
    // [REX] op... with ModRM reg/rm both XMM registers.
    void xmm_rr(std::uint8_t prefix, std::initializer_list<std::uint8_t> op, int reg, int rm) {
        if (prefix) b(prefix);
        if (reg >= 8 || rm >= 8) b(std::uint8_t(0x40 | ((reg >> 3) << 2) | (rm >> 3)));
        for (std::uint8_t o : op) b(o);
        b(std::uint8_t(0xC0 | ((reg & 7) << 3) | (rm & 7)));
    }
    // [REX] op... xmm, [rip + disp32] pointing at `target`.
    void xmm_rip(std::uint8_t prefix, std::initializer_list<std::uint8_t> op, int reg, const std::uint8_t* target) {
        if (prefix) b(prefix);
        if (reg >= 8) b(0x44);
        for (std::uint8_t o : op) b(o);
        b(std::uint8_t(0x05 | ((reg & 7) << 3)));
        const std::intptr_t disp = target - (p_ + 4);
        d32(static_cast<std::uint32_t>(static_cast<std::int32_t>(disp)));
    }
    // [REX] op... xmm, [rsp]
    void xmm_rsp(std::uint8_t prefix, std::initializer_list<std::uint8_t> op, int reg) {
        if (prefix) b(prefix);
        if (reg >= 8) b(0x44);
        for (std::uint8_t o : op) b(o);
        b(std::uint8_t(0x04 | ((reg & 7) << 3)));
        b(0x24);
    }
    // PSRLQ/PSLLQ xmm, imm8: 66 [REX.B] 0F 73 /2 or /6 ib
    void shift_q(int ext, int reg, std::uint8_t count) {
        b(0x66);
        if (reg >= 8) b(0x41);
        b(0x0F);
        b(0x73);
        b(std::uint8_t(0xC0 | (ext << 3) | (reg & 7)));
        b(count);
    }
    void lea_rsp(std::int8_t delta) {  // lea rsp, [rsp + delta]: no flags
        b(0x48);
        b(0x8D);
        b(0x64);
        b(0x24);
        b(static_cast<std::uint8_t>(delta));
    }
    void jmp_to(const std::uint8_t* target) {
        b(0xE9);
        const std::intptr_t rel = target - (p_ + 4);
        d32(static_cast<std::uint32_t>(static_cast<std::int32_t>(rel)));
    }
    std::uint8_t* at() const { return p_; }
    std::size_t size() const { return std::size_t(p_ - start_); }

private:
    std::uint8_t* p_;
    std::uint8_t* start_;
};

inline std::uint64_t field_mask(std::uint8_t len) { return len == 0 || len >= 64 ? ~0ull : (1ull << len) - 1; }

// Multi-byte NOPs (Intel SDM vol. 2B, NOP): fill `n` bytes at `p`.
inline void nops(std::uint8_t* p, std::size_t n) {
    static constexpr std::uint8_t forms[9][9] = {
        {},
        {0x90},
        {0x66, 0x90},
        {0x0F, 0x1F, 0x00},
        {0x0F, 0x1F, 0x40, 0x00},
        {0x0F, 0x1F, 0x44, 0x00, 0x00},
        {0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00},
        {0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00},
        {0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    };
    while (n) {
        const std::size_t k = n > 8 ? 8 : n;
        std::memcpy(p, forms[k], k);
        p += k;
        n -= k;
    }
}

}  // namespace detail

struct Result {
    std::size_t nop = 0, stub = 0, left = 0, unrecognised = 0;
};

// Rewrites the sites of an image whose code is still writable (before the loader protects its sections). `avail(rva)`
// is how many bytes are readable from rva in the image.
inline Result patch_image(std::uint8_t* base, std::size_t image_size, const std::vector<std::uint32_t>& sites,
                          const HostFeatures& host, StubArena& arena) {
    using rt::cpu_emulation::Op;
    Result r;
    for (const std::uint32_t rva : sites) {
        if (rva >= image_size) {
            ++r.unrecognised;
            continue;
        }
        std::uint8_t* site = base + rva;
        const auto d = rt::cpu_emulation::decode(site, image_size - rva);
        switch (d.op) {
            case Op::monitorx:
                if (host.monitorx) continue;
                detail::nops(site, d.length);
                ++r.nop;
                continue;
            case Op::mwaitx:
                if (host.monitorx) continue;
                if (d.length < 2) {
                    ++r.left;
                    continue;
                }
                site[0] = 0xF3;
                site[1] = 0x90;  // PAUSE
                detail::nops(site + 2, d.length - 2);
                ++r.nop;
                continue;
            case Op::clwb:
                if (host.clwb) continue;
                detail::nops(site, d.length);
                ++r.nop;
                continue;
            case Op::insertq_imm:
            case Op::extrq_imm: {
                if (host.sse4a) continue;
                if (d.length < 5) {
                    ++r.left;
                    continue;
                }
                std::uint8_t* stub = arena.take(96);
                if (!stub) {
                    ++r.left;
                    continue;
                }
                // Constants first (16-byte aligned for the legacy-SSE memory operands), code after them.
                const std::uint8_t len = d.len & 63, idx = d.idx & 63;
                const std::uint64_t field = detail::field_mask(len);
                std::uint64_t consts[4] = {};
                std::uint8_t* code = stub + 32;
                detail::Emitter e(code);
                const std::uint8_t* resume = site + d.length;
                if (d.op == Op::extrq_imm) {
                    // dst = (dst >> idx) & field, high quadword cleared.
                    consts[0] = field;
                    std::memcpy(stub, consts, 32);
                    if (idx) e.shift_q(2, d.dst, idx);                          // psrlq dst, idx
                    e.xmm_rip(0x66, {0x0F, 0xDB}, d.dst, stub);                 // pand dst, [field, 0]
                } else {
                    // dst = (dst & ~(field << idx)) | ((src & field) << idx), high quadword cleared.
                    const std::uint64_t placed = idx >= 64 ? 0 : field << idx;
                    consts[0] = field;
                    consts[2] = ~placed;
                    std::memcpy(stub, consts, 32);
                    int t = 0;
                    while (t == d.dst || t == d.src) ++t;
                    e.lea_rsp(-16);
                    e.xmm_rsp(0xF3, {0x0F, 0x7F}, t);                          // movdqu [rsp], xmmT
                    e.xmm_rr(0, {0x0F, 0x28}, t, d.src);                         // movaps xmmT, src
                    e.xmm_rip(0x66, {0x0F, 0xDB}, t, stub);                      // pand xmmT, [field, 0]
                    if (idx) e.shift_q(6, t, idx);                               // psllq xmmT, idx
                    e.xmm_rip(0x66, {0x0F, 0xDB}, d.dst, stub + 16);             // pand dst, [~placed, 0]
                    e.xmm_rr(0x66, {0x0F, 0xEB}, d.dst, t);                      // por dst, xmmT
                    e.xmm_rsp(0xF3, {0x0F, 0x6F}, t);                          // movdqu xmmT, [rsp]
                    e.lea_rsp(16);
                }
                e.jmp_to(resume);
                // The site: JMP rel32 to the stub's code, NOPs to the instruction's end.
                detail::Emitter j(site);
                j.jmp_to(code);
                detail::nops(site + 5, d.length - 5);
                ++r.stub;
                continue;
            }
            case Op::none:
                ++r.unrecognised;
                continue;
            default:
                ++r.left;  // the register forms and CLZERO: the #UD handler keeps them
                continue;
        }
    }
    arena.seal();
    FlushInstructionCache(GetCurrentProcess(), base, image_size);
    return r;
}

}  // namespace rt::cpu_patch
