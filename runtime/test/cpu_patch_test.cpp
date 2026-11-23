// cpu_patch.h: every INSERTQ/EXTRQ immediate form, patched to a JMP and an SSE2 stub, gives what the #UD emulation's
// reference (cpu_emulation.h insert_bits/extract_bits, which matched a Zen 3) gives, on low and high XMM registers and
// on every edge of len/idx, without touching the flags or the XMM register the stub borrows; MONITORX/MWAITX/CLWB
// become NOPs. The patched code is executed natively, so on a CPU without SSE4a an unpatched site would end the test
// with #UD. No game data.
#include "../src/cpu_patch.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const char* what, int a = 0, int b = 0, int c = 0) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s (%d %d %d)\n", what, a, b, c);
    ++failures;
}

struct Xmm {
    std::uint64_t lo, hi;
};

// movdqu xmm<r>, [base] / movdqu [base], xmm<r>, with base rcx(1), rdx(2) or r8(8).
void movdqu(std::vector<std::uint8_t>& c, bool store, int r, int base) {
    c.push_back(0xF3);
    const std::uint8_t rex = std::uint8_t(0x40 | ((r >> 3) << 2) | (base >> 3));
    if (rex != 0x40) c.push_back(rex);
    c.push_back(0x0F);
    c.push_back(store ? 0x7F : 0x6F);
    c.push_back(std::uint8_t(((r & 7) << 3) | (base & 7)));
}

using Fn = int (*)(Xmm* dst, const Xmm* src, Xmm* spare);

struct Built {
    std::vector<std::uint8_t> code;
    std::size_t site = 0;
};

// dst, src, spare loaded; STC; the instruction; SETC AL; dst and spare stored; RET.
Built build(bool insert, int dst, int src, int spare, std::uint8_t len, std::uint8_t idx) {
    Built b;
    auto& c = b.code;
    c.push_back(0x31);
    c.push_back(0xC0);  // xor eax, eax
    movdqu(c, false, dst, 1);
    if (insert) movdqu(c, false, src, 2);
    movdqu(c, false, spare, 8);
    c.push_back(0xF9);  // stc
    b.site = c.size();
    if (insert) {
        c.push_back(0xF2);
        const std::uint8_t rex = std::uint8_t(0x40 | ((dst >> 3) << 2) | (src >> 3));
        if (rex != 0x40) c.push_back(rex);
        c.push_back(0x0F);
        c.push_back(0x78);
        c.push_back(std::uint8_t(0xC0 | ((dst & 7) << 3) | (src & 7)));
    } else {
        c.push_back(0x66);
        if (dst >= 8) c.push_back(0x41);
        c.push_back(0x0F);
        c.push_back(0x78);
        c.push_back(std::uint8_t(0xC0 | (dst & 7)));  // /0
    }
    c.push_back(len);
    c.push_back(idx);
    c.push_back(0x0F);
    c.push_back(0x92);
    c.push_back(0xC0);  // setc al
    movdqu(c, true, dst, 1);
    movdqu(c, true, spare, 8);
    c.push_back(0xC3);
    return b;
}

// The register the stub borrows for INSERTQ (cpu_patch.h: the lowest one that is neither operand).
int borrowed(int dst, int src) {
    int t = 0;
    while (t == dst || t == src) ++t;
    return t;
}

}  // namespace

int main() {
    using namespace rt::cpu_patch;
    const HostFeatures none{};  // patch every site, whatever this CPU has
    std::uint8_t* page = static_cast<std::uint8_t*>(
        VirtualAlloc(nullptr, 0x20000, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    if (!page) return 1;
    StubArena arena(page, 0x20000);

    const std::uint8_t edges[] = {0, 1, 7, 31, 32, 33, 62, 63};
    const int regs[][2] = {{0, 1}, {1, 0}, {3, 9}, {9, 3}, {8, 15}, {15, 8}, {2, 0}};
    const Xmm a{0x0123456789ABCDEFull, 0xFFFFFFFFFFFFFFFFull}, s{0xF0E1D2C3B4A59687ull, 0x1111111111111111ull};
    const Xmm spare_in{0x5555AAAA5555AAAAull, 0x1234123412341234ull};
    std::size_t offset = 0, cases = 0;
    for (const bool insert : {true, false}) {
        for (const auto& rs : regs) {
            StubArena group_arena(page, 0x20000);  // 64 cases a group; one arena holds ~680 stubs
            const int dst = rs[0], src = rs[1];
            const int spare = insert ? borrowed(dst, src) : (dst == 4 ? 5 : 4);
            for (const std::uint8_t len : edges) {
                for (const std::uint8_t idx : edges) {
                    const Built b = build(insert, dst, src, spare, len, idx);
                    if (offset + b.code.size() > 0x18000) return 1;  // the test grew past its code space
                    std::uint8_t* fn = page + offset;
                    std::memcpy(fn, b.code.data(), b.code.size());
                    offset += (b.code.size() + 15) & ~std::size_t(15);
                    const std::vector<std::uint32_t> sites{std::uint32_t(fn + b.site - page)};
                    const Result r = patch_image(page, 0x18000, sites, none, group_arena);
                    expect(r.stub == 1, "the site was replaced by a stub", insert, len, idx);
                    if (r.stub != 1) return 1;  // an unpatched site would #UD on a CPU without SSE4a
                    Xmm d = a, sp = spare_in;
                    const int carry = reinterpret_cast<Fn>(fn)(&d, &s, &sp);
                    const std::uint64_t want = insert ? rt::cpu_emulation::insert_bits(a.lo, s.lo, len, idx)
                                                      : rt::cpu_emulation::extract_bits(a.lo, len, idx);
                    expect(d.lo == want, insert ? "INSERTQ result" : "EXTRQ result", dst, len, idx);
                    expect(d.hi == 0, "the high quadword is cleared, as on Zen 3", insert, len, idx);
                    expect(carry == 1, "the carry flag survives the stub", insert, len, idx);
                    expect(sp.lo == spare_in.lo && sp.hi == spare_in.hi, "the borrowed register is restored", insert,
                           len, idx);
                    ++cases;
                }
            }
        }
    }

    // MONITORX, MWAITX, CLWB: NOPs (and PAUSE), so a function made of them runs and returns.
    {
        std::uint8_t* fn = page + 0x19000;
        const std::uint8_t code[] = {0x0F, 0x01, 0xFA,                    // monitorx
                                     0x0F, 0x01, 0xFB,                    // mwaitx
                                     0x66, 0x0F, 0xAE, 0x31,              // clwb [rcx]
                                     0xB8, 0x2A, 0x00, 0x00, 0x00,        // mov eax, 42
                                     0xC3};
        std::memcpy(fn, code, sizeof code);
        const std::uint32_t base = std::uint32_t(fn - page);
        const Result r = patch_image(page, 0x1A000, {base, base + 3, base + 6}, none, arena);
        expect(r.nop == 3, "MONITORX, MWAITX and CLWB became NOPs");
        expect(fn[3] == 0xF3 && fn[4] == 0x90, "MWAITX became PAUSE");
        Xmm dummy{};
        expect(reinterpret_cast<int (*)(Xmm*)>(fn)(&dummy) == 42, "the NOP'd function runs");
    }

    // A CLZERO zeroing loop (`clzero; add rax, 40h; cmp rax, R; jb back`, R = rdx or r10) becomes a stub that zeroes
    // the same 64-byte lines (do-while: at least one), leaves rax where the loop would, and keeps r11.
    for (const bool r10_limit : {false, true}) {
        // rcx = start, rdx = end, r8 -> r11's value in and out; returns rax.
        std::vector<std::uint8_t> c = {0x4D, 0x8B, 0x18,                  // mov r11, [r8]
                                       0x48, 0x89, 0xC8};                 // mov rax, rcx
        if (r10_limit) c.insert(c.end(), {0x49, 0x89, 0xD2});            // mov r10, rdx
        const std::size_t site = c.size();
        c.insert(c.end(), {0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40});    // clzero; add rax, 40h
        if (r10_limit) c.insert(c.end(), {0x4C, 0x39, 0xD0});            // cmp rax, r10
        else c.insert(c.end(), {0x48, 0x39, 0xD0});                       // cmp rax, rdx
        c.insert(c.end(), {0x72, 0xF4,                                    // jb the clzero
                           0x4D, 0x89, 0x18,                              // mov [r8], r11
                           0xC3});
        std::uint8_t* fn = page + 0x19400 + (r10_limit ? 0x100 : 0);
        std::memcpy(fn, c.data(), c.size());
        const Result r = patch_image(page, 0x1A000, {std::uint32_t(fn + site - page)}, none, arena);
        expect(r.stub == 1, "the CLZERO loop was replaced by a stub", r10_limit);
        if (r.stub != 1) return 1;
        alignas(64) static std::uint8_t buffer[1024];
        using Loop = std::uint64_t (*)(std::uint8_t*, std::uint8_t*, std::uint64_t*);
        struct Case { std::size_t start, end; };
        for (const Case k : {Case{64, 448}, Case{80, 300}, Case{128, 128}, Case{0, 1000}}) {
            std::memset(buffer, 0xAB, sizeof buffer);
            std::uint64_t r11 = 0x1122334455667788ull;
            const std::uint64_t rax = reinterpret_cast<Loop>(fn)(buffer + k.start, buffer + k.end, &r11);
            // The lines the loop zeroes: those holding start, start + 64, ... while below end (at least the first).
            std::size_t lines = 1;
            while (k.start + 64 * lines < k.end) ++lines;
            const std::size_t first = k.start & ~std::size_t(63), last = ((k.start + 64 * (lines - 1)) & ~std::size_t(63)) + 64;
            bool ok = true;
            for (std::size_t i = 0; i < sizeof buffer; ++i)
                ok = ok && buffer[i] == ((i >= first && i < last) ? 0 : 0xAB);
            expect(ok, "CLZERO loop: exactly the loop's lines are zeroed", r10_limit, int(k.start), int(k.end));
            expect(rax == reinterpret_cast<std::uint64_t>(buffer + k.start + 64 * lines), "CLZERO loop: rax ends where "
                   "the loop's does", r10_limit, int(k.start));
            expect(r11 == 0x1122334455667788ull, "CLZERO loop: r11 is kept", r10_limit);
        }
    }

    // A site that is not one of the forms is counted and left alone.
    {
        std::uint8_t* fn = page + 0x19800;
        fn[0] = 0x90;
        fn[1] = 0xC3;
        const Result r = patch_image(page, 0x1A000, {std::uint32_t(fn - page)}, none, arena);
        expect(r.unrecognised == 1 && fn[0] == 0x90, "an unrecognised site is left alone");
    }

    VirtualFree(page, 0, MEM_RELEASE);
    if (failures) return 1;
    std::printf("cpu_patch: %zu INSERTQ/EXTRQ cases patched and run natively match the reference; NOP forms run\n",
                cases);
    return 0;
}
