// Xbox device slot 57's native graphics pipelines (d3d12_xs_native_pso.h, NativeGraphicsPipelines) with no D3D12
// runtime, no GPU and no game data: the descriptions, the digest map and the DXBC containers are built here by hand.
// merge: ours tested its own slot-57 path inside the device proxy (create_native_graphics_pipeline, its stats and
// directory provider); the merge kept upstream's, where the Vulkan device builds the pipeline from a catalog it loads
// once, so this test now drives that class, and one call through the device proxy's factory route.
#include "../shims/d3d12_xs_device.h"
#include "../shims/d3d12_xs_native_pso.h"
#include "../src/native_pso.h"

#include <d3d12.h>
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace np = leonida::native_pso;
namespace xs = rt::d3d12_xs;

int g_failures = 0;
#define EXPECT(c) \
    do { if (!(c)) { std::fprintf(stderr, "xs_native_pso_test line %d: %s\n", __LINE__, #c); ++g_failures; } } while (0)

void put32(std::vector<std::uint8_t>& b, std::size_t at, std::uint32_t v) {
    if (b.size() < at + 4) b.resize(at + 4);
    std::memcpy(b.data() + at, &v, 4);
}

// Blob B: two render targets, D32 depth, triangles, one sample, and a two-element input layout.
std::vector<std::uint8_t> state_b() {
    std::vector<std::uint8_t> b(np::kStateBHeader, 0);
    b[0x10] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    b[0x11] = DXGI_FORMAT_R16G16_FLOAT;
    put32(b, 0x1c, 0xffffffffu);
    b[0x24] = DXGI_FORMAT_D32_FLOAT;
    b[0x25] = 2;
    b[0x26] = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    b[0x28] = 1;
    put32(b, 0x4c, 2);
    const struct { const char* name; std::uint32_t format, offset; } elements[] = {
        {"POSITION", DXGI_FORMAT_R32G32B32_FLOAT, 0}, {"TEXCOORD", DXGI_FORMAT_R32G32_FLOAT, 12}};
    for (const auto& e : elements) {
        const std::uint32_t length = static_cast<std::uint32_t>(std::strlen(e.name) + 1);
        put32(b, b.size(), length);
        b.insert(b.end(), e.name, e.name + length);
        for (const std::uint32_t v : {0u, e.format, 0u, e.offset, 0u, 0u}) put32(b, b.size(), v);
    }
    return b;
}

np::Digest digest(std::uint8_t seed) {
    np::Digest d;
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = static_cast<std::uint8_t>(seed * 31 + i);
    return d;
}

// A stage's records: the digest the catalog finds its container by.
std::vector<std::uint8_t> records(const np::Digest& d) {
    std::vector<std::uint8_t> b;
    for (const std::uint32_t v : {0x403u, 0x41u, 0u, 0x43u}) put32(b, b.size(), v);
    b.insert(b.end(), d.begin(), d.end());
    return b;
}

// A DXBC container with one HASH part (flags, then the digest): what NativeGraphicsPipelines checks a file against.
std::vector<std::uint8_t> container(const np::Digest& d) {
    std::vector<std::uint8_t> b(32, 0);
    std::memcpy(b.data(), "DXBC", 4);
    put32(b, 20, 1);   // version
    put32(b, 28, 1);   // one part
    put32(b, 32, 36);  // its offset
    b.insert(b.end(), {'H', 'A', 'S', 'H'});
    put32(b, b.size(), 20);
    put32(b, b.size(), 0);
    b.insert(b.end(), d.begin(), d.end());
    put32(b, 24, static_cast<std::uint32_t>(b.size()));
    return b;
}

const np::Digest kVs = digest(1), kPs = digest(2), kMs = digest(3), kLost = digest(4), kUnknown = digest(5);
const std::vector<std::uint8_t> kVsBytes = container(kVs), kPsBytes = container(kPs), kMsBytes = container(kMs);
std::filesystem::path g_root, g_spirv;
int g_root_signature = 0;  // only its address is passed through

void write(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()),
                                                static_cast<std::streamsize>(bytes.size()));
}

void make_tree() {
    wchar_t temp[MAX_PATH + 1] = {};
    GetTempPathW(MAX_PATH, temp);
    g_root = std::filesystem::path(temp) / (L"leonida_native_pso_" + std::to_wstring(GetCurrentProcessId()));
    std::filesystem::remove_all(g_root);
    g_spirv = g_root / L"spirv";
    std::filesystem::create_directories(g_spirv);
    std::filesystem::create_directories(g_root / L"dxbc");
    write(g_root / L"dxbc" / L"vs one.dxbc", kVsBytes);
    write(g_root / L"dxbc" / L"ps.dxbc", kPsBytes);
    write(g_root / L"dxbc" / L"ms.dxbc", kMsBytes);
    const auto line = [](const np::Digest& d, const char* stage, const char* path) {
        return np::hex(d) + "\t" + stage + "\t" + path + "\n";
    };
    const std::string map = std::string(np::HashMap::kHeader) + "\n" + line(kVs, "vs", "../dxbc/vs one.dxbc") +
                            line(kPs, "ps", "../dxbc/ps.dxbc") + line(kMs, "ms", "../dxbc/ms.dxbc") +
                            line(kLost, "ps", "../dxbc/gone.dxbc");
    write(g_spirv / L"native_hashes.txt", std::vector<std::uint8_t>(map.begin(), map.end()));
}

struct Desc {
    std::vector<std::uint8_t> a = std::vector<std::uint8_t>(384, 0), b = state_b();
    std::vector<std::uint8_t> code[np::kStageBlocks], rec[np::kStageBlocks];
    np::GraphicsDesc desc{};
    Desc() {
        desc.root_signature = &g_root_signature;
        desc.state_a = {a.data(), a.size()};
        desc.state_b = {b.data(), b.size()};
    }
    Desc& stage(std::size_t k, const np::Digest& d) {
        code[k].assign(64, 0xbf);
        rec[k] = records(d);
        desc.stages[k] = {{code[k].data(), code[k].size()}, {rec[k].data(), rec[k].size()}};
        return *this;
    }
};

// What the pipeline callback was handed.
struct Built {
    int calls = 0;
    std::vector<std::uint8_t> vs, ps, gs, ms;
    const void* root = nullptr;
    UINT rt_count = 0;
    DXGI_FORMAT rtv0 = DXGI_FORMAT_UNKNOWN, rtv1 = DXGI_FORMAT_UNKNOWN, dsv = DXGI_FORMAT_UNKNOWN;
    std::vector<std::string> names;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
};

std::vector<std::uint8_t> bytes_of(const D3D12_SHADER_BYTECODE& code) {
    const auto* p = static_cast<const std::uint8_t*>(code.pShaderBytecode);
    return p ? std::vector<std::uint8_t>(p, p + code.BytecodeLength) : std::vector<std::uint8_t>{};
}

void* const kPipeline = reinterpret_cast<void*>(0x1000);

HRESULT build(xs::NativeGraphicsPipelines& pipelines, const std::filesystem::path& folder, const Desc& d, Built& b,
              void** out) {
    return pipelines.create(folder, d.desc, out, [&](const D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc,
                                                       const D3D12_SHADER_BYTECODE& mesh, const auto&) {
        ++b.calls;
        b.vs = bytes_of(pc.VS);
        b.ps = bytes_of(pc.PS);
        b.gs = bytes_of(pc.GS);
        b.ms = bytes_of(mesh);
        b.root = pc.pRootSignature;
        b.rt_count = pc.NumRenderTargets;
        b.rtv0 = pc.RTVFormats[0];
        b.rtv1 = pc.RTVFormats[1];
        b.dsv = pc.DSVFormat;
        b.topology = pc.PrimitiveTopologyType;
        b.names.clear();
        for (UINT i = 0; i < pc.InputLayout.NumElements; ++i)
            b.names.emplace_back(pc.InputLayout.pInputElementDescs[i].SemanticName);
        *out = kPipeline;
        return S_OK;
    });
}

void test_vs_ps() {
    xs::NativeGraphicsPipelines pipelines;
    Desc d;
    d.stage(2, kPs).stage(0, kVs);
    Built b;
    void* out = nullptr;
    EXPECT(build(pipelines, g_spirv, d, b, &out) == S_OK && out == kPipeline && b.calls == 1);
    EXPECT(b.vs == kVsBytes && b.ps == kPsBytes && b.gs.empty() && b.ms.empty());
    EXPECT(b.root == static_cast<const void*>(&g_root_signature));
    EXPECT(b.rt_count == 2 && b.rtv0 == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB && b.rtv1 == DXGI_FORMAT_R16G16_FLOAT);
    EXPECT(b.dsv == DXGI_FORMAT_D32_FLOAT && b.topology == D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    EXPECT(b.names.size() == 2 && b.names[0] == "POSITION" && b.names[1] == "TEXCOORD");
}

// A container is read from disk once: a later pipeline naming it gets the same bytes even if the file is gone.
void test_container_read_once() {
    xs::NativeGraphicsPipelines pipelines;
    Desc d;
    d.stage(0, kVs).stage(1, kPs);
    Built b;
    void* out = nullptr;
    EXPECT(build(pipelines, g_spirv, d, b, &out) == S_OK && b.vs == kVsBytes);
    const std::filesystem::path vs = g_root / L"dxbc" / L"vs one.dxbc", moved = g_root / L"dxbc" / L"vs one.moved";
    std::filesystem::rename(vs, moved);
    out = nullptr;
    const HRESULT again = build(pipelines, g_spirv, d, b, &out);
    std::filesystem::rename(moved, vs);
    EXPECT(again == S_OK && b.calls == 2 && b.vs == kVsBytes && b.ps == kPsBytes);
}

void test_mesh() {
    xs::NativeGraphicsPipelines pipelines;
    Desc d;
    d.stage(1, kPs).stage(6, kMs);
    Built b;
    void* out = nullptr;
    EXPECT(build(pipelines, g_spirv, d, b, &out) == S_OK && b.calls == 1);
    EXPECT(b.ms == kMsBytes && b.ps == kPsBytes && b.vs.empty() && b.rt_count == 2);
}

// A digest the catalog does not name, or one whose container is gone: E_NOTIMPL and nothing built (slot 57 then goes
// on with the 0x68-byte description).
void test_unresolved() {
    xs::NativeGraphicsPipelines pipelines;
    Built b;
    void* out = nullptr;
    Desc unknown;
    unknown.stage(0, kVs).stage(1, kUnknown);
    EXPECT(build(pipelines, g_spirv, unknown, b, &out) == E_NOTIMPL && !out && b.calls == 0);
    Desc lost;
    lost.stage(0, kVs).stage(1, kLost);
    EXPECT(FAILED(build(pipelines, g_spirv, lost, b, &out)) && !out && b.calls == 0);
    // A container whose HASH part is not its digest is refused.
    write(g_root / L"dxbc" / L"ps.dxbc", kVsBytes);
    xs::NativeGraphicsPipelines fresh;
    Desc wrong;
    wrong.stage(0, kVs).stage(1, kPs);
    EXPECT(FAILED(build(fresh, g_spirv, wrong, b, &out)) && b.calls == 0);
    write(g_root / L"dxbc" / L"ps.dxbc", kPsBytes);
}

// A folder without native_hashes.txt: nothing resolves.
void test_no_catalog() {
    xs::NativeGraphicsPipelines pipelines;
    Desc d;
    d.stage(0, kVs).stage(1, kPs);
    Built b;
    void* out = nullptr;
    EXPECT(build(pipelines, g_root, d, b, &out) == E_NOTIMPL && b.calls == 0);
}

}  // namespace

int main() {
    make_tree();
    test_vs_ps();
    test_container_read_once();
    test_mesh();
    test_unresolved();
    test_no_catalog();
    std::error_code ignored;
    std::filesystem::remove_all(g_root, ignored);
    if (g_failures) {
        std::fprintf(stderr, "xs_native_pso_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("xs_native_pso_test: ok\n");
    return 0;
}
