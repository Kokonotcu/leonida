// D3D12 ResourceBarrier and ID3D12GraphicsCommandList7::Barrier null-pointer safety,
// argument validation parity with native Windows D3D12 (hardware + WARP), and Vulkan replay.
// 100% synthetic resources; no game content.
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

namespace {

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
    void reset() {
        if (p) {
            p->Release();
            p = nullptr;
        }
    }
};

struct DeviceEnv {
    Ref<ID3D12Device> device;
    Ref<ID3D12Resource> buffer;
    Ref<ID3D12Resource> texture;
    bool enhanced_supported = false;

    bool init_resources(ID3D12Device* dev) {
        buffer.reset();
        texture.reset();
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 opts12{};
        enhanced_supported = SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &opts12, sizeof(opts12))) &&
                             opts12.EnhancedBarriersSupported != FALSE;

        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC bdesc{};
        bdesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bdesc.Width = 4096;
        bdesc.Height = 1;
        bdesc.DepthOrArraySize = 1;
        bdesc.MipLevels = 1;
        bdesc.SampleDesc.Count = 1;
        bdesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bdesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bdesc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&buffer.p))))
            return false;

        D3D12_RESOURCE_DESC tdesc{};
        tdesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tdesc.Width = 16;
        tdesc.Height = 16;
        tdesc.DepthOrArraySize = 1;
        tdesc.MipLevels = 1;
        tdesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        tdesc.SampleDesc.Count = 1;
        tdesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        tdesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &tdesc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&texture.p))))
            return false;
        return true;
    }
};

struct HostEnv : DeviceEnv {
    bool use_warp = false;

    bool init(bool warp) {
        use_warp = warp;
        device.reset();
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) return false;
        Ref<IDXGIAdapter> adapter;
        if (warp) {
            if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) return false;
        } else {
            if (FAILED(factory->EnumAdapters(0, &adapter.p))) return false;
        }
        if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device.p)))) return false;
        return init_resources(device.p);
    }

    bool ensure_alive() {
        if (!device.p || device->GetDeviceRemovedReason() != S_OK) return init(use_warp);
        return true;
    }
};

struct ParityCase {
    const char* name;
    bool expected_ok;
    bool enhanced;
    std::function<void(ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource*)> fn;
};

HRESULT run_close_case(DeviceEnv& env, const ParityCase& c) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(env.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(env.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))))
        return E_UNEXPECTED;
    Ref<ID3D12GraphicsCommandList7> list7;
    list->QueryInterface(IID_PPV_ARGS(&list7.p));
    c.fn(list.p, list7.p, env.buffer.p, env.texture.p);
    return list->Close();
}

int run_null_pointer_regression(DeviceEnv& layer) {
    struct NullCase {
        const char* name;
        HRESULT expected_hr;
        bool close_first;
        std::function<void(ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7*)> fn;
    };

    const std::vector<NullCase> cases = {
        {"ResourceBarrier(1, nullptr)", E_INVALIDARG, false,
         [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*) { l->ResourceBarrier(1, nullptr); }},
        {"ResourceBarrier(0, nullptr) on open list", S_OK, false,
         [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*) { l->ResourceBarrier(0, nullptr); }},
        {"ResourceBarrier(0, nullptr) on closed list", E_FAIL, true,
         [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*) { l->ResourceBarrier(0, nullptr); }},
        {"Barrier(1, nullptr)", E_INVALIDARG, false,
         [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7) { l7->Barrier(1, nullptr); }},
        {"Barrier(0, nullptr) on open list", S_OK, false,
         [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7) { l7->Barrier(0, nullptr); }},
        {"Barrier(0, nullptr) on closed list", E_FAIL, true,
         [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7) { l7->Barrier(0, nullptr); }},
        {"Barrier GLOBAL NumBarriers=1, pGlobalBarriers=nullptr", E_INVALIDARG, false,
         [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7) {
             D3D12_BARRIER_GROUP g{};
             g.Type = D3D12_BARRIER_TYPE_GLOBAL;
             g.NumBarriers = 1;
             g.pGlobalBarriers = nullptr;
             l7->Barrier(1, &g);
         }},
        {"Barrier BUFFER NumBarriers=1, pBufferBarriers=nullptr", E_INVALIDARG, false,
         [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7) {
             D3D12_BARRIER_GROUP g{};
             g.Type = D3D12_BARRIER_TYPE_BUFFER;
             g.NumBarriers = 1;
             g.pBufferBarriers = nullptr;
             l7->Barrier(1, &g);
         }},
        {"Barrier TEXTURE NumBarriers=1, pTextureBarriers=nullptr", E_INVALIDARG, false,
         [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7) {
             D3D12_BARRIER_GROUP g{};
             g.Type = D3D12_BARRIER_TYPE_TEXTURE;
             g.NumBarriers = 1;
             g.pTextureBarriers = nullptr;
             l7->Barrier(1, &g);
         }},
        {"Barrier GLOBAL/BUFFER/TEXTURE NumBarriers=0 with null pointers", S_OK, false,
         [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7) {
             D3D12_BARRIER_GROUP g[3]{};
             g[0].Type = D3D12_BARRIER_TYPE_GLOBAL;
             g[1].Type = D3D12_BARRIER_TYPE_BUFFER;
             g[2].Type = D3D12_BARRIER_TYPE_TEXTURE;
             l7->Barrier(3, g);
         }},
    };

    int failures = 0;
    for (const auto& c : cases) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        if (FAILED(layer.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
            FAILED(layer.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p)))) {
            std::fprintf(stderr, "FAIL %s: command list creation failed\n", c.name);
            return 1;
        }
        Ref<ID3D12GraphicsCommandList7> list7;
        if (FAILED(list->QueryInterface(IID_PPV_ARGS(&list7.p))) || !list7.p) {
            std::fprintf(stderr, "FAIL %s: ID3D12GraphicsCommandList7 unavailable\n", c.name);
            return 1;
        }
        if (c.close_first && FAILED(list->Close())) {
            std::fprintf(stderr, "FAIL %s: initial Close failed\n", c.name);
            return 1;
        }
        c.fn(list.p, list7.p);
        const HRESULT hr = list->Close();
        if (hr != c.expected_hr) {
            std::fprintf(stderr, "FAIL null-ptr %s: got 0x%08lx, expected 0x%08lx\n", c.name,
                         static_cast<unsigned long>(hr), static_cast<unsigned long>(c.expected_hr));
            ++failures;
        }
    }
    std::printf("layer null-pointer regression: %zu cases, %d failures\n", cases.size(), failures);
    return failures;
}

std::vector<ParityCase> build_parity_cases() {
    std::vector<ParityCase> cases;

    // Legacy ResourceBarrier
    cases.push_back({"RB(0, nullptr)", true, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource*) {
                         l->ResourceBarrier(0, nullptr);
                     }});
    cases.push_back({"RB TRANSITION pResource=nullptr", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                         b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                         b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                         b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB TRANSITION COMMON->COMMON", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                         b.Transition.pResource = tex;
                         b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                         b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                         b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB TRANSITION RT->RT", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                         b.Transition.pResource = tex;
                         b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                         b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
                         b.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB TRANSITION BEGIN_ONLY|END_ONLY", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                         b.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY | D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
                         b.Transition.pResource = tex;
                         b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                         b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                         b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB TRANSITION unknown flag 0x4", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                         b.Flags = static_cast<D3D12_RESOURCE_BARRIER_FLAGS>(0x4);
                         b.Transition.pResource = tex;
                         b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                         b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                         b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB TRANSITION unknown state bit 0x80000000", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                         b.Transition.pResource = tex;
                         b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                         b.Transition.StateBefore = static_cast<D3D12_RESOURCE_STATES>(0x80000000u);
                         b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB TRANSITION multi-write RT|COPY_DEST", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                         b.Transition.pResource = tex;
                         b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                         b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                         b.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_COPY_DEST;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB TRANSITION write+read COPY_DEST|COPY_SOURCE", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                         b.Transition.pResource = tex;
                         b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                         b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                         b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST | D3D12_RESOURCE_STATE_COPY_SOURCE;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB UAV pResource=nullptr", true, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB UAV Flags=BEGIN_ONLY", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource* buf, ID3D12Resource*) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                         b.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
                         b.UAV.pResource = buf;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB ALIASING null->null", true, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB ALIASING Flags=BEGIN_ONLY", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource* buf, ID3D12Resource*) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                         b.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
                         b.Aliasing.pResourceAfter = buf;
                         l->ResourceBarrier(1, &b);
                     }});
    cases.push_back({"RB unknown Type=99", false, false,
                     [](ID3D12GraphicsCommandList* l, ID3D12GraphicsCommandList7*, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_RESOURCE_BARRIER b{};
                         b.Type = static_cast<D3D12_RESOURCE_BARRIER_TYPE>(99);
                         l->ResourceBarrier(1, &b);
                     }});

    // Enhanced Barrier
    cases.push_back({"EB(0, nullptr)", true, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         l7->Barrier(0, nullptr);
                     }});
    cases.push_back({"EB group NumBarriers=0, ptr=nullptr", true, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_BARRIER_GROUP g{};
                         g.Type = D3D12_BARRIER_TYPE_GLOBAL;
                         g.NumBarriers = 0;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB unknown Type=99, NumBarriers=0", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_BARRIER_GROUP g{};
                         g.Type = static_cast<D3D12_BARRIER_TYPE>(99);
                         g.NumBarriers = 0;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB unknown Type=99, NumBarriers=1", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_GLOBAL_BARRIER gb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL,
                                                 D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_COMMON};
                         D3D12_BARRIER_GROUP g{};
                         g.Type = static_cast<D3D12_BARRIER_TYPE>(99);
                         g.NumBarriers = 1;
                         g.pGlobalBarriers = &gb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB GLOBAL SyncBefore=NONE AccessBefore=COMMON", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_GLOBAL_BARRIER gb{D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_SYNC_ALL,
                                                 D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_COMMON};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_GLOBAL, 1};
                         g.pGlobalBarriers = &gb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB GLOBAL SyncBefore=NONE AccessBefore=NO_ACCESS", true, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_GLOBAL_BARRIER gb{D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_SYNC_ALL,
                                                 D3D12_BARRIER_ACCESS_NO_ACCESS, D3D12_BARRIER_ACCESS_COMMON};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_GLOBAL, 1};
                         g.pGlobalBarriers = &gb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB GLOBAL SyncAfter=NONE AccessAfter=COMMON", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_GLOBAL_BARRIER gb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_NONE,
                                                 D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_COMMON};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_GLOBAL, 1};
                         g.pGlobalBarriers = &gb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB GLOBAL SyncBefore=ALL AccessBefore=NO_ACCESS", true, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_GLOBAL_BARRIER gb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL,
                                                 D3D12_BARRIER_ACCESS_NO_ACCESS, D3D12_BARRIER_ACCESS_COMMON};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_GLOBAL, 1};
                         g.pGlobalBarriers = &gb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB GLOBAL Access=NO_ACCESS|RENDER_TARGET", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_GLOBAL_BARRIER gb{
                             D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL,
                             static_cast<D3D12_BARRIER_ACCESS>(D3D12_BARRIER_ACCESS_NO_ACCESS |
                                                               D3D12_BARRIER_ACCESS_RENDER_TARGET),
                             D3D12_BARRIER_ACCESS_COMMON};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_GLOBAL, 1};
                         g.pGlobalBarriers = &gb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB GLOBAL Sync=COPY Access=RENDER_TARGET", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_GLOBAL_BARRIER gb{D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_SYNC_ALL,
                                                 D3D12_BARRIER_ACCESS_RENDER_TARGET, D3D12_BARRIER_ACCESS_COMMON};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_GLOBAL, 1};
                         g.pGlobalBarriers = &gb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB GLOBAL unknown SyncBefore=0x04000000", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_GLOBAL_BARRIER gb{static_cast<D3D12_BARRIER_SYNC>(0x04000000u), D3D12_BARRIER_SYNC_ALL,
                                                 D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_COMMON};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_GLOBAL, 1};
                         g.pGlobalBarriers = &gb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB GLOBAL unknown AccessBefore=0x04000000", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_GLOBAL_BARRIER gb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL,
                                                 static_cast<D3D12_BARRIER_ACCESS>(0x04000000u),
                                                 D3D12_BARRIER_ACCESS_COMMON};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_GLOBAL, 1};
                         g.pGlobalBarriers = &gb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB BUFFER pResource=nullptr", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_BUFFER_BARRIER bb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL,
                                                 D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_COMMON, nullptr, 0,
                                                 UINT64_MAX};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_BUFFER, 1};
                         g.pBufferBarriers = &bb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB BUFFER Offset=64 Size=64", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource* buf, ID3D12Resource*) {
                         D3D12_BUFFER_BARRIER bb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL,
                                                 D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_COMMON, buf, 64, 64};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_BUFFER, 1};
                         g.pBufferBarriers = &bb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB BUFFER valid Offset=0 Size=UINT64_MAX", true, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource* buf, ID3D12Resource*) {
                         D3D12_BUFFER_BARRIER bb{D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                                                 D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                                                 buf, 0, UINT64_MAX};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_BUFFER, 1};
                         g.pBufferBarriers = &bb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB TEXTURE pResource=nullptr", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource*) {
                         D3D12_TEXTURE_BARRIER tb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL,
                                                  D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_COMMON,
                                                  D3D12_BARRIER_LAYOUT_COMMON, D3D12_BARRIER_LAYOUT_COMMON, nullptr,
                                                  {0xffffffffu, 0, 0, 0, 0, 0}, D3D12_TEXTURE_BARRIER_FLAG_NONE};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_TEXTURE, 1};
                         g.pTextureBarriers = &tb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB TEXTURE unknown LayoutBefore=999", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_TEXTURE_BARRIER tb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL,
                                                  D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_COMMON,
                                                  static_cast<D3D12_BARRIER_LAYOUT>(999), D3D12_BARRIER_LAYOUT_COMMON,
                                                  tex, {0xffffffffu, 0, 0, 0, 0, 0}, D3D12_TEXTURE_BARRIER_FLAG_NONE};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_TEXTURE, 1};
                         g.pTextureBarriers = &tb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB TEXTURE unknown LayoutAfter=999", false, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_TEXTURE_BARRIER tb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL,
                                                  D3D12_BARRIER_ACCESS_COMMON, D3D12_BARRIER_ACCESS_COMMON,
                                                  D3D12_BARRIER_LAYOUT_COMMON, static_cast<D3D12_BARRIER_LAYOUT>(999),
                                                  tex, {0xffffffffu, 0, 0, 0, 0, 0}, D3D12_TEXTURE_BARRIER_FLAG_NONE};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_TEXTURE, 1};
                         g.pTextureBarriers = &tb;
                         l7->Barrier(1, &g);
                     }});
    cases.push_back({"EB TEXTURE valid UNDEFINED->RENDER_TARGET", true, true,
                     [](ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList7* l7, ID3D12Resource*, ID3D12Resource* tex) {
                         D3D12_TEXTURE_BARRIER tb{D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_SYNC_RENDER_TARGET,
                                                  D3D12_BARRIER_ACCESS_NO_ACCESS, D3D12_BARRIER_ACCESS_RENDER_TARGET,
                                                  D3D12_BARRIER_LAYOUT_UNDEFINED, D3D12_BARRIER_LAYOUT_RENDER_TARGET,
                                                  tex, {0xffffffffu, 0, 0, 0, 0, 0}, D3D12_TEXTURE_BARRIER_FLAG_DISCARD};
                         D3D12_BARRIER_GROUP g{D3D12_BARRIER_TYPE_TEXTURE, 1};
                         g.pTextureBarriers = &tb;
                         l7->Barrier(1, &g);
                     }});

    return cases;
}

int run_valid_gpu_replay(DeviceEnv& layer) {
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(layer.device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue.p)))) return 1;

    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(layer.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(layer.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))))
        return 1;
    Ref<ID3D12GraphicsCommandList7> list7;
    if (FAILED(list->QueryInterface(IID_PPV_ARGS(&list7.p))) || !list7.p) return 1;

    // 1. Valid no-ops
    list->ResourceBarrier(0, nullptr);
    list7->Barrier(0, nullptr);

    // 2. Valid legacy transitions (full + split), UAV, and aliasing barriers
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = layer.texture.p;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    list->ResourceBarrier(1, &b);

    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    list->ResourceBarrier(1, &b);
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
    list->ResourceBarrier(1, &b);

    D3D12_RESOURCE_BARRIER uav_all{};
    uav_all.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    D3D12_RESOURCE_BARRIER uav_buf{};
    uav_buf.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav_buf.UAV.pResource = layer.buffer.p;
    D3D12_RESOURCE_BARRIER alias_null{};
    alias_null.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    D3D12_RESOURCE_BARRIER alias_res{};
    alias_res.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    alias_res.Aliasing.pResourceBefore = layer.buffer.p;
    alias_res.Aliasing.pResourceAfter = layer.texture.p;
    D3D12_RESOURCE_BARRIER batch[] = {uav_all, uav_buf, alias_null, alias_res};
    list->ResourceBarrier(4, batch);

    // 3. Valid enhanced global, buffer, and texture barriers
    D3D12_GLOBAL_BARRIER gb{D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_ACCESS_COMMON,
                            D3D12_BARRIER_ACCESS_COMMON};
    D3D12_BUFFER_BARRIER bb{D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            D3D12_BARRIER_ACCESS_COPY_DEST, D3D12_BARRIER_ACCESS_UNORDERED_ACCESS, layer.buffer.p, 0,
                            UINT64_MAX};
    D3D12_TEXTURE_BARRIER tb_to_rt{D3D12_BARRIER_SYNC_NONE, D3D12_BARRIER_SYNC_RENDER_TARGET,
                                   D3D12_BARRIER_ACCESS_NO_ACCESS, D3D12_BARRIER_ACCESS_RENDER_TARGET,
                                   D3D12_BARRIER_LAYOUT_UNDEFINED, D3D12_BARRIER_LAYOUT_RENDER_TARGET, layer.texture.p,
                                   {0xffffffffu, 0, 0, 0, 0, 0}, D3D12_TEXTURE_BARRIER_FLAG_DISCARD};
    D3D12_TEXTURE_BARRIER tb_to_common{D3D12_BARRIER_SYNC_RENDER_TARGET, D3D12_BARRIER_SYNC_ALL,
                                       D3D12_BARRIER_ACCESS_RENDER_TARGET, D3D12_BARRIER_ACCESS_COMMON,
                                       D3D12_BARRIER_LAYOUT_RENDER_TARGET, D3D12_BARRIER_LAYOUT_COMMON, layer.texture.p,
                                       {0xffffffffu, 0, 0, 0, 0, 0}, D3D12_TEXTURE_BARRIER_FLAG_NONE};

    D3D12_BARRIER_GROUP groups[3]{};
    groups[0].Type = D3D12_BARRIER_TYPE_GLOBAL;
    groups[0].NumBarriers = 1;
    groups[0].pGlobalBarriers = &gb;
    groups[1].Type = D3D12_BARRIER_TYPE_BUFFER;
    groups[1].NumBarriers = 1;
    groups[1].pBufferBarriers = &bb;
    groups[2].Type = D3D12_BARRIER_TYPE_TEXTURE;
    groups[2].NumBarriers = 1;
    groups[2].pTextureBarriers = &tb_to_rt;
    list7->Barrier(3, groups);

    D3D12_BARRIER_GROUP back{};
    back.Type = D3D12_BARRIER_TYPE_TEXTURE;
    back.NumBarriers = 1;
    back.pTextureBarriers = &tb_to_common;
    list7->Barrier(1, &back);

    if (FAILED(list->Close())) {
        std::fprintf(stderr, "FAIL GPU replay: Close() failed\n");
        return 1;
    }

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);

    Ref<ID3D12Fence> fence;
    if (FAILED(layer.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))) ||
        FAILED(queue->Signal(fence.p, 1))) {
        std::fprintf(stderr, "FAIL GPU replay: fence signal failed\n");
        return 1;
    }
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 1;
    const bool ok = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                    WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!ok) {
        std::fprintf(stderr, "FAIL GPU replay: fence wait failed\n");
        return 1;
    }
    std::printf("GPU replay (legacy + enhanced barriers): passed\n");
    return 0;
}

}  // namespace

int main() {
    DeviceEnv layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&layer.device.p)))) {
        std::fprintf(stderr, "SKIP: no usable Vulkan device\n");
        return 77;
    }
    if (!layer.init_resources(layer.device.p)) {
        std::fprintf(stderr, "FAIL: layer resource creation failed\n");
        return 1;
    }

    int failures = run_null_pointer_regression(layer);

    HostEnv hw, warp;
    const bool have_hw = hw.init(false);
    const bool have_warp = warp.init(true);
    if (!have_hw) std::printf("note: native hardware D3D12 adapter unavailable; skipping HW parity\n");
    if (!have_warp) std::printf("note: WARP D3D12 adapter unavailable; skipping WARP parity\n");

    const auto cases = build_parity_cases();
    unsigned compared_hw = 0, compared_warp = 0;
    for (const auto& c : cases) {
        const HRESULT layer_hr = run_close_case(layer, c);
        const bool layer_ok = SUCCEEDED(layer_hr);
        if (layer_ok != c.expected_ok) {
            std::fprintf(stderr, "FAIL layer %s: got 0x%08lx, expected %s\n", c.name,
                         static_cast<unsigned long>(layer_hr), c.expected_ok ? "S_OK" : "failure");
            ++failures;
        }
        if (have_hw && (!c.enhanced || hw.enhanced_supported) && hw.ensure_alive()) {
            const HRESULT hw_hr = run_close_case(hw, c);
            ++compared_hw;
            if (SUCCEEDED(hw_hr) != layer_ok) {
                std::fprintf(stderr, "FAIL HW parity %s: layer=0x%08lx host=0x%08lx\n", c.name,
                             static_cast<unsigned long>(layer_hr), static_cast<unsigned long>(hw_hr));
                ++failures;
            }
        }
        if (have_warp && (!c.enhanced || warp.enhanced_supported) && warp.ensure_alive()) {
            const HRESULT warp_hr = run_close_case(warp, c);
            ++compared_warp;
            if (SUCCEEDED(warp_hr) != layer_ok) {
                std::fprintf(stderr, "FAIL WARP parity %s: layer=0x%08lx warp=0x%08lx\n", c.name,
                             static_cast<unsigned long>(layer_hr), static_cast<unsigned long>(warp_hr));
                ++failures;
            }
        }
    }
    std::printf("parity matrix: %zu cases (HW compared=%u, WARP compared=%u), %d failures\n", cases.size(),
                compared_hw, compared_warp, failures);

    if (run_valid_gpu_replay(layer) != 0) ++failures;
    return failures ? 1 : 0;
}
