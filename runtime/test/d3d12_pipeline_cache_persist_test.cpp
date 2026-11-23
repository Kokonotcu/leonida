// D3D-704 on a Vulkan device: the layer's VkPipelineCache outlives the device. Each session creates a layer device on
// the same Vulkan device, creates a compute PSO and releases everything; teardown saves the cache to
// pipeline_cache_runtime_<vendor>_<device>.bin in LEONIDA_PIPELINE_CACHE_DIR and the next device takes it in:
//   - a first session with no file writes one whose header names this device;
//   - the next session loads it, and with no new pipeline leaves it as it was;
//   - a file from another driver (pipelineCacheUUID), a truncated one and one of junk are left out, the PSO is still
//     created, and teardown replaces them with a valid file;
//   - a save that cannot be written (the name is a directory) is logged and the device still goes away cleanly;
//   - LEONIDA_PIPELINE_CACHE_PERSIST=0 neither reads nor writes the file.
// The shader is test/glsl/cs_user_data.comp, as in d3d12_compute_test. No game data. A driver whose pipeline cache
// keeps nothing (only the header) is reported as a skip. Set LEONIDA_VK_VALIDATION=1 to run under the validation layer.
#include "d3d12_test_shaders.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/pipeline_cache_persist.h"
#include "../src/shader_library.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_CS_SPV
#error "LEONIDA_TEST_CS_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace fs = std::filesystem;
using d3d12test::Ref;

std::string contents(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), {}};
}

void put_file(const fs::path& path, const std::string& bytes) {
    std::ofstream(path, std::ios::binary | std::ios::trunc).write(bytes.data(), std::streamsize(bytes.size()));
}

struct Session {
    bool pso_created = false;
    rt::d3d12::PipelineCacheSources sources;
};

// One layer device on the test's Vulkan device: a root signature, one compute PSO, then everything released, which
// tears the layer device down and saves its cache.
Session run_session(const vktest::Device& d, const fs::path& library, const std::vector<std::uint8_t>& root_blob,
                    const std::vector<std::uint8_t>& container) {
    Session out;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.subgroup = d.subgroup;
    info.shader_dir = library.string();
    ID3D12Device* inner = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner)))) return out;
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device.p))))
        return out;
    rt::d3d12::set_device_outer(inner, device.p);
    Ref<ID3D12RootSignature> root;
    if (SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p)))) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root.p;
        desc.CS = {container.data(), container.size()};
        Ref<ID3D12PipelineState> pso;
        out.pso_created = SUCCEEDED(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso.p))) && pso.p;
        out.sources = rt::d3d12::pipeline_cache_sources(device.p);
    }
    return out;  // the Refs release the PSO, the root signature and the device here
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);

    leonida::cached_pso::Identity identity;
    {
        const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            d.instance_proc(d.instance, "vkGetPhysicalDeviceProperties"));
        EXPECT(get_props);
        VkPhysicalDeviceProperties props{};
        get_props(d.physical, &props);
        identity.vendor = props.vendorID;
        identity.device = props.deviceID;
        identity.driver = props.driverVersion;
        std::memcpy(identity.cache_uuid, props.pipelineCacheUUID, VK_UUID_SIZE);
        std::printf("device: %s (%04x:%04x)\n", props.deviceName, props.vendorID, props.deviceID);
    }

    // The program the PSO carries and a translator index that maps its hash to the test module.
    const auto fssm = d3d12test::fssm_program({5000, 0xbf810000u, 0xbf810001u, 0xbf810002u});
    const auto container = d3d12test::shader_container(fssm);
    const testfiles::ScratchDir scratch("leonida_pipeline_cache_persist_test");
    const fs::path library = scratch.path() / "shaders";
    const fs::path caches = scratch.path() / "caches";
    EXPECT(fs::create_directories(library));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SPV), library / "cs_user_data.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_user_data\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(fssm)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": true, \"wave64\": false}}";
    }
    // The console compute layout of d3d12_compute_test: two constants, a root UAV, a one-CBV table.
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1, 0, 0};
    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 2};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor = {0, 0};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {1, &range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_blob = d3d12test::serialize_root_signature(root_desc);
    EXPECT(!root_blob.empty());

    EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_DIR", caches.wstring().c_str()));
    EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_PERSIST", nullptr) || GetLastError() == ERROR_ENVVAR_NOT_FOUND);
    const fs::path saved = caches / leonida::pipeline_cache::file_name(identity);
    const auto run = [&] { return run_session(d, library, root_blob, container); };
    const auto valid_file = [&] {
        const std::string bytes = contents(saved);
        return bytes.size() > sizeof(VkPipelineCacheHeaderVersionOne) &&
               leonida::pipeline_cache::check_header(bytes.data(), bytes.size(), identity) ==
                   leonida::pipeline_cache::Header::ok;
    };

    // 1. No file yet: the cache starts empty, and teardown writes one (the directory is made).
    auto session = run();
    EXPECT(session.pso_created);
    EXPECT(session.sources.saved_bytes == 0);
    if (!fs::exists(saved)) {
        std::printf("SKIP: this driver's pipeline cache keeps nothing beyond its header, so nothing is saved\n");
        return 77;
    }
    EXPECT(valid_file());
    const std::string first = contents(saved);
    std::printf("session 1: saved %zu bytes to %s\n", first.size(), saved.filename().string().c_str());

    // 2. The next device takes the file in; the same PSO adds nothing, so the file is not written again.
    const auto written = fs::last_write_time(saved);
    session = run();
    EXPECT(session.pso_created);
    EXPECT(session.sources.saved_bytes == first.size());
    EXPECT(contents(saved) == first);
    EXPECT(fs::last_write_time(saved) == written);
    std::printf("session 2: loaded %zu bytes, nothing new, the file was left alone\n", session.sources.saved_bytes);

    // 3. Files the device must not take: another driver's, truncated, junk. Each is left out and then replaced.
    std::string other_driver = first;
    other_driver[16] = char(other_driver[16] ^ 0xff);  // pipelineCacheUUID starts at byte 16
    const std::string bad_files[] = {other_driver, first.substr(0, 20), std::string(4096, '\x5a'), std::string()};
    for (const std::string& bad : bad_files) {
        put_file(saved, bad);
        session = run();
        EXPECT(session.pso_created);
        EXPECT(session.sources.saved_bytes == 0);
        EXPECT(valid_file());
    }
    std::printf("session 3: another driver's, truncated, junk and empty files were left out and replaced\n");

    // 4. A save that cannot be written: logged, the device goes away, the PSO worked.
    EXPECT(fs::remove(saved));
    EXPECT(fs::create_directory(saved));
    session = run();
    EXPECT(session.pso_created);
    EXPECT(fs::is_directory(saved));
    EXPECT(std::distance(fs::directory_iterator(caches), fs::directory_iterator{}) == 1);  // no staging file left
    EXPECT(fs::remove(saved));
    std::printf("session 4: a failed save left nothing behind\n");

    // 5. LEONIDA_PIPELINE_CACHE_PERSIST=0: the file is neither read nor written.
    put_file(saved, first);
    EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_PERSIST", L"0"));
    session = run();
    EXPECT(session.pso_created);
    EXPECT(session.sources.saved_bytes == 0);
    EXPECT(fs::remove(saved));
    session = run();
    EXPECT(session.pso_created);
    EXPECT(!fs::exists(saved));
    SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_PERSIST", nullptr);
    std::printf("session 5: LEONIDA_PIPELINE_CACHE_PERSIST=0 neither read nor wrote the file\n");

    std::printf("pipeline cache persistence: saved at teardown, loaded by the next device\n");
    return 0;
}
