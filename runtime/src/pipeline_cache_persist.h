// D3D-704: the runtime keeps the VkPipelineCache entries it learns in a session and offers them to the next one.
//
// Two files feed a device's cache on its first PSO: the `pipelines` build stage's pipeline_cache.bin (the seed, D3D-34)
// and the runtime's own pipeline_cache_runtime_<vendor>_<device>.bin, which device teardown writes. Each is taken only
// when its VkPipelineCacheHeaderVersionOne names this device's vendor, device and pipelineCacheUUID; when both are,
// the second is merged into the first (vkMergePipelineCaches). Teardown writes the cache's data when it grew since it
// was loaded, through a same-directory temporary file and a rename, so a failed save leaves the old file as it was.
// The vendor and device in the name keep two GPUs of one machine from replacing each other's file; a driver update
// changes the UUID, so the old file is ignored once and then replaced.
// Portable: no Vulkan device and no Windows API here, so runtime/test/pipeline_cache_persist_test.cpp runs anywhere.
#pragma once

#include "cached_pso.h"

#include <vulkan/vulkan_core.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <system_error>
#include <vector>

namespace leonida::pipeline_cache {

// Why a file's data was or was not given to vkCreatePipelineCache.
enum class Header {
    ok,
    truncated,     // shorter than VkPipelineCacheHeaderVersionOne
    bad_header,    // headerSize or headerVersion is not VK_PIPELINE_CACHE_HEADER_VERSION_ONE's
    other_device,  // another vendorID or deviceID
    other_driver,  // the same GPU, another pipelineCacheUUID (a driver update)
};

inline const char* describe(Header header) {
    switch (header) {
        case Header::ok: return "matches this device";
        case Header::truncated: return "is shorter than a Vulkan pipeline cache header";
        case Header::bad_header: return "has no Vulkan pipeline cache header (version one)";
        case Header::other_device: return "was made on another GPU";
        case Header::other_driver: return "was made by another driver version (pipelineCacheUUID)";
    }
    return "?";
}

inline Header check_header(const void* data, std::size_t size, const cached_pso::Identity& identity) {
    static_assert(sizeof(VkPipelineCacheHeaderVersionOne) == 32);
    VkPipelineCacheHeaderVersionOne header{};
    if (!data || size < sizeof(header)) return Header::truncated;
    // The header is little-endian, like the host. The driver interprets the opaque payload.
    std::memcpy(&header, data, sizeof(header));
    if (header.headerSize != sizeof(header) || header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
        return Header::bad_header;
    if (header.vendorID != identity.vendor || header.deviceID != identity.device) return Header::other_device;
    if (std::memcmp(header.pipelineCacheUUID, identity.cache_uuid, VK_UUID_SIZE) != 0) return Header::other_driver;
    return Header::ok;
}

// The saved file's name for this device: pipeline_cache_runtime_1002_164e.bin.
inline std::string file_name(const cached_pso::Identity& identity) {
    char name[64];
    std::snprintf(name, sizeof(name), "pipeline_cache_runtime_%04x_%04x.bin", unsigned(identity.vendor),
                  unsigned(identity.device));
    return name;
}

enum class Saved {
    written,
    unchanged,  // the cache holds what it was loaded with: no new pipeline, nothing to write
    empty,      // the driver keeps nothing beyond the header
    failed,     // reading the cache or writing the file failed; an existing file is left as it was
};

struct SaveResult {
    Saved saved = Saved::failed;
    std::size_t bytes = 0;  // the cache's data size
    std::string error;      // what failed, for the log
};

// Teardown's decision and write. `get` is vkGetPipelineCacheData bound to the device and cache:
// VkResult(size_t* size, void* data). `snapshot` reads the whole cache through `get` (pipeline_cache_snapshot, which
// copes with VK_INCOMPLETE) and may throw. `write` publishes the bytes atomically and returns an error code
// (write_pipeline_cache_file). `loaded_bytes` is the cache's data size right after it was created: an in-memory cache
// only ever gains entries, so the same size means nothing was added.
template<class Get, class Snapshot, class Write>
SaveResult save(Get get, Snapshot snapshot, Write write, std::size_t loaded_bytes) {
    SaveResult out;
    try {
        std::size_t size = 0;
        if (const VkResult result = get(&size, nullptr); result != VK_SUCCESS) {
            out.error = "vkGetPipelineCacheData returned " + std::to_string(int(result));
            return out;
        }
        out.bytes = size;
        if (size <= sizeof(VkPipelineCacheHeaderVersionOne)) {
            out.saved = Saved::empty;
            return out;
        }
        if (size == loaded_bytes) {
            out.saved = Saved::unchanged;
            return out;
        }
        const std::vector<char> data = snapshot(get);
        out.bytes = data.size();
        if (const std::error_code error = write(data.data(), data.size())) {
            out.error = error.message();
            return out;
        }
        out.saved = Saved::written;
    } catch (const std::exception& error) {
        out.saved = Saved::failed;
        out.error = error.what();
    }
    return out;
}

}  // namespace leonida::pipeline_cache
