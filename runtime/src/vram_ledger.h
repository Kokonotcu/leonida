#pragma once
// Every vkAllocateMemory and vkFreeMemory the runtime makes, counted: live bytes per memory type and per calling site
// (the return address in gta6rt.exe). The runtime's modules each load their own device functions, but all of them
// start from the one vkGetInstanceProcAddr that wrap() replaces, so nothing escapes the count. What the driver reports
// for a heap (VK_EXT_memory_budget) minus what this ledger holds for it is the driver's own memory.
// Cost: one short lock per allocation and free (a few thousand in a boot), none per frame.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida::vram_ledger {

struct Entry {
    VkDeviceSize bytes;
    std::uint32_t type;
    std::uintptr_t site;  // RVA in gta6rt.exe, or the absolute return address outside it
};

struct Ledger {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, Entry> live;               // VkDeviceMemory -> entry
    VkDeviceSize type_bytes[VK_MAX_MEMORY_TYPES]{};
    std::size_t type_count[VK_MAX_MEMORY_TYPES]{};
    std::unordered_map<std::uintptr_t, std::pair<VkDeviceSize, std::size_t>> sites;  // site -> live bytes, count
};
inline Ledger& ledger() {
    static Ledger* instance = new Ledger;  // never destroyed: frees can come during process teardown
    return *instance;
}

inline PFN_vkGetInstanceProcAddr g_instance_proc = nullptr;
inline PFN_vkGetDeviceProcAddr g_device_proc = nullptr;
inline PFN_vkAllocateMemory g_allocate = nullptr;
inline PFN_vkFreeMemory g_free = nullptr;

// What the allocations made on this thread are for, set by a Purpose around the allocating call. A plain pointer to a
// string literal: gta6rt has no thread_local with a destructor (src/leonida-no-thread-local-dtors).
inline thread_local const char* t_purpose = nullptr;
struct Purpose {
    const char* previous;
    explicit Purpose(const char* what) : previous(t_purpose) { t_purpose = what; }
    ~Purpose() { t_purpose = previous; }
    Purpose(const Purpose&) = delete;
    Purpose& operator=(const Purpose&) = delete;
};

constexpr std::uintptr_t kNamed = std::uintptr_t(1) << 63;  // a site that is a Purpose's text, not an address

// The Purpose when one is set, else the return address (an RVA in gta6rt.exe when it is in it).
inline std::uintptr_t site_of(void* return_address) {
    if (t_purpose) return reinterpret_cast<std::uintptr_t>(t_purpose) | kNamed;
    static const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
    const auto at = reinterpret_cast<std::uintptr_t>(return_address);
    return at >= base && at - base < 0x10000000 ? at - base : at;
}

// A site and memory type in one key: the type in bits 56-62 (addresses and pointers stay below them).
constexpr std::uintptr_t kTypeShift = 56, kTypeMask = std::uintptr_t(0x7f) << kTypeShift;
inline std::uintptr_t key(const Entry& e) { return e.site | (std::uintptr_t(e.type) << kTypeShift); }

inline VkResult VKAPI_CALL allocate(VkDevice device, const VkMemoryAllocateInfo* info,
                                    const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
    const VkResult result = g_allocate(device, info, callbacks, memory);
    if (result != VK_SUCCESS || !info || !memory || !*memory || info->memoryTypeIndex >= VK_MAX_MEMORY_TYPES)
        return result;
    const Entry entry{info->allocationSize, info->memoryTypeIndex, site_of(_ReturnAddress())};
    Ledger& l = ledger();
    std::lock_guard lock(l.mutex);
    l.live[reinterpret_cast<std::uint64_t>(*memory)] = entry;
    l.type_bytes[entry.type] += entry.bytes;
    ++l.type_count[entry.type];
    auto& site = l.sites[key(entry)];
    site.first += entry.bytes;
    ++site.second;
    return result;
}

inline void VKAPI_CALL free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
    if (memory) {
        Ledger& l = ledger();
        std::lock_guard lock(l.mutex);
        if (const auto it = l.live.find(reinterpret_cast<std::uint64_t>(memory)); it != l.live.end()) {
            const Entry& e = it->second;
            l.type_bytes[e.type] -= e.bytes;
            --l.type_count[e.type];
            if (const auto s = l.sites.find(key(e)); s != l.sites.end()) {
                s->second.first -= e.bytes;
                if (--s->second.second == 0) l.sites.erase(s);
            }
            l.live.erase(it);
        }
    }
    g_free(device, memory, callbacks);
}

inline PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name);

// The two counted functions in place of the driver's; everything else is the driver's own.
inline PFN_vkVoidFunction replace(const char* name, PFN_vkVoidFunction real) {
    if (!real || !name) return real;
    if (!std::strcmp(name, "vkAllocateMemory")) {
        g_allocate = reinterpret_cast<PFN_vkAllocateMemory>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(&allocate);
    }
    if (!std::strcmp(name, "vkFreeMemory")) {
        g_free = reinterpret_cast<PFN_vkFreeMemory>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(&free_memory);
    }
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) {
        g_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(real);
        return reinterpret_cast<PFN_vkVoidFunction>(&device_proc);
    }
    return real;
}

inline PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    return replace(name, g_device_proc(device, name));
}

inline PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    return replace(name, g_instance_proc(instance, name));
}

// The loader's vkGetInstanceProcAddr, counted from here on.
inline PFN_vkGetInstanceProcAddr wrap(PFN_vkGetInstanceProcAddr real) {
    if (!real) return real;
    g_instance_proc = real;
    return &instance_proc;
}

// Live bytes of the memory types whose heap is `heap`.
inline VkDeviceSize heap_bytes(const VkPhysicalDeviceMemoryProperties& memory, std::uint32_t heap) {
    Ledger& l = ledger();
    std::lock_guard lock(l.mutex);
    VkDeviceSize total = 0;
    for (std::uint32_t t = 0; t < memory.memoryTypeCount && t < VK_MAX_MEMORY_TYPES; ++t)
        if (memory.memoryTypes[t].heapIndex == heap) total += l.type_bytes[t];
    return total;
}

// "ours: heap 0 2900 MiB in 812 allocations, heap 1 ...; largest sites: +0x1234 1696 MiB (7), ..."
inline std::string text(const VkPhysicalDeviceMemoryProperties& memory, std::size_t sites_shown = 6) {
    Ledger& l = ledger();
    std::lock_guard lock(l.mutex);
    std::string out = "ours:";
    char part[96];
    for (std::uint32_t h = 0; h < memory.memoryHeapCount; ++h) {
        VkDeviceSize bytes = 0;
        std::size_t count = 0;
        for (std::uint32_t t = 0; t < memory.memoryTypeCount && t < VK_MAX_MEMORY_TYPES; ++t)
            if (memory.memoryTypes[t].heapIndex == h) {
                bytes += l.type_bytes[t];
                count += l.type_count[t];
            }
        std::snprintf(part, sizeof part, " heap %u %llu MiB in %zu allocations;", h,
                      static_cast<unsigned long long>(bytes >> 20), count);
        out += part;
    }
    std::vector<std::pair<VkDeviceSize, std::pair<std::uintptr_t, std::size_t>>> sorted;
    for (const auto& [site, v] : l.sites) sorted.push_back({v.first, {site, v.second}});
    std::sort(sorted.rbegin(), sorted.rend());
    out += " largest:";
    for (std::size_t i = 0; i < sorted.size() && i < sites_shown; ++i) {
        const std::uintptr_t site = sorted[i].second.first & ~kTypeMask;
        const unsigned type = unsigned((sorted[i].second.first & kTypeMask) >> kTypeShift);
        if (site & kNamed)
            std::snprintf(part, sizeof part, " %s t%u %llu MiB (%zu),", reinterpret_cast<const char*>(site & ~kNamed),
                          type, static_cast<unsigned long long>(sorted[i].first >> 20), sorted[i].second.second);
        else
            std::snprintf(part, sizeof part, " %#llx t%u %llu MiB (%zu),", static_cast<unsigned long long>(site), type,
                          static_cast<unsigned long long>(sorted[i].first >> 20), sorted[i].second.second);
        out += part;
    }
    return out;
}

}  // namespace leonida::vram_ledger
