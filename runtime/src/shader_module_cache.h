#pragma once

// One VkShaderModule per distinct SPIR-V, shared by every pipeline state built from it.
//
// A title creates tens of thousands of graphics pipeline states up front, most of them from a few thousand vertex and
// pixel shaders (measured on the retail boot under Wine: 50,468 graphics PSOs in the first tenth of the run, about
// 12.7 GB resident, #8851 A37/A42). A PipelineState keeps its modules for the variants it builds at draw time, so
// one module per PSO and stage kept a copy of the same shader per PSO. Here a module is created on the first acquire
// of its words and destroyed with the last release; a module that holds per-PSO edits (the depth24 patch, the f16
// link) has other words and gets its own module, so sharing never changes what a pipeline runs.
//
// Thread-safe: PSOs are created and released from any thread.

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace leonida {

class ShaderModuleCache {
public:
    // `create` returns VK_NULL_HANDLE when vkCreateShaderModule fails; `destroy` is vkDestroyShaderModule.
    using Create = std::function<VkShaderModule(const std::vector<std::uint32_t>& words)>;
    using Destroy = std::function<void(VkShaderModule)>;

    ShaderModuleCache(Create create, Destroy destroy) : create_(std::move(create)), destroy_(std::move(destroy)) {}
    ShaderModuleCache(const ShaderModuleCache&) = delete;
    ShaderModuleCache& operator=(const ShaderModuleCache&) = delete;

    // Releases nothing a caller still holds: every PipelineState releases its modules before the device goes.
    // Modules left over (a leaked PSO) are destroyed here, with the device still alive.
    ~ShaderModuleCache() {
        for (auto& [hash, bucket] : by_hash_)
            for (Entry& entry : bucket) destroy_(entry.module);
    }

    using Words = std::shared_ptr<const std::vector<std::uint32_t>>;

    // Pending graphics recipes keep the translated program on the CPU until replay needs it.
    // This table retains only weak references. A hash collision still requires equal words.
    Words retain_words(std::vector<std::uint32_t> words) {
        const auto hash = hash_words(words);
        std::lock_guard lock(mutex_);
        auto& bucket = pending_[hash];
        for (auto it = bucket.begin(); it != bucket.end();) {
            if (auto live = it->lock()) {
                if (*live == words) return live;
                ++it;
            } else {
                it = bucket.erase(it);
            }
        }
        auto program = std::make_shared<const std::vector<std::uint32_t>>(std::move(words));
        bucket.push_back(program);
        return program;
    }

    // The module for these words, created on first use. VK_NULL_HANDLE when creation fails (nothing is cached).
    // The driver's vkCreateShaderModule runs outside the lock: under it, every pipeline creation on every thread
    // waited for the one in front (the retail boot's game threads sat in this lock for most of a 15 s startup gap).
    // Two threads that miss on the same words both create a module; the second to insert takes the first's and
    // destroys its own.
    VkShaderModule acquire(const std::vector<std::uint32_t>& words) {
        const Fingerprint print = fingerprint(words);
        const std::uint64_t hash = print.a;
        if (const VkShaderModule found = find_and_reference(hash, print)) return found;
        const VkShaderModule created = create_(words);
        if (created == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        VkShaderModule duplicate = VK_NULL_HANDLE;
        VkShaderModule result = created;
        {
            std::lock_guard lock(mutex_);
            std::vector<Entry>& bucket = by_hash_[hash];
            for (Entry& entry : bucket) {
                if (entry.print == print) {  // another thread inserted it meanwhile
                    ++entry.references;
                    ++shared_;
                    duplicate = created;
                    result = entry.module;
                    break;
                }
            }
            if (!duplicate) {
                bucket.push_back(Entry{print, created, 1});
                owner_[created] = hash;
            }
        }
        if (duplicate) destroy_(duplicate);
        return result;
    }

    // One release per acquire that returned `module`; the last one destroys it. Unknown modules are ignored.
    void release(VkShaderModule module) {
        if (module == VK_NULL_HANDLE) return;
        std::lock_guard lock(mutex_);
        const auto owner = owner_.find(module);
        if (owner == owner_.end()) return;
        const auto bucket = by_hash_.find(owner->second);
        for (auto entry = bucket->second.begin(); entry != bucket->second.end(); ++entry) {
            if (entry->module != module) continue;
            if (--entry->references == 0) {
                destroy_(module);
                bucket->second.erase(entry);
                if (bucket->second.empty()) by_hash_.erase(bucket);
                owner_.erase(owner);
            }
            return;
        }
    }

    // Distinct modules alive, and acquires answered with an existing module (for the device's statistics line).
    std::size_t live() const {
        std::lock_guard lock(mutex_);
        return owner_.size();
    }
    std::uint64_t shared() const {
        std::lock_guard lock(mutex_);
        return shared_;
    }

private:
    // What identifies a module's words: two independent 64-bit hashes and the length. An entry kept the words
    // themselves before (2026-11-25): one copy of every live module's SPIR-V, 1.4-3.5 MiB each for the retail title's
    // translated shaders, about 11 GB of heap two minutes into a boot (each a separate large heap block, found by
    // their 0x07230203 headers). Two different programs matching in 128 bits of hash and their length do not occur.
    struct Fingerprint {
        std::uint64_t a = 0, b = 0;
        std::size_t words = 0;
        bool operator==(const Fingerprint&) const = default;
    };
    static Fingerprint fingerprint(const std::vector<std::uint32_t>& words) {
        Fingerprint f{hash_words(words), 0x9E3779B97F4A7C15ull, words.size()};
        for (const std::uint32_t word : words) {  // a multiply-rotate mix, independent of FNV-1a
            f.b = (f.b ^ word) * 0xC2B2AE3D27D4EB4Full;
            f.b = (f.b << 31) | (f.b >> 33);
        }
        f.b ^= words.size() * 0x165667B19E3779F9ull;
        return f;
    }

    // The cached module for these words with one more reference, or VK_NULL_HANDLE.
    VkShaderModule find_and_reference(std::uint64_t hash, const Fingerprint& print) {
        std::lock_guard lock(mutex_);
        const auto bucket = by_hash_.find(hash);
        if (bucket == by_hash_.end()) return VK_NULL_HANDLE;
        for (Entry& entry : bucket->second) {
            if (entry.print == print) {
                ++entry.references;
                ++shared_;
                return entry.module;
            }
        }
        return VK_NULL_HANDLE;
    }

    struct Entry {
        Fingerprint print;
        VkShaderModule module;
        std::uint32_t references;
    };

    static std::uint64_t hash_words(const std::vector<std::uint32_t>& words) {
        std::uint64_t hash = 14695981039346656037ull;  // FNV-1a over the words
        for (const std::uint32_t word : words) {
            hash ^= word;
            hash *= 1099511628211ull;
        }
        return hash ^ words.size();
    }

    std::unordered_map<std::uint64_t, std::vector<std::weak_ptr<const std::vector<std::uint32_t>>>> pending_;
    Create create_;
    Destroy destroy_;
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, std::vector<Entry>> by_hash_;
    std::unordered_map<VkShaderModule, std::uint64_t> owner_;
    std::uint64_t shared_ = 0;
};

}  // namespace leonida
