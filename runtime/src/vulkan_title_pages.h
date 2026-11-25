// Shared 64 KiB pages of video memory for the textures a title places in its own memory (Xbox device slot 133).
//
// On the console a placed texture is its memory: the GPU samples the title's pages, the title reuses them (its
// streaming pool hands freed memory to newer textures, transient targets of a frame share addresses) and creates
// several resources over one address expecting one memory. Here each such texture used to be a committed image of its
// own, kept until the title released the object: in the retail title's first loading screen 32,000 textures summed
// 8.9 GB over 6.0 GB of title memory, 12,000 of them under newer ones, and the overflow past an 8 GB card's VRAM went
// to system memory on top of the title's own 14 GB.
//
// So, each texture is a sparse-residency image on a pool of 64 KiB pages of video memory (system memory past the VRAM
// budget). A render target or UAV texture takes the console's model: one page for each 64 KiB page of the title's
// address space it covers (tile i to the i-th 64 KiB page boundary inside its range; tiles past its console size to
// pages keyed by (address, size, tile)), shared by every target over that memory. A texture the CPU fills has pages of
// its own, bound only for the mips the title's views read (see place). Pages no texture uses go back to the pool.
// (Superseding streamed textures over another's memory, XCL's rule, unbound textures the title still drew: its
// streaming pool packs partly resident textures 16 KiB apart over ranges of 684 KiB.)
//
// Design after XCL's vram.rs (github.com/ivall/xcl, "GPU-owned textures in title memory, aliased as on the console",
// GPL-2.0-only: the design is followed, no code is taken), translated from D3D12 reserved resources and
// UpdateTileMappings to Vulkan sparse images and vkQueueBindSparse.
#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <vector>

namespace leonida {

class TitlePages {
public:
    static constexpr VkDeviceSize kPage = 64 * 1024;
    static constexpr std::uint32_t kBlockPages = 256;  // 16 MiB of video memory per allocation

    struct Functions {
        // One block of `size` bytes of memory type `type`, or VK_NULL_HANDLE. The device's allocator: it refuses a
        // device-local block past the driver's VRAM budget, so the pool goes on in system memory instead of
        // overcommitting the card (8.1 GB used of a 7.2 GB budget ended a run in a GPU page fault).
        std::function<VkDeviceMemory(std::uint32_t type, VkDeviceSize size)> allocate;
        PFN_vkFreeMemory free = nullptr;
        PFN_vkGetImageMemoryRequirements requirements = nullptr;
        PFN_vkGetImageSparseMemoryRequirements sparse_requirements = nullptr;
    };

    // One 64 KiB piece of an image: a block of texels of a mip level (image bind) or a page of a mip tail or metadata
    // (opaque bind at resource_offset).
    struct Tile {
        bool opaque = false;
        VkImageSubresource subresource{};
        VkOffset3D offset{};
        VkExtent3D extent{};
        VkDeviceSize resource_offset = 0;
        VkSparseMemoryBindFlags flags = 0;
        // The mip the tile holds texels of (a mip tail page: the tail's first mip; metadata: 0, bound with any).
        std::uint32_t mip = 0;
    };
    static constexpr std::uint32_t kNoMip = ~0u;

    // The binds one vkQueueBindSparse batch makes for an image: `first` when the image was never bound (its layout is
    // set before, so a first use cannot discard the contents of pages it shares).
    struct Op {
        std::uintptr_t id = 0;
        VkImage image = VK_NULL_HANDLE;
        bool first = false;
        bool full = true;  // binds all the image's tiles (a render target); false: more tiles of a streamed texture
        std::vector<VkSparseImageMemoryBind> image_binds;
        std::vector<VkSparseMemoryBind> opaque_binds;
    };

    TitlePages(VkDevice device, Functions functions, const VkPhysicalDeviceMemoryProperties& memory)
        : device_(device), fn_(functions), memory_(memory) {}
    TitlePages(const TitlePages&) = delete;
    TitlePages& operator=(const TitlePages&) = delete;
    ~TitlePages() {
        for (const Block& block : blocks_) fn_.free(device_, block.memory, nullptr);
    }

    // The tiles of a sparse-residency image in page order: per array layer, per mip level above the mip tail, its
    // blocks row by row, then the mip tails (one, or one per layer) and any metadata, a page each. False when the
    // image's sparse layout is not one of 64 KiB blocks with a standard shape.
    bool layout(VkImage image, const VkImageCreateInfo& info, std::vector<Tile>& tiles, std::uint32_t& types) const {
        VkMemoryRequirements requirements{};
        fn_.requirements(device_, image, &requirements);
        if (requirements.alignment != kPage || requirements.size % kPage) return false;
        types = requirements.memoryTypeBits;
        std::uint32_t count = 0;
        fn_.sparse_requirements(device_, image, &count, nullptr);
        if (!count) return false;
        std::vector<VkSparseImageMemoryRequirements> sparse(count);
        fn_.sparse_requirements(device_, image, &count, sparse.data());
        tiles.clear();
        const std::uint32_t layers = info.imageType == VK_IMAGE_TYPE_3D ? 1 : info.arrayLayers;
        std::vector<Tile> tails;
        for (const auto& r : sparse) {
            const auto& f = r.formatProperties;
            if (f.flags & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT) return false;
            const bool single = (f.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0;
            if (f.aspectMask & VK_IMAGE_ASPECT_METADATA_BIT) {
                for (std::uint32_t layer = 0; layer < (single ? 1u : layers); ++layer)
                    add_pages(tails, r.imageMipTailOffset + layer * r.imageMipTailStride, r.imageMipTailSize,
                              VK_SPARSE_MEMORY_BIND_METADATA_BIT, 0);
                continue;
            }
            const VkExtent3D g = f.imageGranularity;
            if (!g.width || !g.height || !g.depth) return false;
            const std::uint32_t tail_first = std::min(r.imageMipTailFirstLod, info.mipLevels);
            for (std::uint32_t layer = 0; layer < layers; ++layer) {
                for (std::uint32_t mip = 0; mip < tail_first; ++mip) {
                    const std::uint32_t w = std::max(1u, info.extent.width >> mip);
                    const std::uint32_t h = std::max(1u, info.extent.height >> mip);
                    const std::uint32_t d = std::max(1u, info.extent.depth >> mip);
                    for (std::uint32_t z = 0; z < d; z += g.depth)
                        for (std::uint32_t y = 0; y < h; y += g.height)
                            for (std::uint32_t x = 0; x < w; x += g.width) {
                                Tile t;
                                t.mip = mip;
                                t.subresource = {f.aspectMask, mip, layer};
                                t.offset = {std::int32_t(x), std::int32_t(y), std::int32_t(z)};
                                t.extent = {std::min(g.width, w - x), std::min(g.height, h - y),
                                            std::min(g.depth, d - z)};
                                tiles.push_back(t);
                            }
                }
            }
            if (tail_first < info.mipLevels && r.imageMipTailSize)
                for (std::uint32_t layer = 0; layer < (single ? 1u : layers); ++layer)
                    add_pages(tails, r.imageMipTailOffset + layer * r.imageMipTailStride, r.imageMipTailSize, 0,
                              tail_first);
        }
        tiles.insert(tiles.end(), tails.begin(), tails.end());
        return !tiles.empty();
    }

    // Places `image` (sparse, never bound) over the title's [va, va + bytes). False when no page could be had (the
    // caller makes a committed image).
    // A render target or UAV texture (`streamed` false) is bound at once to the title's pages by address: tile i to the
    // i-th 64 KiB page boundary inside its range. GPU targets the title places over one address are transients of a
    // frame that take turns on the same memory, so they share its pages, as on the console.
    // A texture the CPU fills (`streamed`) gets pages of its own (shared only with textures at the same address and
    // size), bound per mip as the title's views ask for them (require_mip). Its PC tiles are in another order than its
    // console layout, so sharing pages by address with its neighbours mixed their texels. And the title's streaming
    // pool packs partly resident textures: on the console a texture's smallest mips come first in its memory, and a
    // 1024x1024 BC1 texture (684 KiB) with only its mip tail resident is placed 16 KiB before the next one, its view
    // starting at the first resident mip. Binding only the mips the views name keeps such a texture to its tail page
    // instead of 684 KiB, as the console keeps it to 16 KiB.
    bool place(std::uintptr_t id, VkImage image, std::vector<Tile> tiles, std::uint32_t types, std::uint64_t va,
               std::uint64_t bytes, bool streamed) {
        std::lock_guard lock(mutex_);
        Target t;
        t.image = image;
        t.tiles = std::move(tiles);
        t.types = types;
        t.va = va;
        t.bytes = bytes;
        t.streamed = streamed;
        if (!streamed) {
            if (!acquire(t)) return false;
            queue_bind(id, t, true);
        }
        t.mapped = true;
        targets_.emplace(id, std::move(t));
        ++placed_;
        return true;
    }

    // A view of streamed texture `id` reads from mip `mip` on: binds that mip's tiles and those of every mip between it
    // and the ones bound already (on pages of its own). True when it bound something: the texture's bytes are then
    // uploaded again (`take_remapped`). False otherwise, also when the pool could not grow (the mips stay unbound and
    // read zero).
    bool require_mip(std::uintptr_t id, std::uint32_t mip) {
        std::lock_guard lock(mutex_);
        const auto it = targets_.find(id);
        if (it == targets_.end() || !it->second.streamed) return false;
        Target& t = it->second;
        if (t.bound_mip != kNoMip && mip >= t.bound_mip) return false;
        if (t.slots.size() != t.tiles.size()) t.slots.assign(t.tiles.size(), {});
        Op op;
        op.id = id;
        op.image = t.image;
        op.first = !t.ever_bound;
        op.full = false;
        const std::size_t held = t.extra.size();
        for (std::size_t i = 0; i < t.tiles.size(); ++i) {
            const Tile& tile = t.tiles[i];
            // Already bound: its mip is at or past the bound one; metadata goes with the first bind.
            const bool metadata = (tile.flags & VK_SPARSE_MEMORY_BIND_METADATA_BIT) != 0;
            const bool bound = t.bound_mip != kNoMip && (metadata || tile.mip >= t.bound_mip);
            if (bound || (!metadata && tile.mip < mip)) continue;
            Slot slot;
            if (!take_own(t, std::uint32_t(i), slot)) {
                // Nothing of this call stays held: what it took goes back.
                while (t.extra.size() > held) {
                    put_own(t.extra.back());
                    t.extra.pop_back();
                }
                return false;
            }
            t.slots[i] = slot;
            const Block& block = blocks_[slot.block];
            const VkDeviceSize offset = VkDeviceSize(slot.index) * kPage;
            if (tile.opaque)
                op.opaque_binds.push_back({tile.resource_offset, kPage, block.memory, offset, tile.flags});
            else
                op.image_binds.push_back({tile.subresource, tile.offset, tile.extent, block.memory, offset, 0});
        }
        t.bound_mip = mip;
        t.ever_bound = true;
        if (op.image_binds.empty() && op.opaque_binds.empty()) return false;
        pending_.push_back(std::move(op));
        remapped_.push_back(id);
        return true;
    }

    // The title released the texture: its pages are no longer its.
    void release(std::uintptr_t id) {
        std::lock_guard lock(mutex_);
        const auto it = targets_.find(id);
        if (it == targets_.end()) return;
        Target t = std::move(it->second);
        targets_.erase(it);
        // A bind still waiting to be made names an image about to be destroyed.
        pending_.erase(std::remove_if(pending_.begin(), pending_.end(), [&](const Op& op) { return op.id == id; }),
                       pending_.end());
        give_back(t);
    }

    // Whether the texture is on pages (bound, or streamed and bound as its views ask).
    bool mapped(std::uintptr_t id) const {
        std::lock_guard lock(mutex_);
        const auto it = targets_.find(id);
        return it != targets_.end() && it->second.mapped;
    }
    // Bound (a streamed texture: some mip of it) and no bind of it waiting: a copy into it now lands in its pages (one
    // before the flush would be lost).
    bool ready(std::uintptr_t id) const {
        std::lock_guard lock(mutex_);
        const auto it = targets_.find(id);
        if (it == targets_.end() || !it->second.mapped) return false;
        if (it->second.streamed && it->second.bound_mip == kNoMip) return false;
        return std::none_of(pending_.begin(), pending_.end(), [&](const Op& op) { return op.id == id; });
    }
    // The most detailed mip with texels in memory: copies of the mips before it would land nowhere.
    std::uint32_t first_mip(std::uintptr_t id) const {
        std::lock_guard lock(mutex_);
        const auto it = targets_.find(id);
        if (it == targets_.end() || !it->second.streamed) return 0;
        return it->second.bound_mip == kNoMip ? 0 : it->second.bound_mip;
    }
    bool paged(std::uintptr_t id) const {
        std::lock_guard lock(mutex_);
        return targets_.count(id) != 0;
    }

    // The binds to make (the caller submits them in one batch), one op per image: a render target's op binds all its
    // tiles, so of two only the last counts; a streamed texture's ops bind tiles not bound before, so they add up. The
    // merged op keeps `first` from those before it. One op per image keeps the batch free of two binds of one tile,
    // whose order within a batch would be up to the driver.
    std::vector<Op> take() {
        std::lock_guard lock(mutex_);
        std::vector<Op> out;
        std::unordered_map<std::uintptr_t, std::size_t> at;
        for (Op& op : pending_) {
            const auto it = at.find(op.id);
            if (it == at.end()) {
                at.emplace(op.id, out.size());
                out.push_back(std::move(op));
                continue;
            }
            Op& merged = out[it->second];
            const bool first = merged.first || op.first;
            if (op.full) {
                merged = std::move(op);
            } else {
                merged.image_binds.insert(merged.image_binds.end(), op.image_binds.begin(), op.image_binds.end());
                merged.opaque_binds.insert(merged.opaque_binds.end(), op.opaque_binds.begin(), op.opaque_binds.end());
            }
            merged.first = first;
        }
        pending_.clear();
        return out;
    }
    bool pending() const {
        std::lock_guard lock(mutex_);
        return !pending_.empty();
    }
    // Textures bound again since the last call (a newer one over them was released): their bytes must be uploaded.
    std::vector<std::uintptr_t> take_remapped() {
        std::lock_guard lock(mutex_);
        return std::exchange(remapped_, {});
    }
    // Diagnostics: a texture's range, false when it is not on pages.
    bool range(std::uintptr_t id, std::uint64_t& va, std::uint64_t& bytes) const {
        std::lock_guard lock(mutex_);
        const auto it = targets_.find(id);
        if (it == targets_.end()) return false;
        va = it->second.va;
        bytes = it->second.bytes;
        return true;
    }

    struct Stats {
        std::size_t textures = 0, mapped = 0, pages = 0, blocks = 0, local_blocks = 0;
        std::size_t streamed = 0, streamed_unbound = 0;  // streamed textures, those no view has asked a mip of yet
        std::uint64_t placed = 0;
    };
    Stats stats() const {
        std::lock_guard lock(mutex_);
        Stats s;
        s.textures = targets_.size();
        for (const auto& [id, t] : targets_) {
            s.mapped += t.mapped;
            s.streamed += t.streamed;
            s.streamed_unbound += t.streamed && t.bound_mip == kNoMip;
        }
        s.pages = pages_.size() + extra_.size();
        s.blocks = blocks_.size();
        for (const Block& b : blocks_) s.local_blocks += local_type(b.type);
        s.placed = placed_;
        return s;
    }

private:
    struct Slot {
        std::uint32_t block = 0;
        std::uint32_t index = 0;
    };
    struct Page {
        Slot slot;
        std::uint32_t users = 0;
    };
    struct Block {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::uint32_t type = 0;
    };
    // A page of one texture's own (not shared by title address): textures at the same address and size share it.
    using ExtraKey = std::tuple<std::uint64_t, std::uint64_t, std::uint32_t>;  // (va, bytes, tile)
    struct Target {
        VkImage image = VK_NULL_HANDLE;
        std::vector<Tile> tiles;
        std::uint32_t types = 0;
        std::uint64_t va = 0, bytes = 0;
        bool mapped = false;
        bool ever_bound = false;
        bool streamed = false;
        std::uint32_t bound_mip = kNoMip;  // streamed: the most detailed mip bound (its tiles and all smaller ones)
        std::vector<std::uint64_t> own;    // page addresses it holds
        std::vector<ExtraKey> extra;       // (va, bytes, tile) pages it holds
        std::vector<Slot> slots;           // per tile, while mapped (render targets) or bound (streamed)
    };

    static void add_pages(std::vector<Tile>& out, VkDeviceSize offset, VkDeviceSize size, VkSparseMemoryBindFlags flags,
                          std::uint32_t mip) {
        for (VkDeviceSize at = 0; at < size; at += kPage) {
            Tile t;
            t.opaque = true;
            t.resource_offset = offset + at;
            t.flags = flags;
            t.mip = mip;
            out.push_back(t);
        }
    }

    // A free page of a memory type `types` allows, from a new block when none is free.
    bool local_type(std::uint32_t type) const {
        return (memory_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    }
    bool take_slot(std::uint32_t types, Slot& out) {
        // A free page in video memory first, then a new block there (refused past the VRAM budget), then a free page
        // in system memory, then a new block there. Type 0 is system memory on NVIDIA: walking the types in index
        // order put pages in system memory while video memory had free ones.
        const auto take_free = [&](bool local) {
            for (std::uint32_t type = 0; type < memory_.memoryTypeCount; ++type) {
                if (!(types & (1u << type)) || local_type(type) != local) continue;
                auto& free = free_[type];
                if (!free.empty()) {
                    out = free.back();
                    free.pop_back();
                    return true;
                }
            }
            return false;
        };
        for (const bool local : {true, false}) {
            if (take_free(local)) return true;
            for (std::uint32_t type = 0; type < memory_.memoryTypeCount; ++type) {
                if (!(types & (1u << type)) || local_type(type) != local) continue;
                const VkDeviceMemory memory = fn_.allocate(type, kPage * kBlockPages);
                if (!memory) continue;
                const auto block = std::uint32_t(blocks_.size());
                blocks_.push_back({memory, type});
                auto& free = free_[type];
                for (std::uint32_t i = kBlockPages; i-- > 1;) free.push_back({block, i});
                out = {block, 0};
                return true;
            }
        }
        return false;
    }
    bool slot_allowed(const Slot& slot, std::uint32_t types) const {
        return (types & (1u << blocks_[slot.block].type)) != 0;
    }

    // Pages for every tile of `t` (shared by address where others hold them). False, and nothing held, when the pool
    // could not grow.
    bool acquire(Target& t) {
        const std::uint64_t first = (t.va + kPage - 1) / kPage * kPage;
        const std::uint64_t end = t.va + t.bytes;
        const std::size_t own = first < end ? std::size_t((end - first + kPage - 1) / kPage) : 0;
        t.own.clear();
        t.extra.clear();
        t.slots.assign(t.tiles.size(), {});
        for (std::size_t i = 0; i < t.tiles.size(); ++i) {
            Page* page = nullptr;
            if (i < own) {
                const std::uint64_t address = first + i * kPage;
                const auto it = pages_.find(address);
                if (it != pages_.end() && slot_allowed(it->second.slot, t.types)) {
                    page = &it->second;
                } else if (it == pages_.end()) {
                    Slot slot;
                    if (!take_slot(t.types, slot)) return fail(t);
                    page = &pages_.emplace(address, Page{slot, 0}).first->second;
                }
                if (page) {
                    ++page->users;
                    t.own.push_back(address);
                    t.slots[i] = page->slot;
                    continue;
                }
                // A page another texture holds in a memory type this one cannot use: a page of its own.
            }
            if (!take_own(t, std::uint32_t(i), t.slots[i])) return fail(t);
        }
        return true;
    }
    bool fail(Target& t) {
        give_back(t);
        return false;
    }

    void give_back(Target& t) {
        for (const std::uint64_t address : t.own) {
            const auto it = pages_.find(address);
            if (it != pages_.end() && --it->second.users == 0) {
                free_[blocks_[it->second.slot.block].type].push_back(it->second.slot);
                pages_.erase(it);
            }
        }
        for (const auto& key : t.extra) put_own(key);
        t.own.clear();
        t.extra.clear();
        t.slots.clear();
    }

    // A page of `t`'s own for tile `i` (shared with textures at the same address and size), held in t.extra.
    bool take_own(Target& t, std::uint32_t i, Slot& out) {
        const ExtraKey key{t.va, t.bytes, i};
        auto it = extra_.find(key);
        if (it != extra_.end() && !slot_allowed(it->second.slot, t.types)) return false;
        if (it == extra_.end()) {
            Slot slot;
            if (!take_slot(t.types, slot)) return false;
            it = extra_.emplace(key, Page{slot, 0}).first;
        }
        ++it->second.users;
        t.extra.push_back(key);
        out = it->second.slot;
        return true;
    }
    void put_own(const ExtraKey& key) {
        const auto it = extra_.find(key);
        if (it != extra_.end() && --it->second.users == 0) {
            free_[blocks_[it->second.slot.block].type].push_back(it->second.slot);
            extra_.erase(it);
        }
    }

    void queue_bind(std::uintptr_t id, Target& t, bool first) {
        Op op;
        op.id = id;
        op.image = t.image;
        op.first = first && !t.ever_bound;
        t.ever_bound = true;
        for (std::size_t i = 0; i < t.tiles.size(); ++i) {
            const Tile& tile = t.tiles[i];
            const Block& block = blocks_[t.slots[i].block];
            const VkDeviceSize offset = VkDeviceSize(t.slots[i].index) * kPage;
            if (tile.opaque)
                op.opaque_binds.push_back({tile.resource_offset, kPage, block.memory, offset, tile.flags});
            else
                op.image_binds.push_back({tile.subresource, tile.offset, tile.extent, block.memory, offset, 0});
        }
        pending_.push_back(std::move(op));
    }

    VkDevice device_;
    Functions fn_;
    VkPhysicalDeviceMemoryProperties memory_;
    mutable std::mutex mutex_;
    std::vector<Block> blocks_;
    std::vector<Slot> free_[VK_MAX_MEMORY_TYPES];
    std::unordered_map<std::uint64_t, Page> pages_;  // title page address -> its video memory
    std::map<ExtraKey, Page> extra_;
    std::unordered_map<std::uintptr_t, Target> targets_;
    std::uint64_t placed_ = 0;
    std::vector<Op> pending_;
    std::vector<std::uintptr_t> remapped_;
};

}  // namespace leonida
