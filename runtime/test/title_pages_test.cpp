// Offline check of leonida::TitlePages's bookkeeping (no GPU): fake allocation and sparse requirements.
#include "../src/vulkan_title_pages.h"

#include <cstdio>
#include <cstdlib>

using leonida::TitlePages;

static int failures = 0;
#define EXPECT(c)                                                            \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);                 \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static std::uintptr_t next_memory = 0x1000;
static int local_refused = 0;  // device-local blocks to refuse (over the VRAM budget)
static VkDeviceMemory fake_allocate(std::uint32_t type, VkDeviceSize) {
    if (type == 1 && local_refused > 0) {
        --local_refused;
        return VK_NULL_HANDLE;
    }
    return reinterpret_cast<VkDeviceMemory>(next_memory += 0x10);
}
static int freed = 0;
static VKAPI_ATTR void VKAPI_CALL fake_free(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { ++freed; }
// Every image: 64 KiB tiles, 256x256 texel granularity, mip tail of 1 page from the mip where the extent is < 256.
static VKAPI_ATTR void VKAPI_CALL fake_requirements(VkDevice, VkImage, VkMemoryRequirements* r) {
    r->size = 64 * 1024 * 16;
    r->alignment = 64 * 1024;
    r->memoryTypeBits = 0x3;
}
static std::uint32_t tail_first = 2;
static VKAPI_ATTR void VKAPI_CALL fake_sparse(VkDevice, VkImage, std::uint32_t* count,
                                              VkSparseImageMemoryRequirements* out) {
    if (!out) {
        *count = 1;
        return;
    }
    VkSparseImageMemoryRequirements r{};
    r.formatProperties.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.formatProperties.imageGranularity = {256, 256, 1};
    r.imageMipTailFirstLod = tail_first;
    r.imageMipTailSize = 64 * 1024;
    r.imageMipTailOffset = 64 * 1024 * 10;
    r.imageMipTailStride = 64 * 1024;
    out[0] = r;
}

static std::size_t binds(const TitlePages::Op& op) { return op.image_binds.size() + op.opaque_binds.size(); }

int main() {
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryTypeCount = 2;
    memory.memoryTypes[0] = {0, 1};
    memory.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    const TitlePages::Functions fns{fake_allocate, fake_free, fake_requirements, fake_sparse};
    {
        TitlePages pages(reinterpret_cast<VkDevice>(1), fns, memory);
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.extent = {512, 512, 1};  // mip 0: 2x2 tiles, mip 1: 1 tile, mips 2+: tail
        info.mipLevels = 5;
        info.arrayLayers = 1;
        std::vector<TitlePages::Tile> tiles;
        std::uint32_t types = 0;
        EXPECT(pages.layout(reinterpret_cast<VkImage>(10), info, tiles, types));
        EXPECT(tiles.size() == 6);  // 4 + 1 + tail page
        EXPECT(types == 0x3);
        EXPECT(tiles.back().opaque && tiles.back().resource_offset == 64 * 1024 * 10 && tiles.back().mip == 2);
        EXPECT(tiles[0].mip == 0 && tiles[4].mip == 1);

        // Render targets: bound at once, pages shared by title address.
        const std::uint64_t va = 0x200000, bytes = 6 * 64 * 1024;
        EXPECT(pages.place(1, reinterpret_cast<VkImage>(11), tiles, types, va, bytes, false));
        EXPECT(pages.place(2, reinterpret_cast<VkImage>(12), tiles, types, va, bytes, false));
        auto s = pages.stats();
        EXPECT(s.pages == 6);  // shared, not 12
        auto ops = pages.take();
        EXPECT(ops.size() == 2 && ops[0].first && ops[1].first);
        EXPECT(ops[0].image_binds.size() == 5 && ops[0].opaque_binds.size() == 1);
        EXPECT(ops[0].image_binds[0].memory == ops[1].image_binds[0].memory &&
               ops[0].image_binds[0].memoryOffset == ops[1].image_binds[0].memoryOffset);
        // Another range over the same memory: shares the pages it covers, nothing unbound.
        EXPECT(pages.place(3, reinterpret_cast<VkImage>(13), tiles, types, va + 0x10000, bytes, false));
        EXPECT(pages.mapped(1) && pages.mapped(2) && pages.mapped(3));
        ops = pages.take();
        EXPECT(ops.size() == 1 && binds(ops[0]) == 6);
        s = pages.stats();
        EXPECT(s.pages == 7);  // one more page past the old range

        // Streamed textures: nothing bound until a view asks for a mip; pages of their own.
        EXPECT(pages.place(4, reinterpret_cast<VkImage>(14), tiles, types, va, bytes, true));
        EXPECT(pages.take().empty());
        EXPECT(!pages.ready(4));
        s = pages.stats();
        EXPECT(s.streamed == 1 && s.streamed_unbound == 1 && s.pages == 7);
        EXPECT(pages.require_mip(4, 2));  // the tail only
        EXPECT(!pages.ready(4));          // its bind not made yet
        ops = pages.take();
        EXPECT(ops.size() == 1 && ops[0].first && ops[0].image_binds.empty() && ops[0].opaque_binds.size() == 1);
        EXPECT(pages.ready(4) && pages.first_mip(4) == 2);
        EXPECT(pages.take_remapped().size() == 1);
        EXPECT(pages.stats().pages == 8);
        EXPECT(!pages.require_mip(4, 3));  // in the tail already
        // Mip 1 then mip 0 before a flush: one op adding the five tiles, `first` no longer set.
        EXPECT(pages.require_mip(4, 1));
        EXPECT(pages.require_mip(4, 0));
        ops = pages.take();
        EXPECT(ops.size() == 1 && !ops[0].first && ops[0].image_binds.size() == 5 && ops[0].opaque_binds.empty());
        EXPECT(pages.first_mip(4) == 0 && pages.stats().pages == 13);
        // Its own pages: not those of the render target at the same address.
        bool shared = false;
        for (const auto& b : ops[0].image_binds) shared = shared || b.memoryOffset < 7 * 64 * 1024;
        EXPECT(!shared);

        // A second streamed texture at the same address and size shares its pages; one 16 KiB on does not.
        EXPECT(pages.place(5, reinterpret_cast<VkImage>(15), tiles, types, va, bytes, true));
        EXPECT(pages.require_mip(5, 2));
        EXPECT(pages.stats().pages == 13);
        EXPECT(pages.place(6, reinterpret_cast<VkImage>(16), tiles, types, va + 0x4000, bytes, true));
        EXPECT(pages.require_mip(6, 2));
        EXPECT(pages.stats().pages == 14);
        pages.take();

        for (std::uintptr_t id : {1, 2, 3, 4, 5, 6}) pages.release(id);
        s = pages.stats();
        EXPECT(s.textures == 0 && s.pages == 0);

        // Resident mips: a texture placed inside a streamed one's range owns that memory, so the older one keeps only
        // the mips whose console bytes come before it (ends: mip 0 needs 6 pages, mip 1 2, the tail 1).
        const std::vector<std::uint64_t> ends{6 * 64 * 1024, 2 * 64 * 1024, 64 * 1024, 64 * 1024, 64 * 1024};
        EXPECT(pages.place(7, reinterpret_cast<VkImage>(17), tiles, types, 0x4000000, bytes, true, ends));
        EXPECT(pages.require_mip(7, 0));  // its view reads from mip 0
        pages.take();
        pages.take_remapped();
        EXPECT(pages.stats().pages == 6 && pages.first_mip(7) == 0);
        EXPECT(pages.place(8, reinterpret_cast<VkImage>(18), tiles, types, 0x4000000 + 0x10000, bytes, true, ends));
        EXPECT(pages.first_mip(7) == 2);  // only the tail is its own now
        EXPECT(pages.stats().pages == 1);
        ops = pages.take();
        EXPECT(ops.size() == 1 && ops[0].image_binds.size() == 5 && ops[0].image_binds[0].memory == VK_NULL_HANDLE);
        // A view of the newer one; then the newer one goes: the older one's mips come back and are uploaded again.
        EXPECT(pages.require_mip(8, 0));
        pages.take();
        pages.take_remapped();
        pages.release(8);
        EXPECT(pages.first_mip(7) == 0 && pages.stats().pages == 6);
        EXPECT(pages.take_remapped().size() == 1);
        ops = pages.take();
        EXPECT(ops.size() == 1 && ops[0].image_binds.size() == 5 && ops[0].image_binds[0].memory != VK_NULL_HANDLE);
        // Unbind then bind of one tile before a flush: one bind per tile, the last.
        EXPECT(pages.place(9, reinterpret_cast<VkImage>(19), tiles, types, 0x4000000 + 0x10000, bytes, true, ends));
        pages.release(9);
        ops = pages.take();
        EXPECT(ops.size() == 1 && ops[0].image_binds.size() == 5 && ops[0].image_binds[0].memory != VK_NULL_HANDLE);
        pages.release(7);
        EXPECT(pages.stats().pages == 0);

        // A texture the GPU copies into: every mip bound at once, and a texture placed inside it later trims nothing.
        EXPECT(pages.place(10, reinterpret_cast<VkImage>(20), tiles, types, 0x5000000, bytes, true, ends));
        EXPECT(pages.require_all(10));
        EXPECT(pages.first_mip(10) == 0 && pages.stats().pages == 6);
        EXPECT(!pages.require_all(10));  // once
        EXPECT(pages.place(11, reinterpret_cast<VkImage>(21), tiles, types, 0x5000000 + 0x10000, bytes, true, ends));
        EXPECT(pages.first_mip(10) == 0 && pages.stats().pages == 6);
        pages.take();
        pages.release(11);
        pages.release(10);
        EXPECT(pages.stats().pages == 0);

        // The page budget: within it every mip; past it a new streamed texture leaves its mip 0 (4 tiles) out, and one
        // bound already keeps what it has.
        pages.set_budget(2 * 64 * 1024);
        EXPECT(pages.place(12, reinterpret_cast<VkImage>(22), tiles, types, 0x6000000, bytes, true));
        EXPECT(pages.require_mip(12, 0));
        EXPECT(pages.first_mip(12) == 0 && pages.stats().pages == 6);  // within budget when it was bound
        EXPECT(pages.place(13, reinterpret_cast<VkImage>(23), tiles, types, 0x7000000, bytes, true));
        EXPECT(pages.require_mip(13, 0));
        EXPECT(pages.first_mip(13) == 1 && pages.stats().pages == 8);  // mip 1 (1 tile) and the tail page
        pages.take();
        pages.release(12);
        pages.release(13);
        pages.set_budget(0);
        EXPECT(pages.stats().pages == 0);
    }
    {
        // Past the VRAM budget a block comes from system memory; once video memory gives blocks again, they come first
        // (not the free pages of the system-memory block).
        TitlePages pages(reinterpret_cast<VkDevice>(1), fns, memory);
        std::vector<TitlePages::Tile> one(1);
        one[0].opaque = true;
        local_refused = 1;
        EXPECT(pages.place(1, reinterpret_cast<VkImage>(21), one, 0x3, 0x10000, 0x10000, false));
        auto s = pages.stats();
        EXPECT(s.blocks == 1 && s.local_blocks == 0);
        EXPECT(pages.place(2, reinterpret_cast<VkImage>(22), one, 0x3, 0x40000, 0x10000, false));
        s = pages.stats();
        EXPECT(s.blocks == 2 && s.local_blocks == 1);
        // Only system memory allowed: its free pages.
        EXPECT(pages.place(3, reinterpret_cast<VkImage>(23), one, 0x1, 0x80000, 0x10000, false));
        s = pages.stats();
        EXPECT(s.blocks == 2);
    }
    EXPECT(freed > 0);
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
