#pragma once

// Device-lost diagnostics for the D3D12 layer: auto-breadcrumbs per command list (DRED-style), off unless
// LEONIDA_DRED=1 (per list) or 2 (also per operation). With VK_AMD_buffer_marker the GPU records its progress; without
// it the ring keeps only what the CPU knows (which lists each queue submitted) and every state is `unknown`.
//
// Each command list of an ExecuteCommandLists gets a slot of four 32-bit words (begin, end, progress, padding) in a
// host-visible, coherent buffer owned by this ring (nothing the replay uses). Around the list's commands the backend
// records
//   vkCmdWriteBufferMarkerAMD(TOP_OF_PIPE,    slot.begin, tag)   the GPU reached the list's first command
//   vkCmdWriteBufferMarkerAMD(BOTTOM_OF_PIPE, slot.end,   tag)   every command of the list completed
// `tag` is unique per submitted list (never 0) and is what the GPU writes, so the CPU never clears a slot: a word
// counts only when it holds this list's own tag, and a late write from an older list that used the slot before can
// never pass for progress of the current one. At level 2 the begin marker also clears the progress word on the GPU,
// and emit() writes the op numbers into it (breadcrumb_ops_completed; slot reuse in mark()).
//
// States, from what is known and nothing more:
//   recorded   the markers are in a command buffer, which was not (yet) submitted: such entries are not kept
//   submitted  vkQueueSubmit succeeded (CPU knowledge); the GPU has written neither word
//   started    the begin word holds the tag: the GPU front end reached the list
//   completed  the end word holds the tag: all of the list's commands finished
// After a device loss no memory dependency can make the GPU's writes visible any more; what the mapped words hold
// then is the best information there is (as with DRED on D3D12), and the report says so.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include "command_list.h"
#include "debug_targets.h"  // arena_mapped: the stuck dispatch's descriptors in a hang report
#include "vulkan_memory.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace leonida {

// The D3D12_AUTO_BREADCRUMB_OP values (d3d12.h, Windows SDK 10.0.26100) of the operations the layer records.
namespace breadcrumb_op {
inline constexpr std::uint32_t set_marker = 0, begin_event = 1, end_event = 2, draw_instanced = 3,
                               draw_indexed_instanced = 4, execute_indirect = 5, dispatch = 6,
                               copy_buffer_region = 7, copy_texture_region = 8, copy_resource = 9,
                               resolve_subresource = 11, clear_render_target_view = 12,
                               clear_unordered_access_view = 13, clear_depth_stencil_view = 14,
                               resource_barrier = 15, resolve_query_data = 18, atomic_copy_buffer_uint = 23,
                               resolve_subresource_region = 25, write_buffer_immediate = 26,
                               build_raytracing_acceleration_structure = 31, barrier = 45;
}  // namespace breadcrumb_op

// The D3D12 breadcrumb operation a recorded command stands for, or false for a command that is not a D3D12
// operation of its own (a predication change, a query begin or end, which DRED does not list either).
inline bool breadcrumb_op_of(const Command& command, std::uint32_t& op) {
    namespace o = breadcrumb_op;
    return std::visit(
        [&](const auto& c) {
            using T = std::decay_t<decltype(c)>;
            bool known = true;
            if constexpr (std::is_same_v<T, TransitionBarrier> || std::is_same_v<T, UavBarrier> ||
                          std::is_same_v<T, AliasingBarrier>) {
                op = o::resource_barrier;
            } else if constexpr (std::is_same_v<T, GlobalBarrier> || std::is_same_v<T, BufferBarrier> ||
                                 std::is_same_v<T, TextureBarrier>) {
                op = o::barrier;  // ID3D12GraphicsCommandList7::Barrier
            } else if constexpr (std::is_same_v<T, ClearRenderTarget>) {
                op = o::clear_render_target_view;
            } else if constexpr (std::is_same_v<T, ClearDepthStencil>) {
                op = o::clear_depth_stencil_view;
            } else if constexpr (std::is_same_v<T, ClearUnorderedAccess>) {
                op = o::clear_unordered_access_view;
            } else if constexpr (std::is_same_v<T, CopyBuffer>) {
                op = o::copy_buffer_region;
            } else if constexpr (std::is_same_v<T, CopyTexture>) {
                op = o::copy_texture_region;
            } else if constexpr (std::is_same_v<T, CopyResource>) {
                op = o::copy_resource;
            } else if constexpr (std::is_same_v<T, ResolveSubresource>) {
                op = c.region ? o::resolve_subresource_region : o::resolve_subresource;
            } else if constexpr (std::is_same_v<T, WriteImmediate>) {
                op = o::write_buffer_immediate;
            } else if constexpr (std::is_same_v<T, AtomicCopyBuffer>) {
                op = o::atomic_copy_buffer_uint;
            } else if constexpr (std::is_same_v<T, ResolveQueries>) {
                op = o::resolve_query_data;
            } else if constexpr (std::is_same_v<T, DebugMarker>) {
                op = c.kind == DebugMarker::Kind::begin ? o::begin_event
                     : c.kind == DebugMarker::Kind::end ? o::end_event
                                                        : o::set_marker;
            } else if constexpr (std::is_same_v<T, ExecuteIndirect>) {
                op = o::execute_indirect;
            } else if constexpr (std::is_same_v<T, BuildAccelerationStructure>) {
                op = o::build_raytracing_acceleration_structure;
            } else if constexpr (std::is_same_v<T, Dispatch>) {
                op = o::dispatch;
            } else if constexpr (std::is_same_v<T, Draw>) {
                op = c.indexed ? o::draw_indexed_instanced : o::draw_instanced;
            } else {
                known = false;  // Predication, BeginQuery, EndQuery, DiscardResource: no breadcrumb of their own
            }
            return known;
        },
        command);
}

inline const char* breadcrumb_op_name(std::uint32_t op) {
    namespace o = breadcrumb_op;
    switch (op) {
        case o::set_marker: return "SetMarker";
        case o::begin_event: return "BeginEvent";
        case o::end_event: return "EndEvent";
        case o::draw_instanced: return "DrawInstanced";
        case o::draw_indexed_instanced: return "DrawIndexedInstanced";
        case o::execute_indirect: return "ExecuteIndirect";
        case o::dispatch: return "Dispatch";
        case o::copy_buffer_region: return "CopyBufferRegion";
        case o::copy_texture_region: return "CopyTextureRegion";
        case o::copy_resource: return "CopyResource";
        case o::resolve_subresource: return "ResolveSubresource";
        case o::resolve_subresource_region: return "ResolveSubresourceRegion";
        case o::clear_render_target_view: return "ClearRenderTargetView";
        case o::clear_unordered_access_view: return "ClearUnorderedAccessView";
        case o::clear_depth_stencil_view: return "ClearDepthStencilView";
        case o::resource_barrier: return "ResourceBarrier";
        case o::resolve_query_data: return "ResolveQueryData";
        case o::atomic_copy_buffer_uint: return "AtomicCopyBufferUINT";
        case o::write_buffer_immediate: return "WriteBufferImmediate";
        case o::build_raytracing_acceleration_structure: return "BuildRaytracingAccelerationStructure";
        case o::barrier: return "Barrier";
        default: return "?";
    }
}

// `unknown`: the ring runs without VK_AMD_buffer_marker (CPU only) and knows the list was submitted, nothing about
// the GPU's progress.
enum class BreadcrumbState : std::uint8_t { submitted, started, completed, unknown };

// What the two words of a slot say about the list tagged `tag` (see the top of this file).
inline BreadcrumbState breadcrumb_state(std::uint32_t begin_word, std::uint32_t end_word, std::uint32_t tag) {
    if (end_word == tag) return BreadcrumbState::completed;
    if (begin_word == tag) return BreadcrumbState::started;
    return BreadcrumbState::submitted;
}

// Per-operation progress (PoC): after its k-th D3D12 operation (0-based) a list's stream writes `first + k` into its
// slot's progress word, at BOTTOM_OF_PIPE. Op numbers are unique per list and never 0, so a word from the slot's
// previous owner (an older range) or the initial 0 falls outside [first, first + count) and counts as nothing.
// Unsigned arithmetic, so correct across the 32-bit wrap. A slot passes to a new owner only once its previous one
// finished, and the new owner's begin marker clears the word (BreadcrumbRing::mark, record_begin).
inline std::uint32_t breadcrumb_ops_completed(std::uint32_t word, std::uint32_t first, std::size_t count) {
    const std::uint32_t k = word - first;
    return k < count ? k + 1 : 0;
}

inline const char* breadcrumb_state_name(BreadcrumbState state) {
    switch (state) {
        case BreadcrumbState::completed: return "completed";
        case BreadcrumbState::started: return "started (not completed)";
        case BreadcrumbState::submitted: return "submitted (not started)";
        case BreadcrumbState::unknown: break;
    }
    return "submitted, GPU progress UNKNOWN";
}

// A debug name (UTF-16 on Windows, UTF-32 elsewhere) as UTF-8 for the report; unpaired surrogates become U+FFFD.
inline std::string breadcrumb_utf8(const std::wstring& name) {
    std::string out;
    for (std::size_t i = 0; i < name.size(); ++i) {
        std::uint32_t c = static_cast<std::uint32_t>(name[i]);
        if constexpr (sizeof(wchar_t) == 2) {
            if (c >= 0xd800 && c < 0xdc00 && i + 1 < name.size()) {
                const std::uint32_t low = static_cast<std::uint32_t>(name[i + 1]);
                if (low >= 0xdc00 && low < 0xe000) {
                    c = 0x10000 + ((c - 0xd800) << 10) + (low - 0xdc00);
                    ++i;
                }
            }
        }
        if ((c >= 0xd800 && c < 0xe000) || c > 0x10ffff) c = 0xfffd;
        if (c < 0x80) {
            out += static_cast<char>(c);
        } else if (c < 0x800) {
            out += static_cast<char>(0xc0 | (c >> 6));
            out += static_cast<char>(0x80 | (c & 0x3f));
        } else if (c < 0x10000) {
            out += static_cast<char>(0xe0 | (c >> 12));
            out += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (c & 0x3f));
        } else {
            out += static_cast<char>(0xf0 | (c >> 18));
            out += static_cast<char>(0x80 | ((c >> 12) & 0x3f));
            out += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
            out += static_cast<char>(0x80 | (c & 0x3f));
        }
    }
    return out;
}

// The shader names of pipeline state objects, by the opaque id commands carry (the PSO object), filled by the device
// while breadcrumbs are on: a list stuck on the GPU then names the shader of the operation it stopped in.
inline std::mutex& breadcrumb_pso_names_mutex() {
    static std::mutex m;
    return m;
}
inline std::unordered_map<std::uintptr_t, std::string>& breadcrumb_pso_names() {
    static std::unordered_map<std::uintptr_t, std::string> names;
    return names;
}
inline void note_breadcrumb_pso_name(std::uintptr_t id, std::string name) {
    std::lock_guard lock(breadcrumb_pso_names_mutex());
    breadcrumb_pso_names()[id] = std::move(name);
}
inline std::string breadcrumb_pso_name(std::uintptr_t id) {
    std::lock_guard lock(breadcrumb_pso_names_mutex());
    const auto it = breadcrumb_pso_names().find(id);
    return it == breadcrumb_pso_names().end() ? std::string() : it->second;
}

struct BreadcrumbFunctions {
    PFN_vkCreateBuffer create_buffer = nullptr;
    PFN_vkDestroyBuffer destroy_buffer = nullptr;
    PFN_vkGetBufferMemoryRequirements requirements = nullptr;
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    PFN_vkBindBufferMemory bind = nullptr;
    PFN_vkMapMemory map = nullptr;
    PFN_vkUnmapMemory unmap = nullptr;
    PFN_vkCmdWriteBufferMarkerAMD write_marker = nullptr;
};

class BreadcrumbRing {
public:
    // A list's place in the ring between recording its markers and submitting them.
    struct Mark {
        std::uint32_t slot = 0;
        std::uint32_t tag = 0;
        std::uint64_t list_id = 0;
        std::uint64_t queue_id = 0;  // the backend queue that submitted it, as an opaque number
        std::vector<std::uint32_t> ops;
        std::uint64_t sequence = 0;  // the order mark() handed it out in; does not wrap
        std::wstring list_name, queue_name;  // copies of the D3D12 debug names at submission, when given
        std::uint32_t first_op = 0;           // the op number of ops[0] (breadcrumb_ops_completed)
        bool progress_safe = true;            // level 2: its slot's earlier owners can write nothing more into it
        std::vector<std::uint32_t> positions;  // the command index of each op, for the per-op markers
        std::vector<std::uintptr_t> pipelines;  // the PSO id of each draw or dispatch op (0 for the others)
        // A dispatch's group counts and its first four user-data words (0 for the other ops): a hang report then
        // says whether a dispatch was huge or what it was pointed at.
        std::vector<std::array<std::uint32_t, 11>> details;
        // For a dispatch whose user-data words point into the GPU address arena: the 4 dwords at each such word at
        // submission (op index, 8 x 4 dwords; 0 for a word that is not such an address). The report shows them
        // beside what is there at report time, which later writes may have changed (a ring the title reuses).
        std::vector<std::pair<std::uint32_t, std::array<std::uint32_t, 32>>> submitted_words;
        // An acceleration-structure build's inputs (op index, text): which build a hang stopped in (run 03:27 stopped
        // in op 419 of a list of builds).
        std::vector<std::pair<std::uint32_t, std::string>> notes;
    };
    // A submitted list as the breadcrumbs see it now.
    struct Entry {
        std::uint32_t tag = 0;
        std::uint64_t list_id = 0;  // the CommandList the ops were recorded into, as an opaque number
        std::vector<std::uint32_t> ops;
        BreadcrumbState state = BreadcrumbState::submitted;
        std::uint64_t queue_id = 0;
        std::wstring list_name, queue_name;
        std::vector<std::uintptr_t> pipelines;  // as Mark::pipelines
        std::vector<std::array<std::uint32_t, 11>> details;  // as Mark::details
        std::vector<std::pair<std::uint32_t, std::array<std::uint32_t, 32>>> submitted_words;  // as Mark's
        std::vector<std::pair<std::uint32_t, std::string>> notes;                              // as Mark's
        bool progress_known = false;      // GPU breadcrumbs: completed_ops is what the progress word says
        bool no_slot = false;             // level 2 with every slot busy: no GPU marker, progress unknown
        std::uint32_t completed_ops = 0;  // operations finished, in order (D3D12's *pLastBreadcrumbValue)
    };

    static constexpr std::uint32_t kSlots = 1024;  // lists kept: the newest 1024 submitted
    static constexpr std::uint32_t kNoSlot = ~0u;  // Mark::slot of a list that got none (CPU history only)
    static constexpr std::size_t kNoSlotKept = 64;  // such lists kept besides the slots' own: the newest 64
    static constexpr std::size_t kUnknownShown = 16;  // lists the report shows when GPU progress is unknown

    BreadcrumbRing() = default;
    BreadcrumbRing(const BreadcrumbRing&) = delete;
    BreadcrumbRing& operator=(const BreadcrumbRing&) = delete;
    ~BreadcrumbRing() {
        if (mapped_) fn_.unmap(device_, memory_);
        if (buffer_) fn_.destroy_buffer(device_, buffer_, nullptr);
        if (memory_) fn_.free(device_, memory_, nullptr);
    }

    // GPU breadcrumbs. False (whatever was made is freed with the ring) when the marker command is missing or no
    // host-visible coherent memory fits; the caller may fall back to initialize_cpu_only().
    // `per_op`: a progress marker after every D3D12 operation (BUILD-1655), or only the two around each list.
    bool initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory, const BreadcrumbFunctions& fn,
                    bool per_op = true) {
        if (!fn.write_marker || !fn.create_buffer || !fn.map) return false;
        per_op_ = per_op;
        device_ = device;
        fn_ = fn;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = VkDeviceSize(kSlots) * kWords * sizeof(std::uint32_t);
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;  // the usage buffer markers need
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (fn_.create_buffer(device_, &info, nullptr, &buffer_) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        fn_.requirements(device_, buffer_, &requirements);
        constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (const std::uint32_t type :
             memory_type_order(memory, requirements.memoryTypeBits, host, 0, requirements.size)) {
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = type;
            if (fn_.allocate(device_, &allocation, nullptr, &memory_) == VK_SUCCESS) break;
            memory_ = VK_NULL_HANDLE;
        }
        if (!memory_ || fn_.bind(device_, buffer_, memory_, 0) != VK_SUCCESS ||
            fn_.map(device_, memory_, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&mapped_)) != VK_SUCCESS) {
            mapped_ = nullptr;
            return false;
        }
        for (std::uint32_t i = 0; i < kSlots * kWords; ++i) mapped_[i] = 0;  // no tag or op number is 0
        gpu_ = true;
        return true;
    }

    // No GPU writes and no Vulkan objects: the ring keeps what the CPU knows (which lists each queue submitted),
    // and every entry's GPU progress is unknown. For a device without VK_AMD_buffer_marker.
    void initialize_cpu_only() { gpu_ = false; }

    bool gpu() const { return gpu_; }

    // The D3D12 debug names the next mark() of `list` (or of a list on `queue`) carries; an empty name forgets it.
    // Called before every submission with the names as they are then, so an address a destroyed object left behind
    // never lends its old name to a new one. Names are copies; no D3D12 object is referenced.
    void label_list(const CommandList* list, std::wstring name) { label(list_names_, list, std::move(name)); }
    void label_queue(const void* queue, std::wstring name) { label(queue_names_, queue, std::move(name)); }

    // The debug names the ring holds for `list` on `queue` now, as copies. A batch held on the CPU takes them when
    // it is held, while the application's objects are alive; by the time it is released, their addresses may
    // belong to other objects.
    struct Names {
        std::wstring list, queue;
    };
    Names names_of(const CommandList* list, const void* queue) const {
        std::lock_guard lock(mutex_);
        Names names;
        if (const auto it = list_names_.find(list); it != list_names_.end()) names.list = it->second;
        if (const auto it = queue_names_.find(queue); it != queue_names_.end()) names.queue = it->second;
        return names;
    }

    // Takes the next slot and tag for `list`, submitted by `queue`, and notes its D3D12 operations. For a batch held
    // on the CPU and released later, `list` is the held copy, `identity` the list the application passed (its
    // address is the list id; it is only stored as a number, never dereferenced or looked up) and `held` the names
    // taken when it was held.
    Mark mark(const CommandList& list, const void* queue = nullptr, const CommandList* identity = nullptr,
              const Names* held = nullptr) {
        if (!identity) identity = &list;
        Mark m;
        std::uint32_t op = 0;
        const auto commands = list.commands();
        for (std::size_t i = 0; i < commands.size(); ++i)
            if (breadcrumb_op_of(commands[i], op)) {
                m.ops.push_back(op);
                m.positions.push_back(static_cast<std::uint32_t>(i));
                std::uintptr_t pso = 0;
                std::array<std::uint32_t, 11> detail{};
                if (const auto* d = std::get_if<Dispatch>(&commands[i])) {
                    pso = d->pipeline;
                    detail = {d->x, d->y, d->z, d->user_data[0], d->user_data[1], d->user_data[2], d->user_data[3],
                              d->user_data[4], d->user_data[5], d->user_data[6], d->user_data[7]};
                    if (debug_targets::arena_mapped) {
                        std::array<std::uint32_t, 32> words{};
                        bool any = false;
                        for (int w = 0; w < 8; ++w)
                            if (arena_word(d->user_data[w])) {
                                const auto* v = reinterpret_cast<const std::uint32_t*>(debug_targets::arena_mapped +
                                                                                       d->user_data[w]);
                                for (int k = 0; k < 4; ++k) words[w * 4 + k] = v[k];
                                any = true;
                            }
                        if (any) m.submitted_words.emplace_back(static_cast<std::uint32_t>(m.ops.size() - 1), words);
                    }
                }
                m.details.push_back(detail);
                if (const auto* b = std::get_if<BuildAccelerationStructure>(&commands[i])) {
                    std::uint64_t primitives = 0;
                    for (const auto& t : b->triangles)
                        primitives += (t.index_type ? t.index_count : t.vertex_count) / 3;
                    char text[384];
                    if (b->is_tlas)
                        std::snprintf(text, sizeof(text), "TLAS%s, %u instances%s at %#llx, flags %#x, dst %#llx, scratch %#llx",
                                      b->update ? " update" : "", b->instance_count,
                                      b->instance_pointers ? " (pointers)" : "",
                                      static_cast<unsigned long long>(b->instances_va), b->build_flags,
                                      static_cast<unsigned long long>(b->dst_va),
                                      static_cast<unsigned long long>(b->scratch_va));
                    else {
                        const RtTriangleGeometry* g = b->triangles.empty() ? nullptr : &b->triangles.front();
                        std::snprintf(text, sizeof(text),
                                      "BLAS%s, %zu geometries, %llu triangles, flags %#x, dst %#llx, scratch %#llx; "
                                      "first: vertices %#llx (%u, stride %u, format %u), indices %#llx (%u, type %u)",
                                      b->update ? " update" : "", b->triangles.size(),
                                      static_cast<unsigned long long>(primitives), b->build_flags,
                                      static_cast<unsigned long long>(b->dst_va),
                                      static_cast<unsigned long long>(b->scratch_va),
                                      static_cast<unsigned long long>(g ? g->vertex_va : 0), g ? g->vertex_count : 0,
                                      g ? g->vertex_stride : 0, g ? g->vertex_format : 0,
                                      static_cast<unsigned long long>(g ? g->index_va : 0), g ? g->index_count : 0,
                                      g ? g->index_type : 0);
                    }
                    m.notes.emplace_back(static_cast<std::uint32_t>(m.ops.size() - 1), text);
                }
                if (const auto* g = std::get_if<Draw>(&commands[i])) pso = g->pipeline;
                else if (const auto* x = std::get_if<ExecuteIndirect>(&commands[i]))
                    pso = x->pipeline ? x->pipeline : (x->state ? x->state->pipeline : 0);
                m.pipelines.push_back(pso);
            }
        {
            std::lock_guard lock(mutex_);
            // A range of op numbers that does not contain 0 ("never written").
            const auto count = static_cast<std::uint32_t>(m.ops.size());
            if (next_op_ + 1 == 0 || next_op_ + count < next_op_) next_op_ = 0;
            m.first_op = next_op_ + 1;
            next_op_ += count;
            if (++next_tag_ == 0) ++next_tag_;  // 0 means "never written"
            m.tag = next_tag_;
            m.sequence = ++next_sequence_;
            m.slot = next_slot_;
            if (gpu_ && per_op_) {
                // A slot is taken over only once its owner wrote its end word (or the slot is free): a finished list
                // writes nothing more, so no older list's write can land in this list's words, however many op
                // numbers were handed out since. This assumes the host sees a list's op markers no later than its
                // end marker; the spec orders them on the queue but promises the host nothing (docs section 4.11). If every slot still holds a list the GPU has not finished, this
                // list takes none (review of #5314): no GPU marker at all, the owners stay, and it is kept in the
                // history with its GPU progress unknown.
                m.slot = kNoSlot;
                m.progress_safe = false;
                for (std::uint32_t i = 0; i < kSlots; ++i) {
                    const std::uint32_t slot = (next_slot_ + i) % kSlots;
                    if (owner_[slot] == 0 || read(slot, 1) == owner_[slot]) {
                        m.slot = slot;
                        m.progress_safe = true;
                        break;
                    }
                }
            }
            if (m.slot != kNoSlot) next_slot_ = (m.slot + 1) % kSlots;
            // The slot is about to carry this list's tag: the list that used it before can no longer be read.
            // Queues submit in their own order, so the history is not sorted by slot; snapshot() skips any
            // entry whose slot has a newer owner.
            if (m.slot != kNoSlot) owner_[m.slot] = m.tag;
            if (held) {
                m.list_name = held->list;
                m.queue_name = held->queue;
            } else {
                if (const auto it = list_names_.find(&list); it != list_names_.end()) m.list_name = it->second;
                if (const auto it = queue_names_.find(queue); it != queue_names_.end()) m.queue_name = it->second;
            }
        }
        m.list_id = reinterpret_cast<std::uintptr_t>(identity);
        m.queue_id = reinterpret_cast<std::uintptr_t>(queue);
        return m;
    }

    // Nothing is recorded without GPU breadcrumbs, or for a list that got no slot. With per-op markers the begin
    // marker also clears the progress word, so the previous owner's last op number does not sit there.
    void record_begin(VkCommandBuffer command_buffer, const Mark& m) const {
        if (!gpu_ || m.slot == kNoSlot) return;
        fn_.write_marker(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, buffer_, offset(m.slot, 0), m.tag);
        if (op_markers_for(m))
            fn_.write_marker(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, buffer_, offset(m.slot, 2), 0);
    }
    // For emit(): where and how the per-op markers of `m` are written; write == nullptr without GPU breadcrumbs.
    PFN_vkCmdWriteBufferMarkerAMD op_marker_function() const { return gpu_ && per_op_ ? fn_.write_marker : nullptr; }
    // Whether `m` gets per-op markers: level 2, and a slot no earlier list can still write into.
    bool op_markers_for(const Mark& m) const { return gpu_ && per_op_ && m.progress_safe; }
    bool per_op() const { return gpu_ && per_op_; }

    // For tests: the next list's op numbers start after `last`, so a test can cross the 32-bit wrap without
    // submitting four billion operations.
    void skip_op_numbers_to(std::uint32_t last) {
        std::lock_guard lock(mutex_);
        next_op_ = last;
    }
    // For tests: what the history keeps per operation, in 32-bit words (the ops; the per-op command positions are
    // dropped when a list is submitted).
    std::size_t history_words() const {
        std::lock_guard lock(mutex_);
        std::size_t words = 0;
        for (const auto& m : history_) words += m.ops.size() + m.positions.capacity();
        return words;
    }
    // The kept lists' dispatches whose descriptor tables (the user-data words that are arena addresses, 0x80 bytes
    // each, read now) hold a V# whose range reaches [lo, hi): which GPU work may have written title memory there (the
    // transient-pool watch, d3d12_vulkan_device.cpp, after five-star-recomp's --watch-transient). Newest first, at most
    // `limit` lines. A table rewritten since its dispatch shows what it holds now.
    void report_reaching(std::FILE* out, std::uint64_t lo, std::uint64_t hi, int limit = 40) const {
        std::lock_guard lock(mutex_);
        int shown = 0;
        for (auto m = history_.rbegin(); m != history_.rend() && shown < limit; ++m) {
            const bool words = gpu_ && m->slot != kNoSlot && owner_[m->slot] == m->tag;
            const BreadcrumbState state =
                words ? breadcrumb_state(read(m->slot, 0), read(m->slot, 1), m->tag) : BreadcrumbState::unknown;
            for (std::size_t k = 0; k < m->details.size() && shown < limit; ++k) {
                const auto& d = m->details[k];
                if (!(d[0] | d[1] | d[2])) continue;  // not a dispatch
                for (int w = 3; w < 11 && shown < limit; ++w) {
                    if (!arena_word(d[w])) continue;
                    std::uint32_t t[32];
                    if (!debug_targets::read_gpu_va(d[w], t, sizeof(t))) continue;
                    for (int q = 0; q < 8; ++q) {
                        const std::uint32_t* v = t + 4 * q;
                        const std::uint64_t base = v[0] | (std::uint64_t(v[1] & 0xFFFF) << 32);
                        const std::uint32_t stride = (v[1] >> 16) & 0x3FFF;
                        const std::uint64_t extent = std::uint64_t(v[2]) * (stride ? stride : 1);
                        if (!base || !v[2] || base >= hi || base + extent <= lo) continue;
                        const std::uintptr_t pso = k < m->pipelines.size() ? m->pipelines[k] : 0;
                        const std::string name = pso ? breadcrumb_pso_name(pso) : std::string();
                        std::fprintf(out,
                                     "[rt] transient watch:   tag %u op %zu (%s) %s groups %u x %u x %u: user data %d "
                                     "table +%02x V# base %#llx stride %u records %u\n",
                                     m->tag, k, breadcrumb_state_name(state), name.empty() ? "(unnamed)" : name.c_str(),
                                     d[0], d[1], d[2], w - 3, q * 16, static_cast<unsigned long long>(base), stride,
                                     v[2]);
                        ++shown;
                    }
                }
            }
        }
        if (!shown) std::fprintf(out, "[rt] transient watch:   no kept dispatch's tables reach it\n");
    }

    VkBuffer marker_buffer() const { return buffer_; }
    VkDeviceSize progress_offset(const Mark& m) const { return offset(m.slot, 2); }

    void record_end(VkCommandBuffer command_buffer, const Mark& m) const {
        if (gpu_ && m.slot != kNoSlot)
            fn_.write_marker(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, buffer_, offset(m.slot, 1), m.tag);
    }

    // After vkQueueSubmit: keeps the marks when the submission succeeded, forgets them when it did not.
    void submitted(std::vector<Mark>&& marks, bool ok) {
        std::lock_guard lock(mutex_);
        if (!ok) {
            // Never ran, so it writes nothing: its slots are free again.
            for (const auto& m : marks)
                if (m.slot != kNoSlot && owner_[m.slot] == m.tag) owner_[m.slot] = 0;
            return;
        }
        for (auto& m : marks) {
            std::vector<std::uint32_t>().swap(m.positions);  // only emit() needs them: the history keeps ops alone
            history_.push_back(std::move(m));
            if (history_.size() > kSlots + kNoSlotKept) {
                // The oldest list that no longer holds a slot goes (one whose slot passed on, or one that got none);
                // a list that still holds its slot, possibly one the GPU never finished, stays. At most kSlots lists
                // hold a slot, so there is always one to drop, and the newest lists without a slot are kept.
                auto drop = history_.begin();
                while (drop != history_.end() && drop->slot != kNoSlot && owner_[drop->slot] == drop->tag) ++drop;
                history_.erase(drop == history_.end() ? history_.begin() : drop);
            }
        }
    }

    // The submitted lists, oldest first, with what their slots say now (unknown without GPU breadcrumbs). Queues
    // call submitted() in their own order, so the history is sorted by when each list was marked.
    std::vector<Entry> snapshot() const {
        std::lock_guard lock(mutex_);
        std::vector<const Mark*> kept;
        kept.reserve(history_.size());
        for (const auto& m : history_)  // a list whose slot was handed to a later one is gone
            if (m.slot == kNoSlot || owner_[m.slot] == m.tag) kept.push_back(&m);
        std::sort(kept.begin(), kept.end(), [](const Mark* a, const Mark* b) { return a->sequence < b->sequence; });
        std::vector<Entry> out;
        out.reserve(kept.size());
        for (const Mark* m : kept) {
            const bool words = gpu_ && m->slot != kNoSlot;
            const BreadcrumbState state =
                words ? breadcrumb_state(read(m->slot, 0), read(m->slot, 1), m->tag) : BreadcrumbState::unknown;
            Entry e{m->tag, m->list_id, m->ops, state, m->queue_id, m->list_name, m->queue_name, m->pipelines};
            e.details = m->details;
            e.submitted_words = m->submitted_words;
            e.notes = m->notes;
            e.no_slot = gpu_ && m->slot == kNoSlot;
            if (state == BreadcrumbState::completed) {
                e.progress_known = true;  // the end word: every operation finished
                e.completed_ops = static_cast<std::uint32_t>(m->ops.size());
            } else if (gpu_ && per_op_ && m->progress_safe && m->slot != kNoSlot) {
                // Only a word in this list's own range counts; anything else reads as no operation finished, never
                // as more. A finished operation also proves the list started, should its begin word not show yet.
                e.progress_known = true;
                e.completed_ops = breadcrumb_ops_completed(read(m->slot, 2), m->first_op, m->ops.size());
                if (e.completed_ops && e.state == BreadcrumbState::submitted) e.state = BreadcrumbState::started;
            }
            out.push_back(std::move(e));
        }
        return out;
    }

    // For a device-lost message: with GPU breadcrumbs the lists that did not complete and the last one that did;
    // without them the last kUnknownShown lists submitted, their progress unknown.
    void report(std::FILE* out) const { write_report(out, snapshot()); }

    static void write_report(std::FILE* out, const std::vector<Entry>& entries) {
        if (entries.empty()) {
            std::fprintf(out, "[rt] d3d12 DRED: no command list was submitted since the diagnostics started\n");
            return;
        }
        std::size_t first = entries.size();
        // CPU only (no marker extension): every entry is unknown. A GPU ring's lists that got no slot are unknown too,
        // but they are reported among the others.
        if (entries.back().state == BreadcrumbState::unknown && !entries.back().no_slot) {
            first = entries.size() > kUnknownShown ? entries.size() - kUnknownShown : 0;
            std::fprintf(out, "[rt] d3d12 DRED: GPU progress UNKNOWN (no VK_AMD_buffer_marker); the last %zu of the "
                              "%zu lists submitted, oldest first (CPU knowledge only):\n",
                         entries.size() - first, entries.size());
            for (std::size_t i = first; i < entries.size(); ++i) write_entry(out, entries[i]);
            return;
        }
        for (std::size_t i = 0; i < entries.size(); ++i)
            if (entries[i].state != BreadcrumbState::completed) {
                first = i;
                break;
            }
        if (first == entries.size()) {
            std::fprintf(out, "[rt] d3d12 DRED: every one of the %zu lists kept completed\n", entries.size());
            return;
        }
        if (first > 0) first -= 1;  // the last completed list before the first that did not, for context
        std::size_t untracked = 0;
        for (const auto& e : entries) untracked += e.no_slot;
        std::fprintf(out, "[rt] d3d12 DRED: %zu of the last %zu lists did not complete (GPU writes after a device "
                          "loss may be incomplete)", count_incomplete(entries) - untracked, entries.size());
        if (untracked)
            std::fprintf(out, "; %zu more not tracked on the GPU (every breadcrumb slot was busy)", untracked);
        std::fprintf(out, ":\n");
        for (std::size_t i = first; i < entries.size(); ++i) write_entry(out, entries[i]);
    }

private:
    template<class Map, class Key>
    void label(Map& names, Key key, std::wstring name) {
        std::lock_guard lock(mutex_);
        if (name.empty()) {
            names.erase(key);
            return;
        }
        if (names.size() >= 4096 && !names.count(key)) names.clear();  // bounded; refreshed before each submission
        names[key] = std::move(name);
    }

    static void write_entry(std::FILE* out, const Entry& e) {
        std::fprintf(out, "[rt] d3d12 DRED:   list %#llx", static_cast<unsigned long long>(e.list_id));
        if (!e.list_name.empty()) std::fprintf(out, " \"%s\"", breadcrumb_utf8(e.list_name).c_str());
        std::fprintf(out, " on queue %#llx", static_cast<unsigned long long>(e.queue_id));
        if (!e.queue_name.empty()) std::fprintf(out, " \"%s\"", breadcrumb_utf8(e.queue_name).c_str());
        if (e.progress_known && e.state == BreadcrumbState::started)
            std::fprintf(out, ", tag %u: %s, %u of %zu operations completed", e.tag, breadcrumb_state_name(e.state),
                         e.completed_ops, e.ops.size());
        else if (e.state == BreadcrumbState::started)  // list markers only: how far it got is not known
            std::fprintf(out, ", tag %u: %s, %zu operations, progress inside the list not recorded", e.tag,
                         breadcrumb_state_name(e.state), e.ops.size());
        else
            std::fprintf(out, ", tag %u: %s, %zu operations", e.tag, breadcrumb_state_name(e.state), e.ops.size());
        const std::size_t shown = e.ops.size() < 8 ? e.ops.size() : 8;
        for (std::size_t k = 0; k < shown; ++k)
            std::fprintf(out, "%s%s", k ? ", " : ": ", breadcrumb_op_name(e.ops[k]));
        std::fprintf(out, "%s%s\n", e.ops.size() > shown ? ", ..." : "",
                     e.no_slot ? " (every breadcrumb slot was busy: not tracked on the GPU)" : "");
        // A list the GPU stopped inside: the operations around the first one that did not finish, numbered, with
        // the shader of each draw or dispatch.
        if (e.progress_known && e.state == BreadcrumbState::started && e.completed_ops < e.ops.size()) {
            const std::size_t at = e.completed_ops;
            const std::size_t from = at > 4 ? at - 4 : 0, to = std::min(e.ops.size(), at + 3);
            for (std::size_t k = from; k < to; ++k) {
                const std::uintptr_t pso = k < e.pipelines.size() ? e.pipelines[k] : 0;
                const std::string name = pso ? breadcrumb_pso_name(pso) : std::string();
                std::fprintf(out, "[rt] d3d12 DRED:     %s op %zu %s%s%s", k == at ? "->" : "  ", k,
                             breadcrumb_op_name(e.ops[k]), pso ? " pso " : "",
                             pso ? (name.empty() ? "(unnamed)" : name.c_str()) : "");
                if (k < e.details.size() && (e.details[k][0] | e.details[k][1] | e.details[k][2])) {
                    const auto& d = e.details[k];
                    std::fprintf(out, " groups %u x %u x %u, user data %08x %08x %08x %08x %08x %08x %08x %08x", d[0], d[1],
                                 d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10]);
                    // The descriptors a word names when it is an address in the GPU address arena (the V# a shader
                    // loads through it): 4 dwords each, for the stuck operation.
                    if (k == at && debug_targets::arena_mapped) {
                        const std::array<std::uint32_t, 32>* then = nullptr;
                        for (const auto& s : e.submitted_words)
                            if (s.first == k) then = &s.second;
                        for (int w = 3; w < 11; ++w)
                            if (arena_word(d[w])) {
                                const auto* v = reinterpret_cast<const std::uint32_t*>(debug_targets::arena_mapped + d[w]);
                                std::fprintf(out, "\n[rt] d3d12 DRED:          at user data %d (%#x): %08x %08x %08x %08x",
                                             w - 3, d[w], v[0], v[1], v[2], v[3]);
                                if (then) {
                                    const std::uint32_t* s = then->data() + (w - 3) * 4;
                                    std::fprintf(out, " (at submission %08x %08x %08x %08x)", s[0], s[1], s[2], s[3]);
                                }
                                debug_targets::log_table(out, w - 3, d[w]);
                            }
                    }
                }
                for (const auto& [op, text] : e.notes)
                    if (op == k) std::fprintf(out, ": %s", text.c_str());
                std::fprintf(out, "\n");
            }
        }
    }
    // A user-data word that is a 4-aligned address of 16 bytes in the GPU address arena, past its zero page.
    static bool arena_word(std::uint32_t w) {
        return w >= 0x10000 && w % 4 == 0 && std::uint64_t(w) + 16 <= debug_targets::arena_size;
    }
    static std::size_t count_incomplete(const std::vector<Entry>& entries) {
        std::size_t n = 0;
        for (const auto& e : entries) n += e.state != BreadcrumbState::completed;
        return n;
    }
    VkDeviceSize offset(std::uint32_t slot, std::uint32_t word) const {
        return (VkDeviceSize(slot) * kWords + word) * sizeof(std::uint32_t);
    }
    std::uint32_t read(std::uint32_t slot, std::uint32_t word) const {
        return static_cast<const volatile std::uint32_t*>(mapped_)[slot * kWords + word];
    }

    VkDevice device_ = VK_NULL_HANDLE;
    BreadcrumbFunctions fn_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    std::uint32_t* mapped_ = nullptr;
    bool gpu_ = false;
    mutable std::mutex mutex_;
    std::uint32_t next_tag_ = 0;
    std::uint32_t next_slot_ = 0;
    std::uint64_t next_sequence_ = 0;
    std::uint32_t next_op_ = 0;  // the last op number handed out
    bool per_op_ = true;
    static constexpr std::uint32_t kWords = 4;  // per slot: begin tag, end tag, progress (op number), padding
    std::deque<Mark> history_;
    std::uint32_t owner_[kSlots] = {};  // the tag each slot was last handed out with
    std::unordered_map<const CommandList*, std::wstring> list_names_;
    std::unordered_map<const void*, std::wstring> queue_names_;
};

}  // namespace leonida
