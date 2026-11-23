// Device-lost diagnostics (runtime/src/vulkan_breadcrumbs.h), without a device loss:
//   1. the state of a list from its two marker words (submitted / started / completed) and the D3D12 operations
//      recorded for it, on synthetic data;
//   2. the report format on synthetic entries;
//   3. on a GPU with VK_AMD_buffer_marker: a list that ran reads "completed"; a list held behind a Wait on a fence
//      that is not signalled reads "submitted" (the GPU has not started it, whatever the CPU submitted) until the
//      fence is signalled, then "completed"; a ring that wraps forgets the lists whose slots were reused;
//   4. fallbacks: no marker command, or a context that was never configured, leaves the breadcrumbs off and
//      execute() unchanged.
// The words are read after the queue is idle; on a real loss they are best effort (no memory dependency to the
// host is possible then), which the report says. Skips the GPU part (exit 77 when nothing else ran) without a
// Vulkan 1.3 device or the extension. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

std::string report_of(const std::vector<BreadcrumbRing::Entry>& entries) {
#ifdef _MSC_VER
    std::FILE* f = nullptr;
    if (tmpfile_s(&f) != 0 || !f) return {};
#else
    std::FILE* f = std::tmpfile();
    if (!f) return {};
#endif
    BreadcrumbRing::write_report(f, entries);
    std::rewind(f);
    std::string out;
    char buffer[512];
    std::size_t n = 0;
    while ((n = std::fread(buffer, 1, sizeof buffer, f)) > 0) out.append(buffer, n);
    std::fclose(f);
    return out;
}

const BreadcrumbRing::Entry* find(const std::vector<BreadcrumbRing::Entry>& entries, std::uintptr_t list) {
    for (const auto& e : entries)
        if (e.list_id == list) return &e;
    return nullptr;
}

int synthetic() {
    // 1. A word holds the tag only once the GPU wrote it: an older tag (slot reuse) or 0 is "not yet".
    EXPECT(breadcrumb_state(0, 0, 7) == BreadcrumbState::submitted);
    EXPECT(breadcrumb_state(3, 3, 7) == BreadcrumbState::submitted);  // the previous list of the slot
    EXPECT(breadcrumb_state(7, 3, 7) == BreadcrumbState::started);
    EXPECT(breadcrumb_state(7, 7, 7) == BreadcrumbState::completed);
    EXPECT(std::strcmp(breadcrumb_state_name(BreadcrumbState::started), "started (not completed)") == 0);

    CommandList list;
    EXPECT(list.copy_buffer({2, 0, 1, 0, 16}));
    EXPECT(list.uav_barrier({2}));
    EXPECT(list.close());
    std::vector<std::uint32_t> ops;
    std::uint32_t op = 0;
    for (const auto& command : list.commands())
        if (breadcrumb_op_of(command, op)) ops.push_back(op);
    EXPECT(ops.size() == 2);
    EXPECT(ops[0] == breadcrumb_op::copy_buffer_region && ops[1] == breadcrumb_op::resource_barrier);
    EXPECT(breadcrumb_op::copy_buffer_region == 7 && breadcrumb_op::resource_barrier == 15);  // D3D12 values

    // 2. The report starts at the last completed list before the first one that did not complete.
    std::vector<BreadcrumbRing::Entry> entries = {
        {1, 0x10, {breadcrumb_op::copy_buffer_region}, BreadcrumbState::completed},
        {2, 0x20, {breadcrumb_op::dispatch}, BreadcrumbState::completed},
        {3, 0x30, std::vector<std::uint32_t>(10, breadcrumb_op::draw_indexed_instanced), BreadcrumbState::started},
        {4, 0x40, {}, BreadcrumbState::submitted},
    };
    const std::string text = report_of(entries);
    EXPECT(text.find("2 of the last 4 lists did not complete") != std::string::npos);
    EXPECT(text.find("list 0x10") == std::string::npos);
    EXPECT(text.find("list 0x20 on queue 0, tag 2: completed, 1 operations: Dispatch") != std::string::npos);
    EXPECT(text.find("list 0x30 on queue 0, tag 3: started (not completed), 10 operations, progress inside the list "
                     "not recorded: DrawIndexedInstanced") !=
           std::string::npos);
    EXPECT(text.find(", ...") != std::string::npos);  // more than 8 operations are cut
    EXPECT(text.find("list 0x40 on queue 0, tag 4: submitted (not started), 0 operations") != std::string::npos);
    EXPECT(text.find("may be incomplete") != std::string::npos);
    entries.resize(2);
    EXPECT(report_of(entries).find("every one of the 2 lists kept completed") != std::string::npos);
    EXPECT(report_of({}).find("no command list was submitted") != std::string::npos);

    // Without GPU breadcrumbs: the last 16 submitted lists, their progress UNKNOWN, with names and queue.
    std::vector<BreadcrumbRing::Entry> cpu(20);
    for (std::uint32_t i = 0; i < cpu.size(); ++i) {
        cpu[i].tag = i + 1;
        cpu[i].list_id = 0x100 + i;
        cpu[i].queue_id = 0x9000;
        cpu[i].state = BreadcrumbState::unknown;
    }
    cpu.back().list_name = L"shadow pass \u00e9";  // UTF-8 in the report
    cpu.back().queue_name = L"direct";
    const std::string unknown = report_of(cpu);
    EXPECT(unknown.find("GPU progress UNKNOWN (no VK_AMD_buffer_marker); the last 16 of the 20 lists submitted") !=
           std::string::npos);
    EXPECT(unknown.find("list 0x103 ") == std::string::npos && unknown.find("list 0x104 ") != std::string::npos);
    EXPECT(unknown.find("list 0x113 \"shadow pass \xc3\xa9\" on queue 0x9000 \"direct\", tag 20: submitted, GPU "
                        "progress UNKNOWN") != std::string::npos);
    EXPECT(unknown.find("completed") == std::string::npos && unknown.find("not started") == std::string::npos);
    // A submission that failed leaves nothing behind; the next one is kept.
    {
        BreadcrumbRing ring;
        ring.initialize_cpu_only();
        std::vector<BreadcrumbRing::Mark> lost{ring.mark(list)};
        ring.submitted(std::move(lost), false);
        EXPECT(ring.snapshot().empty());
        std::vector<BreadcrumbRing::Mark> kept{ring.mark(list)};
        const std::uint32_t tag = kept[0].tag;
        ring.submitted(std::move(kept), true);
        const auto kept_entries = ring.snapshot();
        EXPECT(kept_entries.size() == 1 && kept_entries[0].tag == tag &&
               kept_entries[0].state == BreadcrumbState::unknown);
    }
    // Per-operation progress: the word holds first + k after op k; anything outside the list's range is nothing.
    EXPECT(breadcrumb_ops_completed(0, 100, 3) == 0);    // never written
    EXPECT(breadcrumb_ops_completed(99, 100, 3) == 0);   // the slot's previous owner
    EXPECT(breadcrumb_ops_completed(100, 100, 3) == 1);
    EXPECT(breadcrumb_ops_completed(102, 100, 3) == 3);
    EXPECT(breadcrumb_ops_completed(103, 100, 3) == 0);  // a later owner (cannot be this list's)
    EXPECT(breadcrumb_ops_completed(1, 0xfffffffeu, 4) == 4);  // across the 32-bit wrap
    EXPECT(breadcrumb_ops_completed(5, 5, 0) == 0);      // a list without operations
    BreadcrumbRing::Entry partial;
    partial.tag = 9;
    partial.list_id = 0x77;
    partial.state = BreadcrumbState::started;
    partial.ops = {breadcrumb_op::copy_buffer_region, breadcrumb_op::draw_instanced, breadcrumb_op::dispatch};
    partial.progress_known = true;
    partial.completed_ops = 1;
    EXPECT(report_of({partial}).find("tag 9: started (not completed), 1 of 3 operations completed: CopyBufferRegion, "
                                     "DrawInstanced, Dispatch") != std::string::npos);
    partial.completed_ops = 0;  // observed: none finished yet, which is not the same as unknown
    EXPECT(report_of({partial}).find("tag 9: started (not completed), 0 of 3 operations completed") != std::string::npos);
    partial.progress_known = false;
    EXPECT(report_of({partial}).find("3 operations, progress inside the list not recorded") != std::string::npos);
    std::printf("breadcrumbs: state, operations and report on synthetic data ok\n");
    return 0;
}


void submit_to(BreadcrumbRing& ring, const BreadcrumbRing::Mark& m) {
    std::vector<BreadcrumbRing::Mark> v{m};
    ring.submitted(std::move(v), true);
}

// Slot ownership at level 2 on a ring whose "GPU" is this test: the marker words are a vector the test owns, the
// marker command writes into it, and the test runs each list's markers itself, in the order a GPU might. No device.
// The case from the review of #5314 (DoubleD17): every slot held by an unfinished list A..., then B and C.
namespace fake {
std::vector<std::uint32_t> words;
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer* b) {
    *b = reinterpret_cast<VkBuffer>(std::uintptr_t(0x1000));
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL requirements(VkDevice, VkBuffer, VkMemoryRequirements* r) {
    *r = {VkDeviceSize(words.size() * 4), 4, 1};
}
VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory* m) {
    *m = reinterpret_cast<VkDeviceMemory>(std::uintptr_t(0x2000));
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL bind(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL map(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void** out) {
    *out = words.data();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL unmap(VkDevice, VkDeviceMemory) {}
VKAPI_ATTR void VKAPI_CALL write_marker(VkCommandBuffer, VkPipelineStageFlagBits, VkBuffer, VkDeviceSize offset,
                                        std::uint32_t value) {
    words[offset / 4] = value;  // "recording" a marker runs it at once: the test orders the lists' markers itself
}
}  // namespace fake

int slot_ownership() {
    const std::size_t slots = BreadcrumbRing::kSlots;
    fake::words.assign(slots * 4, 0xdeadbeefu);  // initialize() must clear them
    BreadcrumbFunctions fn;
    fn.create_buffer = fake::create_buffer;
    fn.destroy_buffer = fake::destroy_buffer;
    fn.requirements = fake::requirements;
    fn.allocate = fake::allocate;
    fn.free = fake::free_memory;
    fn.bind = fake::bind;
    fn.map = fake::map;
    fn.unmap = fake::unmap;
    fn.write_marker = fake::write_marker;
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryTypeCount = 1;
    memory.memoryTypes[0] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0};
    memory.memoryHeapCount = 1;
    memory.memoryHeaps[0] = {VkDeviceSize(1) << 30, 0};
    BreadcrumbRing ring;
    EXPECT(ring.initialize(reinterpret_cast<VkDevice>(std::uintptr_t(0x3000)), memory, fn, true));
    EXPECT(ring.gpu() && ring.per_op());
    CommandList one;
    EXPECT(one.copy_buffer({2, 0, 1, 0, 16}) && one.close());
    const auto id = [](std::size_t k) { return reinterpret_cast<const CommandList*>(std::uintptr_t(0x10000 + 16 * k)); };
    const int q1 = 0, q2 = 0;  // two queues, as opaque addresses
    const void* queue1 = &q1;
    const void* queue2 = &q2;
    const VkCommandBuffer cb = VK_NULL_HANDLE;
    // The GPU part of a list: its begin word, its operation (if it has the per-op marker) and its end word.
    const auto begin = [&](const BreadcrumbRing::Mark& m) { ring.record_begin(cb, m); };
    const auto op = [&](const BreadcrumbRing::Mark& m) {
        if (ring.op_markers_for(m)) fn.write_marker(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ring.marker_buffer(),
                                                    ring.progress_offset(m), m.first_op);
    };
    const auto end = [&](const BreadcrumbRing::Mark& m) { ring.record_end(cb, m); };
    const auto submit = [&](const BreadcrumbRing::Mark& m, bool ok) {
        std::vector<BreadcrumbRing::Mark> v{m};
        ring.submitted(std::move(v), ok);
    };

    // Every slot is taken by a list the GPU started and has not finished; A is the first, on queue 1.
    std::vector<BreadcrumbRing::Mark> running;
    for (std::size_t k = 0; k < slots; ++k) {
        running.push_back(ring.mark(one, queue1, id(k)));
        submit(running.back(), true);
        begin(running.back());
    }
    const BreadcrumbRing::Mark a = running[0];
    const std::vector<std::uint32_t> before = fake::words;

    // B, on queue 2, finds no free slot. It must take none: no word of any slot changes, A keeps its slot.
    const BreadcrumbRing::Mark b = ring.mark(one, queue2, id(slots));
    EXPECT(!ring.op_markers_for(b));
    submit(b, true);
    begin(b);
    op(b);
    end(b);  // B finishes on the GPU
    EXPECT(fake::words == before);
    // C comes after B finished: there is still no free slot, so C must not take A's (or any running list's).
    const BreadcrumbRing::Mark c = ring.mark(one, queue2, id(slots + 1));
    EXPECT(!ring.op_markers_for(c));
    submit(c, true);
    begin(c);
    op(c);
    op(a);  // A's late operation marker
    EXPECT(fake::words == [&] {
        auto w = before;
        if (ring.op_markers_for(a)) w[a.slot * 4 + 2] = a.first_op;
        return w;
    }());
    auto entries = ring.snapshot();
    const auto* ea = find(entries, reinterpret_cast<std::uintptr_t>(id(0)));
    EXPECT(ea && ea->state == BreadcrumbState::started && ea->progress_known && ea->completed_ops == 1);  // A stays
    const auto* eb = find(entries, reinterpret_cast<std::uintptr_t>(id(slots)));
    const auto* ec = find(entries, reinterpret_cast<std::uintptr_t>(id(slots + 1)));
    EXPECT(eb && ec && eb->state == BreadcrumbState::unknown && ec->state == BreadcrumbState::unknown);
    EXPECT(!eb->progress_known && !ec->progress_known && eb->completed_ops == 0 && ec->completed_ops == 0);
    const std::string text = report_of(entries);
    EXPECT(text.find("no VK_AMD_buffer_marker") == std::string::npos);
    EXPECT(text.find("every breadcrumb slot was busy") != std::string::npos);
    EXPECT(text.find("; 2 more not tracked on the GPU") != std::string::npos);  // B and C, not counted as incomplete
    // A failed submission without a slot frees nothing either.
    const BreadcrumbRing::Mark f = ring.mark(one, queue2, id(slots + 2));
    submit(f, false);
    EXPECT(!ring.op_markers_for(ring.mark(one, queue2, id(slots + 3))));  // marked, never submitted: no slot

    // A finishes: its slot is free, and the next list takes exactly it, with per-op progress.
    end(a);
    const BreadcrumbRing::Mark d = ring.mark(one, queue1, id(slots + 4));
    EXPECT(d.slot == a.slot && ring.op_markers_for(d));
    submit(d, true);
    begin(d);
    EXPECT(fake::words[d.slot * 4 + 2] == 0);  // the begin marker cleared A's old progress word
    op(d);
    end(d);
    // A failed submission gives its slot back at once.
    const BreadcrumbRing::Mark e = ring.mark(one, queue1, id(slots + 5));
    EXPECT(e.slot == d.slot && ring.op_markers_for(e));
    submit(e, false);
    const BreadcrumbRing::Mark g = ring.mark(one, queue1, id(slots + 6));
    EXPECT(g.slot == e.slot && ring.op_markers_for(g));
    submit(g, true);
    begin(g);
    op(g);
    end(g);
    entries = ring.snapshot();
    const auto* eg = find(entries, reinterpret_cast<std::uintptr_t>(id(slots + 6)));
    EXPECT(eg && eg->state == BreadcrumbState::completed && eg->completed_ops == 1);
    // Unfinished lists stay in the history however many lists without a slot came after them.
    for (std::size_t k = 0; k < slots + 8; ++k) submit(ring.mark(one, queue2, id(slots + 10 + k)), true);
    entries = ring.snapshot();
    EXPECT(entries.size() <= slots + BreadcrumbRing::kNoSlotKept);
    EXPECT(find(entries, reinterpret_cast<std::uintptr_t>(id(1))) != nullptr);  // the second running list
    EXPECT(find(entries, reinterpret_cast<std::uintptr_t>(id(slots - 1))) != nullptr);
    EXPECT(find(entries, reinterpret_cast<std::uintptr_t>(id(slots + 10 + slots + 7))) != nullptr);  // the newest

    // CPU only: no list ever claims GPU progress or gets a marker, however many are submitted.
    BreadcrumbRing cpu;
    cpu.initialize_cpu_only();
    for (std::size_t k = 0; k < slots + 5; ++k) {
        const BreadcrumbRing::Mark m = cpu.mark(one, queue1, id(k));
        EXPECT(!cpu.op_markers_for(m));
        submit_to(cpu, m);
    }
    for (const auto& x : cpu.snapshot())
        EXPECT(x.state == BreadcrumbState::unknown && !x.progress_known && x.completed_ops == 0);
    std::printf("breadcrumbs: slot ownership with every slot busy (review of #5314: A/B/C), failed submissions, "
                "history and CPU only ok\n");
    return 0;
}

}  // namespace

int main() {
    if (synthetic() || slot_ownership()) return 1;

    vktest::Device d;
    d.request_buffer_marker = true;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_breadcrumbs_test: no Vulkan 1.3 device, GPU part skipped\n");
        return 0;
    }

    void* src_ptr = nullptr;
    void* dst_ptr = nullptr;
    const VkBuffer src = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_ptr);
    const VkBuffer dst = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst_ptr);
    EXPECT(src && dst);
    ResourceBinding src_rb{};
    src_rb.buffer = src;
    src_rb.size = 64;
    ResourceBinding dst_rb{};
    dst_rb.buffer = dst;
    dst_rb.size = 64;
    dst_rb.host_readable = true;
    CommandList a, b;
    EXPECT(a.copy_buffer({2, 0, 1, 0, 16}));
    EXPECT(a.close());
    EXPECT(b.copy_buffer({2, 16, 1, 16, 16}));
    EXPECT(b.uav_barrier({2}));
    EXPECT(b.close());
    const CommandList* pa = &a;
    const CommandList* pb = &b;

    // List markers only (no per-op markers): a finished list reads all its operations completed (its end word);
    // inside a list nothing is known, so a started one would say nothing about its operations, not "0 completed".
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        if (d.buffer_marker) {
            EXPECT(context.configure_breadcrumbs(d.memory_properties(), true, false, false));
            BreadcrumbRing* ring = context.breadcrumbs();
            EXPECT(ring && ring->gpu() && !ring->per_op() && !ring->op_marker_function());
            context.add_resource(1, src_rb);
            context.add_resource(2, dst_rb);
            VulkanQueue queue(context, d.queue, d.family);
            EXPECT(queue.execute({&pb, 1}) == ReplayResult::success);
            EXPECT(queue.idle());
            const auto entries = ring->snapshot();
            EXPECT(entries.size() == 1 && entries[0].state == BreadcrumbState::completed &&
                   entries[0].progress_known && entries[0].completed_ops == 2);
            std::printf("breadcrumbs: list markers only: a finished list, no per-op progress claimed ok\n");
        }
    }

    // 4. Fallbacks: a context that was not configured has no breadcrumbs; one without the marker command refuses;
    //    without the marker extension the ring keeps the CPU's view only.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        EXPECT(context.breadcrumbs() == nullptr);
        context.add_resource(1, src_rb);
        context.add_resource(2, dst_rb);
        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());
        EXPECT(queue.execute({&pa, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        BreadcrumbRing ring;
        BreadcrumbFunctions none;  // no vkCmdWriteBufferMarkerAMD
        EXPECT(!ring.initialize(d.device, d.memory_properties(), none));
        EXPECT(ring.snapshot().empty());
        std::printf("breadcrumbs: off without configuration or marker command ok\n");
    }
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        EXPECT(!context.configure_breadcrumbs(d.memory_properties(), false, false));  // CPU only
        BreadcrumbRing* ring = context.breadcrumbs();
        EXPECT(ring && !ring->gpu());
        context.add_resource(1, src_rb);
        context.add_resource(2, dst_rb);
        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());
        ring->label_queue(&queue, L"copy queue");
        ring->label_list(pa, L"list A");
        ring->label_list(pb, L"");  // unnamed
        const CommandList* both[] = {&a, &b};
        EXPECT(queue.execute(both) == ReplayResult::success);
        EXPECT(queue.idle());
        const auto entries = ring->snapshot();
        EXPECT(entries.size() == 2);
        for (const auto& e : entries) {
            EXPECT(e.state == BreadcrumbState::unknown && !e.progress_known && e.completed_ops == 0);  // no GPU writes
            EXPECT(e.queue_id == reinterpret_cast<std::uintptr_t>(&queue) && e.queue_name == L"copy queue");
        }
        EXPECT(entries[0].list_name == L"list A" && entries[1].list_name.empty());
        ring->label_list(pa, L"");  // a later submission without a name carries none
        EXPECT(queue.execute({&pa, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        EXPECT(ring->snapshot().back().list_name.empty());
        // A queue that never names itself (the device's staging queue) and lands at the address of a named queue
        // that was destroyed must not carry that queue's name.
        std::optional<VulkanQueue> reused;
        reused.emplace(context, d.queue, d.family);
        ring->label_queue(&*reused, L"destroyed queue");
        reused.reset();
        reused.emplace(context, d.queue, d.family);  // the same storage, so the same address
        EXPECT(reused->ok());
        EXPECT(reused->execute({&pa, 1}) == ReplayResult::success);
        EXPECT(reused->idle());
        EXPECT(ring->snapshot().back().queue_name.empty());
        reused.reset();
        std::printf("breadcrumbs: CPU only without VK_AMD_buffer_marker: submitted lists, queue and names, progress "
                    "UNKNOWN ok\n");
    }

    if (!d.buffer_marker) {
        std::fprintf(stderr, "vk_breadcrumbs_test: no VK_AMD_buffer_marker, GPU breadcrumbs skipped\n");
        return d.validation_errors() ? 1 : 0;
    }

    // 3. The GPU's view of two lists.
    {
        VulkanContext context(d.device_proc, d.device, true);
        EXPECT(context.loaded());
        EXPECT(context.configure_breadcrumbs(d.memory_properties(), true, false));
        EXPECT(context.breadcrumbs()->gpu());
        BreadcrumbRing* ring = context.breadcrumbs();
        EXPECT(ring);
        context.add_resource(1, src_rb);
        context.add_resource(2, dst_rb);
        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());
        VulkanFence fence(context, 0);
        EXPECT(fence.semaphore() != VK_NULL_HANDLE);

        EXPECT(queue.execute({&pa, 1}) == ReplayResult::success);
        EXPECT(queue.idle());
        auto entries = ring->snapshot();
        EXPECT(entries.size() == 1);
        EXPECT(entries[0].list_id == reinterpret_cast<std::uintptr_t>(pa));
        EXPECT(entries[0].state == BreadcrumbState::completed);
        EXPECT(entries[0].ops.size() == 1 && entries[0].ops[0] == breadcrumb_op::copy_buffer_region);

        // A Wait for a fence value no Signal has reached holds the batch on the CPU (QueueHolds): nothing reached
        // vkQueueSubmit, so the ring has nothing for it. The Signal releases it; the ring then marks the held copy
        // under the list the application passed, with that list's name.
        ring->label_list(pb, L"held list");
        queue.wait(fence.semaphore(), 1);
        EXPECT(queue.execute({&pb, 1}) == ReplayResult::success);
        entries = ring->snapshot();
        EXPECT(entries.size() == 1 && !find(entries, reinterpret_cast<std::uintptr_t>(pb)));
        EXPECT(report_of(entries).find("every one of the 1 lists kept completed") != std::string::npos);

        EXPECT(fence.signal(1));
        EXPECT(queue.idle());
        entries = ring->snapshot();
        EXPECT(entries.size() == 2);
        const auto* released = find(entries, reinterpret_cast<std::uintptr_t>(pb));
        EXPECT(released && released->state == BreadcrumbState::completed && released->list_name == L"held list");
        EXPECT(released->ops.size() == 2 && released->ops[1] == breadcrumb_op::resource_barrier);
        EXPECT(released->progress_known && released->completed_ops == 2);  // the held copy got the per-op markers
        ring->label_list(pb, L"");
        std::printf("breadcrumbs: completed / held behind an unsignalled fence, then released under its own id and "
                    "name / completed ok\n");

        // Held batches keep the names they had at ExecuteCommandLists, while the application's objects were alive:
        //   1. X is held, then X and its queue are renamed: the breadcrumb keeps the old names;
        //   2. the application releases X and a new list Y takes its storage, is named and runs at once on another
        //      queue before X's batch is released: X's breadcrumb must not take Y's name;
        //   3. a second batch, Z, is held on the other queue behind another fence; Z is released first, then X:
        //      each keeps its own name, and the ring lists them in the order they reached vkQueueSubmit.
        {
            VulkanQueue other(context, d.queue, d.family);
            EXPECT(other.ok());
            VulkanFence fence_x(context, 0), fence_z(context, 0);
            EXPECT(fence_x.semaphore() != VK_NULL_HANDLE && fence_z.semaphore() != VK_NULL_HANDLE);
            std::optional<CommandList> storage;
            storage.emplace(b);  // X: b's two operations
            const CommandList* px = &*storage;
            ring->label_queue(&queue, L"queue one");
            ring->label_list(px, L"X");
            queue.wait(fence_x.semaphore(), 1);
            EXPECT(queue.execute({&px, 1}) == ReplayResult::success);  // held
            ring->label_list(px, L"X renamed");                         // 1.
            ring->label_queue(&queue, L"queue one renamed");
            storage.reset();                                            // 2. X released ...
            storage.emplace(a);                                         //    ... Y (a's one operation) in its place
            EXPECT(&*storage == px);
            ring->label_list(px, L"Y");
            EXPECT(other.execute({&px, 1}) == ReplayResult::success);   //    Y is not held: it runs now
            EXPECT(other.idle());
            const CommandList* pz = pa;                                 // 3.
            ring->label_list(pz, L"Z");
            other.wait(fence_z.semaphore(), 1);
            EXPECT(other.execute({&pz, 1}) == ReplayResult::success);  // held
            ring->label_list(pz, L"Z renamed");
            EXPECT(fence_z.signal(1));  // releases Z first
            EXPECT(other.idle());
            EXPECT(fence_x.signal(1));  // then X
            EXPECT(queue.idle());
            entries = ring->snapshot();
            EXPECT(entries.size() >= 3);
            const BreadcrumbRing::Entry& y = entries[entries.size() - 3];
            const BreadcrumbRing::Entry& z = entries[entries.size() - 2];
            const BreadcrumbRing::Entry& x = entries[entries.size() - 1];
            EXPECT(y.list_name == L"Y" && y.ops.size() == 1 && y.queue_id == reinterpret_cast<std::uintptr_t>(&other));
            EXPECT(z.list_name == L"Z" && z.list_id == reinterpret_cast<std::uintptr_t>(pz) && z.ops.size() == 1);
            EXPECT(x.list_name == L"X" && x.list_id == reinterpret_cast<std::uintptr_t>(px) && x.ops.size() == 2);
            EXPECT(x.queue_name == L"queue one" && x.queue_id == reinterpret_cast<std::uintptr_t>(&queue));
            EXPECT(y.tag < z.tag && z.tag < x.tag);  // marked when submitted, not when held
            for (const auto* e : {&y, &z, &x}) EXPECT(e->state == BreadcrumbState::completed);
            ring->label_list(px, L"");
            ring->label_list(pz, L"");
            ring->label_queue(&queue, L"");
            std::printf("breadcrumbs: held batches keep the names they had when held (renamed, address reused, "
                        "released out of order) ok\n");
        }

        // Per-operation markers: three operations in one list, two lists in one submission, and a list that is
        // reset and recorded again between two submissions; each history keeps what was submitted.
        {
            CommandList three;
            // Bytes 16..48 only: `a`, in the same submission, writes 0..16, and two copies into the same bytes would need
            // a copy -> copy barrier between them.
            EXPECT(three.copy_buffer({2, 16, 1, 16, 16}) && three.uav_barrier({2}) && three.copy_buffer({2, 32, 1, 32, 16}));
            EXPECT(three.close());
            const CommandList* pthree = &three;
            const CommandList* pair[] = {pthree, pa};
            EXPECT(queue.execute(pair) == ReplayResult::success);
            EXPECT(queue.idle());
            three = CommandList{};  // Reset: one operation now
            EXPECT(three.copy_buffer({2, 48, 1, 48, 16}) && three.close());
            EXPECT(queue.execute({&pthree, 1}) == ReplayResult::success);
            EXPECT(queue.idle());
            entries = ring->snapshot();
            EXPECT(entries.size() >= 3);
            const auto& first = entries[entries.size() - 3];
            const auto& second = entries[entries.size() - 2];
            const auto& again = entries[entries.size() - 1];
            EXPECT(first.ops.size() == 3 && first.progress_known && first.completed_ops == 3);
            EXPECT(second.ops.size() == 1 && second.completed_ops == 1);
            EXPECT(again.list_id == first.list_id && again.ops.size() == 1 && again.completed_ops == 1);
            std::printf("breadcrumbs: per-operation progress over 3 + 1 operations and a reset list ok\n");
        }

        // Op numbers across the 32-bit wrap, on the GPU: a range that would contain 0 starts after it, and the
        // lists on both sides of the wrap read every operation as completed.
        {
            CommandList four;
            EXPECT(four.copy_buffer({2, 16, 1, 16, 16}) && four.uav_barrier({2}) && four.copy_buffer({2, 32, 1, 32, 16}) &&
                   four.uav_barrier({2}));
            EXPECT(four.close());
            const CommandList* pfour = &four;
            ring->skip_op_numbers_to(0xfffffffdu);  // the next list's 4 op numbers would wrap through 0
            for (int i = 0; i < 3; ++i) EXPECT(queue.execute({&pfour, 1}) == ReplayResult::success);
            EXPECT(queue.idle());
            entries = ring->snapshot();
            for (std::size_t i = entries.size() - 3; i < entries.size(); ++i)
                EXPECT(entries[i].ops.size() == 4 && entries[i].progress_known && entries[i].completed_ops == 4);
            std::printf("breadcrumbs: per-operation progress across the 32-bit wrap of op numbers ok\n");
        }

        // Two lists in one ExecuteCommandLists get a slot each, and the ring keeps the newest kSlots lists.
        const CommandList* both[] = {&a, &b};
        for (std::uint32_t i = 0; i < BreadcrumbRing::kSlots / 2; ++i)
            EXPECT(queue.execute(both) == ReplayResult::success);
        EXPECT(queue.idle());
        entries = ring->snapshot();
        EXPECT(entries.size() == BreadcrumbRing::kSlots);
        for (const auto& e : entries)
            EXPECT(e.state == BreadcrumbState::completed && e.progress_known && e.completed_ops == e.ops.size());
        EXPECT(entries.back().list_id == reinterpret_cast<std::uintptr_t>(pb));
        EXPECT(entries[entries.size() - 2].list_id == reinterpret_cast<std::uintptr_t>(pa));
        std::printf("breadcrumbs: %u lists across a ring wrap ok\n", BreadcrumbRing::kSlots + 2);

        // Two queues on two threads through the full ring: every list gets its own slot and tag, the snapshot is
        // in marking order, and the lists whose slots were reused are gone.
        VulkanQueue second(context, d.queue, d.family);
        EXPECT(second.ok());
        constexpr std::uint32_t per_thread = 400;
        bool ok_a = true, ok_b = true;
        std::thread ta([&] {
            for (std::uint32_t i = 0; i < per_thread; ++i)
                ok_a = ok_a && queue.execute({&pa, 1}) == ReplayResult::success;
        });
        std::thread tb([&] {
            for (std::uint32_t i = 0; i < per_thread; ++i)
                ok_b = ok_b && second.execute({&pb, 1}) == ReplayResult::success;
        });
        ta.join();
        tb.join();
        EXPECT(ok_a && ok_b);
        EXPECT(queue.idle() && second.idle());
        entries = ring->snapshot();
        EXPECT(entries.size() == BreadcrumbRing::kSlots);
        std::uint32_t from_a = 0, from_b = 0;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            EXPECT(entries[i].state == BreadcrumbState::completed && entries[i].completed_ops == entries[i].ops.size());
            if (i) EXPECT(entries[i].tag > entries[i - 1].tag);  // marking order, no tag twice
            if (i >= entries.size() - 2 * per_thread) {
                from_a += entries[i].list_id == reinterpret_cast<std::uintptr_t>(pa);
                from_b += entries[i].list_id == reinterpret_cast<std::uintptr_t>(pb);
            }
        }
        EXPECT(from_a == per_thread && from_b == per_thread);  // the newest 800 are the two threads' lists
        std::printf("breadcrumbs: two queues on two threads, %u lists, ordered and recycled ok\n", 2 * per_thread);
    }

    // Cost (informational, NVIDIA only so far): lists of N operations (a copy, then a copy -> copy barrier, repeated),
    // without the diagnostics, with the list markers only and with the per-op markers. `record` is execute()
    // (prepare, record, submit) on the CPU; `wait` is idle() after it, the GPU finishing. Best of 7.
    {
        const GlobalBarrier copy_to_copy{d3d12_sync::copy, d3d12_sync::copy, d3d12_access::copy_dest,
                                         d3d12_access::copy_dest};
        for (const std::uint32_t n : {100u, 1000u, 2000u, 10000u}) {
            CommandList many;
            for (std::uint32_t i = 0; i < n / 2; ++i)
                EXPECT(many.copy_buffer({2, 0, 1, 0, 16}) && many.barrier(copy_to_copy));
            EXPECT(many.close());
            const CommandList* pmany = &many;
            double record[3] = {1e9, 1e9, 1e9}, wait[3] = {1e9, 1e9, 1e9};
            for (int mode = 0; mode < 3; ++mode) {  // 0 off, 1 list markers, 2 per-op markers
                VulkanContext context(d.device_proc, d.device, true);
                EXPECT(context.loaded());
                if (mode) EXPECT(context.configure_breadcrumbs(d.memory_properties(), true, false, mode == 2));
                context.add_resource(1, src_rb);
                context.add_resource(2, dst_rb);
                VulkanQueue queue(context, d.queue, d.family);
                for (int run = 0; run < 7; ++run) {
                    const auto t0 = std::chrono::steady_clock::now();
                    EXPECT(queue.execute({&pmany, 1}) == ReplayResult::success);
                    const auto t1 = std::chrono::steady_clock::now();
                    EXPECT(queue.idle());
                    const auto t2 = std::chrono::steady_clock::now();
                    record[mode] = std::min(record[mode], std::chrono::duration<double, std::milli>(t1 - t0).count());
                    wait[mode] = std::min(wait[mode], std::chrono::duration<double, std::milli>(t2 - t1).count());
                }
                if (mode) {
                    const auto e = context.breadcrumbs()->snapshot();
                    EXPECT(e.size() == 7 && e.back().ops.size() == n && e.back().completed_ops == n);
                    // The history keeps the ops alone: the per-op positions went with the submission.
                    EXPECT(context.breadcrumbs()->history_words() == 7u * n);
                }
            }
            std::printf("breadcrumbs: cost %5u ops: record %.3f / %.3f / %.3f ms, wait %.3f / %.3f / %.3f ms "
                        "(off / list markers / per-op markers)\n",
                        n, record[0], record[1], record[2], wait[0], wait[1], wait[2]);
        }
    }

    const unsigned errors = d.validation_errors();
    std::printf("breadcrumbs: %u validation errors\n", errors);
    return errors ? 1 : 0;
}
