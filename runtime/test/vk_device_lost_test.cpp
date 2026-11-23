// The device-lost report path of the D3D12 layer's diagnostics (runtime/src/vulkan_backend.h, vulkan_breadcrumbs.h),
// with a controlled loss: the device is real and idle, but vkQueueSubmit returns VK_ERROR_DEVICE_LOST when the test
// says so (without reaching the driver), vkCmdWriteBufferMarkerAMD only counts, and vkGetDeviceFaultInfoEXT is a
// mock with fault data, no data, or an error in its first or second call. Nothing provokes a GPU hang or touches a
// driver's loss handling. Checked:
//   - VK_ERROR_DEVICE_LOST from a submission marks the context lost and prints the report once, with the breadcrumbs
//     (GPU mode) or the CPU's view with UNKNOWN progress (no marker extension), the queue and list names, and the
//     fault description and addresses when there are any;
//   - "no device fault information" when the extension is off, has nothing, or either of its queries fails; no
//     vendor binary data is asked for;
//   - later calls (mark_device_lost, is_device_lost, execute) print nothing more and make no further submission,
//     marker write or fault query;
//   - eight threads reporting the loss at once: one report, all threads return (no deadlock).
// Skips (77) without a Vulkan 1.3 device. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <future>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", label, __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

namespace mock {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;

std::atomic<bool> lose_next_submit{false};
std::atomic<bool> lost{false};
std::atomic<unsigned> submits_after_loss{0};
std::atomic<unsigned> markers{0};
std::atomic<unsigned> fault_calls{0};
std::atomic<unsigned> fault_violations{0};  // vendor binary data asked for, or a call before the loss

enum class Fault { data, empty, fail_counts, fail_info };
std::atomic<Fault> fault{Fault::data};

void reset(Fault f) {
    lose_next_submit = false;
    lost = false;
    submits_after_loss = 0;
    markers = 0;
    fault_calls = 0;
    fault_violations = 0;
    fault = f;
}

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    if (lost) {
        ++submits_after_loss;
        return VK_ERROR_DEVICE_LOST;
    }
    if (lose_next_submit.exchange(false)) {
        lost = true;
        return VK_ERROR_DEVICE_LOST;  // the driver never sees this submission
    }
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR void VKAPI_CALL write_marker(VkCommandBuffer, VkPipelineStageFlagBits, VkBuffer, VkDeviceSize, std::uint32_t) {
    ++markers;  // recorded nowhere: the words keep 0, so the GPU's progress reads "not started"
}

VKAPI_ATTR VkResult VKAPI_CALL device_fault_info(VkDevice, VkDeviceFaultCountsEXT* counts, VkDeviceFaultInfoEXT* info) {
    ++fault_calls;
    if (!lost) ++fault_violations;  // the spec allows the query only after VK_ERROR_DEVICE_LOST
    const Fault f = fault;
    if (!info) {
        if (f == Fault::fail_counts) return VK_ERROR_OUT_OF_HOST_MEMORY;
        counts->addressInfoCount = f == Fault::empty ? 0 : 2;
        counts->vendorInfoCount = f == Fault::empty ? 0 : 1;
        counts->vendorBinarySize = f == Fault::empty ? 0 : 64;
        return VK_SUCCESS;
    }
    if (counts->vendorBinarySize || info->pVendorBinaryData) ++fault_violations;
    if (f == Fault::fail_info) return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (f == Fault::empty) {
        info->description[0] = '\0';
        return VK_SUCCESS;
    }
    std::snprintf(info->description, sizeof info->description, "mock page fault");
    if (info->pAddressInfos && counts->addressInfoCount >= 2) {
        info->pAddressInfos[0] = {VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT, 0x12340000, 0xfff};
        info->pAddressInfos[1] = {VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT, 0x55660000, 0};
    }
    if (info->pVendorInfos && counts->vendorInfoCount >= 1) {
        std::snprintf(info->pVendorInfos[0].description, sizeof info->pVendorInfos[0].description, "vendor");
        info->pVendorInfos[0].vendorFaultCode = 7;
        info->pVendorInfos[0].vendorFaultData = 9;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    if (n == "vkCmdWriteBufferMarkerAMD") return reinterpret_cast<PFN_vkVoidFunction>(write_marker);
    if (n == "vkGetDeviceFaultInfoEXT") return reinterpret_cast<PFN_vkVoidFunction>(device_fault_info);
    return real_get_device(device, name);
}
}  // namespace mock

std::string read_all(std::FILE* f) {
    std::fflush(f);
    std::rewind(f);
    std::string out;
    char buffer[512];
    std::size_t n = 0;
    while ((n = std::fread(buffer, 1, sizeof buffer, f)) > 0) out.append(buffer, n);
    return out;
}

std::FILE* temp_file() {
#ifdef _MSC_VER
    std::FILE* f = nullptr;
    return tmpfile_s(&f) == 0 ? f : nullptr;
#else
    return std::tmpfile();
#endif
}

std::size_t count(const std::string& text, const char* what) {
    std::size_t n = 0;
    for (std::size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
    return n;
}

struct Scene {
    vktest::Device& d;
    VkBuffer src = VK_NULL_HANDLE, dst = VK_NULL_HANDLE;
    CommandList a, b;
};

// One context through a loss: list A submitted and idle, then the submission of list B is lost.
int lose(const char* label, Scene& scene, bool gpu, bool fault_enabled, mock::Fault fault, std::string& report) {
    mock::reset(fault);
    std::FILE* out = temp_file();
    EXPECT(out);
    {
        VulkanContext context(mock::get_device, scene.d.device, true);
        EXPECT(context.loaded());
        context.set_report_output(out);
        EXPECT(context.configure_breadcrumbs(scene.d.memory_properties(), gpu, fault_enabled) == gpu);
        ResourceBinding src{}, dst{};
        src.buffer = scene.src;
        src.size = 64;
        dst.buffer = scene.dst;
        dst.size = 64;
        context.add_resource(1, src);
        context.add_resource(2, dst);
        VulkanQueue queue(context, scene.d.queue, scene.d.family);
        EXPECT(queue.ok());
        BreadcrumbRing* ring = context.breadcrumbs();
        ring->label_queue(&queue, L"mock queue");
        ring->label_list(&scene.a, L"list A");
        const CommandList* pa = &scene.a;
        const CommandList* pb = &scene.b;
        EXPECT(queue.execute({&pa, 1}) == ReplayResult::success);
        EXPECT(queue.idle());  // nothing of the device is busy when the loss comes
        EXPECT(!context.is_device_lost());
        EXPECT(mock::fault_calls == 0);  // never asked before a loss

        mock::lose_next_submit = true;
        EXPECT(queue.execute({&pb, 1}) == ReplayResult::unsupported);
        EXPECT(context.is_device_lost());
        const unsigned markers = mock::markers, faults = mock::fault_calls;
        // Per list: begin, the begin's progress-word clear and end, then one after each operation (per-op
        // progress); A's and B's, recorded before B's submission failed.
        std::uint32_t op = 0;
        unsigned ops = 0;
        for (const CommandList* list : {pa, pb})
            for (const auto& command : list->commands()) ops += breadcrumb_op_of(command, op);
        EXPECT(markers == (gpu ? 6u + ops : 0u));
        // The mock records no marker, so nothing the GPU did is known: no list may claim an operation completed.
        for (const auto& e : ring->snapshot()) EXPECT(e.completed_ops == 0 && e.state != BreadcrumbState::completed);
        EXPECT(faults == (fault_enabled ? (fault == mock::Fault::fail_counts ? 1u : 2u) : 0u));

        // After the loss: no second report, no submission, no marker, no fault query.
        context.mark_device_lost();
        EXPECT(context.is_device_lost());
        EXPECT(queue.execute({&pa, 1}) == ReplayResult::unsupported);
        EXPECT(mock::submits_after_loss == 0 && mock::markers == markers && mock::fault_calls == faults);
        EXPECT(mock::fault_violations == 0);
    }
    report = read_all(out);
    std::fclose(out);
    EXPECT(count(report, "[rt] d3d12 DRED: ") >= 2);
    EXPECT(count(report, "lists did not complete") + count(report, "GPU progress UNKNOWN (no") == 1);  // once
    return 0;
}

int scenarios(Scene& scene) {
    const char* label = "device lost";
    std::string r;

    // GPU breadcrumbs, fault data: A was submitted and (markers dropped) never started; B's failed submission is
    // not kept. The fault description and both addresses follow.
    if (lose(label, scene, true, true, mock::Fault::data, r)) return 1;
    EXPECT(r.find("1 of the last 1 lists did not complete") != std::string::npos);
    EXPECT(r.find("\"list A\" on queue ") != std::string::npos && r.find("\"mock queue\"") != std::string::npos);
    EXPECT(r.find("submitted (not started), 1 operations: CopyBufferRegion") != std::string::npos);
    EXPECT(r.find("device fault: mock page fault") != std::string::npos);
    EXPECT(r.find("address 0x12340000 (precision 0xfff, type 2)") != std::string::npos);
    EXPECT(r.find("address 0x55660000") != std::string::npos);
    std::printf("device lost: GPU breadcrumbs + fault data, reported once ok\n");

    // CPU only (no VK_AMD_buffer_marker): the submitted list with its queue, progress UNKNOWN, never "completed".
    if (lose(label, scene, false, true, mock::Fault::data, r)) return 1;
    EXPECT(r.find("GPU progress UNKNOWN (no VK_AMD_buffer_marker); the last 1 of the 1 lists submitted") !=
           std::string::npos);
    EXPECT(r.find("\"list A\" on queue ") != std::string::npos);
    EXPECT(r.find("submitted, GPU progress UNKNOWN, 1 operations: CopyBufferRegion") != std::string::npos);
    EXPECT(r.find("completed") == std::string::npos && r.find("not started") == std::string::npos);
    EXPECT(r.find("device fault: mock page fault") != std::string::npos);
    std::printf("device lost: CPU only, progress UNKNOWN, fault data ok\n");

    // No fault information: the extension off (never queried), nothing to say, or a failing query.
    for (const auto& [enabled, fault, what] :
         {std::tuple{false, mock::Fault::data, "extension off"}, std::tuple{true, mock::Fault::empty, "no data"},
          std::tuple{true, mock::Fault::fail_counts, "counts query fails"},
          std::tuple{true, mock::Fault::fail_info, "info query fails"}}) {
        if (lose(label, scene, true, enabled, fault, r)) return 1;
        EXPECT(r.find("no device fault information (VK_EXT_device_fault)") != std::string::npos);
        EXPECT(r.find("device fault:") == std::string::npos && r.find("address 0x") == std::string::npos);
        std::printf("device lost: %s -> no device fault information ok\n", what);
    }
    return 0;
}

// Eight threads see the loss at once, through mark_device_lost and is_device_lost: one report, no deadlock.
int concurrent(Scene& scene) {
    const char* label = "concurrent";
    mock::reset(mock::Fault::data);
    mock::lost = true;  // the fault query is legal from here on
    std::FILE* out = temp_file();
    EXPECT(out);
    {
        VulkanContext context(mock::get_device, scene.d.device, true);
        EXPECT(context.loaded());
        context.set_report_output(out);
        context.configure_breadcrumbs(scene.d.memory_properties(), false, true);
        std::atomic<bool> go{false};
        std::vector<std::future<void>> threads;
        for (int i = 0; i < 8; ++i)
            threads.push_back(std::async(std::launch::async, [&, i] {
                while (!go) std::this_thread::yield();
                for (int k = 0; k < 100; ++k) {
                    if (i % 2) context.mark_device_lost();
                    else (void)context.is_device_lost();
                }
            }));
        go = true;
        for (auto& t : threads) EXPECT(t.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
        EXPECT(context.is_device_lost());
        EXPECT(mock::fault_calls == 2);  // one report: its two queries
    }
    const std::string r = read_all(out);
    std::fclose(out);
    EXPECT(count(r, "no command list was submitted") == 1 && count(r, "device fault: mock page fault") == 1);
    std::printf("device lost: eight threads at once, one report, no deadlock ok\n");
    return 0;
}

// Level 2 never reuses a slot whose list the GPU has not been seen to finish. The mock records no marker, so no
// list ever reads finished: after kSlots lists every slot is held, and the next list takes none and records no
// marker at all; its progress is unknown, not "0 completed".
int busy_slots(Scene& scene) {
    const char* label = "device lost: busy slots";
    mock::reset(mock::Fault::empty);
    VulkanContext context(mock::get_device, scene.d.device, true);
    EXPECT(context.loaded());
    EXPECT(context.configure_breadcrumbs(scene.d.memory_properties(), true, false));
    ResourceBinding src{}, dst{};
    src.buffer = scene.src;
    src.size = 64;
    dst.buffer = scene.dst;
    dst.size = 64;
    context.add_resource(1, src);
    context.add_resource(2, dst);
    VulkanQueue queue(context, scene.d.queue, scene.d.family);
    EXPECT(queue.ok());
    BreadcrumbRing* ring = context.breadcrumbs();
    EXPECT(ring && ring->per_op());
    const CommandList* pa = &scene.a;  // one operation
    for (std::uint32_t i = 0; i < BreadcrumbRing::kSlots; ++i) EXPECT(queue.execute({&pa, 1}) == ReplayResult::success);
    EXPECT(queue.idle());
    EXPECT(mock::markers == BreadcrumbRing::kSlots * 4u);  // begin, its progress clear, end and one per-op marker
    EXPECT(queue.execute({&pa, 1}) == ReplayResult::success);
    EXPECT(queue.idle());
    EXPECT(mock::markers == BreadcrumbRing::kSlots * 4u);  // no free slot: no marker at all, no slot taken
    const auto entries = ring->snapshot();
    EXPECT(!entries.empty() && !entries.back().progress_known && entries.back().completed_ops == 0 &&
           entries.back().no_slot && entries.back().state == BreadcrumbState::unknown);
    EXPECT(entries.size() == BreadcrumbRing::kSlots + 1);  // the 1024 unfinished lists and the one without a slot
    for (const auto& e : entries) EXPECT(e.completed_ops == 0 && e.state != BreadcrumbState::completed);
    std::printf("device lost: every slot held by an unfinished list: no per-op markers, progress unknown ok\n");
    return 0;
}

}  // namespace

int main() {
    const char* label = "vk_device_lost_test";
    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_device_lost_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    mock::real_get_device = d.device_proc;
    mock::real_submit = reinterpret_cast<PFN_vkQueueSubmit>(d.device_proc(d.device, "vkQueueSubmit"));
    EXPECT(mock::real_submit);

    Scene scene{d};
    void* src_ptr = nullptr;
    void* dst_ptr = nullptr;
    scene.src = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_ptr);
    scene.dst = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst_ptr);
    EXPECT(scene.src && scene.dst);
    EXPECT(scene.a.copy_buffer({2, 0, 1, 0, 16}) && scene.a.close());
    EXPECT(scene.b.copy_buffer({2, 16, 1, 16, 16}) && scene.b.close());

    if (scenarios(scene) || concurrent(scene) || busy_slots(scene)) return 1;
    const unsigned errors = d.validation_errors();
    std::printf("device lost: %u validation errors\n", errors);
    return errors ? 1 : 0;
}
