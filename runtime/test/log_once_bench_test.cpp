// Review I11: what a repeated stub() or log_once() costs, the path every call after the first takes. stub() runs on
// every RSSetViewports with more than one viewport (so on draws), and log_once() on every CheckFeatureSupport. Each
// took a global mutex and built a std::string, to find the text already printed. Measures stub() on one thread and
// on four at once, and CheckFeatureSupport(ARCHITECTURE) on the Vulkan device, in ns per call. A benchmark: it
// prints and passes; the review's numbers are in the I11 commit. Skips the CheckFeatureSupport part (77 only if
// nothing ran) without a Vulkan GPU. No shaders or game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/log_seen.h"

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace rt::d3d12 {
void stub(const char* method);
}

namespace {

double ns_per_call(std::chrono::steady_clock::time_point start, std::size_t calls) {
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / double(calls);
}

}  // namespace

int main() {
    // The site cache itself: a text and value are seen once remembered, another value or text is not.
    {
        static rt::log_seen::Sites sites;
        static const char text[] = "a site";
        static const char other[] = "a site";  // the same characters at another address: another site
        if (sites.seen(text, 1)) return 1;
        sites.remember(text, 1);
        if (!sites.seen(text, 1) || sites.seen(text, 2) || sites.seen(other, 1)) return 1;
    }
    constexpr std::size_t kCalls = 2'000'000;
    rt::d3d12::stub("ID3D12Bench::Repeated");  // the first call prints
    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kCalls; ++i) rt::d3d12::stub("ID3D12Bench::Repeated");
    const double one_thread = ns_per_call(start, kCalls);

    constexpr unsigned kThreads = 4;
    std::vector<std::thread> threads;
    start = std::chrono::steady_clock::now();
    for (unsigned t = 0; t < kThreads; ++t)
        threads.emplace_back([] {
            for (std::size_t i = 0; i < kCalls / kThreads; ++i) rt::d3d12::stub("ID3D12Bench::Repeated");
        });
    for (auto& thread : threads) thread.join();
    const double four_threads = ns_per_call(start, kCalls);
    std::printf("stub() already printed: %.1f ns per call on one thread, %.1f ns per call (wall) on %u threads\n",
                one_thread, four_threads, kThreads);

    vktest::Device d;
    if (d.initialize() != vktest::DeviceStatus::ready) {
        std::printf("no Vulkan GPU: CheckFeatureSupport not measured\n");
        return 0;
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    ID3D12Device* device = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device)))) return 1;
    D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture)))) return 1;
    constexpr std::size_t kQueries = 1'000'000;
    start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kQueries; ++i)
        device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture));
    const double query = ns_per_call(start, kQueries);
    std::printf("CheckFeatureSupport(ARCHITECTURE): %.1f ns per call\n", query);
    device->Release();
    std::printf("log_once_bench_test: OK\n");
    return 0;
}
