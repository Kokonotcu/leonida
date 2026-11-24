#include <windows.h>
#include <ApiRefState.h>

#include <cstdio>
#include <atomic>
#include <functional>
#include <thread>

static int failures = 0;

// Every thread of the test runs above normal priority (see main): a new thread starts at normal priority.
static void RaisePriority()
{
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
}

// One contender thread for all the race rounds: a round posts its contender's body and waits for it to finish. A new
// thread per round started at normal priority and waited a scheduler quantum to run whenever every core was busy.
class Contender
{
public:
    Contender() : thread_([this] { Loop(); }) {}

    ~Contender()
    {
        Post(nullptr);
        thread_.join();
    }

    // Runs `body` on the contender thread; Finish() waits for it.
    void Post(std::function<void()> body)
    {
        body_ = std::move(body);
        posted_.fetch_add(1, std::memory_order_release);
        posted_.notify_one();
    }

    void Finish()
    {
        const int target = posted_.load(std::memory_order_relaxed);
        for (int seen = finished_.load(std::memory_order_acquire); seen != target;
             seen = finished_.load(std::memory_order_acquire))
            finished_.wait(seen, std::memory_order_acquire);
    }

private:
    void Loop()
    {
        RaisePriority();
        for (int seen = 0;;)
        {
            posted_.wait(seen, std::memory_order_acquire);
            seen = posted_.load(std::memory_order_acquire);
            if (!body_)
                return;
            body_();
            finished_.store(seen, std::memory_order_release);
            finished_.notify_one();
        }
    }

    std::function<void()> body_;
    std::atomic<int> posted_{ 0 };
    std::atomic<int> finished_{ 0 };
    std::thread thread_;
};

void Check(bool condition, const char* name)
{
    std::printf("%s %s\n",
        condition ? "PASS" : "FAIL", name);

    if (!condition)
        ++failures;
}

struct RaceContext
{
    ApiRefState state;
    HANDLE rundownEntered = nullptr;
    HANDLE allowRundownExit = nullptr;
};

DWORD WINAPI RundownThread(void* arg)
{
    auto& ctx = *static_cast<RaceContext*>(arg);

    if (ctx.state.AddRef() != 0)
        return 1;

    auto finalRelease = ctx.state.Release();

    if (!finalRelease.startRundown ||
        finalRelease.remaining != 0)
        return 2;

    // Simulate an internal AddRef during RundownObject().
    if (ctx.state.AddRef() != 0)
        return 3;

    SetEvent(ctx.rundownEntered);

    if (WaitForSingleObject(
        ctx.allowRundownExit, 3000) != WAIT_OBJECT_0)
        return 4;

    auto internalRelease = ctx.state.Release();

    if (internalRelease.startRundown ||
        internalRelease.remaining != 0)
        return 5;

    return 0;
}

void CheckPinWinsBeforeFinalRelease()
{
    constexpr int rounds = 200;
    int failuresLocal = 0;
    Contender contender;

    for (int i = 0; i < rounds; ++i)
    {
        ApiRefState state;
        state.AddRef();

        std::atomic<bool> pinFinished{ false };
        std::atomic<bool> allowPinRelease{ false };

        bool acquired = false;
        ApiRefState::ReleaseResult pinRelease{ 0, false };

        // The two handshakes only order the steps (the pin comes first by construction), so they block rather than
        // spin: a yield loop costs a scheduler quantum per wait when every core is busy (ctest -j), which made the
        // 1200 rounds take minutes.
        contender.Post([&]
        {
            acquired = state.TryAddRef();

            pinFinished.store(true, std::memory_order_release);
            pinFinished.notify_one();

            if (acquired)
            {
                allowPinRelease.wait(false, std::memory_order_acquire);

                pinRelease = state.Release();
            }
        });

        pinFinished.wait(false, std::memory_order_acquire);

        const auto ownerRelease = state.Release();

        bool valid =
            acquired &&
            ownerRelease.remaining == 1 &&
            !ownerRelease.startRundown &&
            state.RefCount() == 1 &&
            !state.IsDying();

        allowPinRelease.store(true, std::memory_order_release);
        allowPinRelease.notify_one();

        contender.Finish();

        valid =
            valid &&
            pinRelease.remaining == 0 &&
            pinRelease.startRundown &&
            state.RefCount() == 0 &&
            state.IsDying() &&
            !state.TryAddRef();

        if (!valid)
        {
            ++failuresLocal;
        }
    }

    Check(failuresLocal == 0,
          "pin acquired before final owner Release stays valid");

    std::printf(
        "forced pin-first rounds: %d, failures: %d\n",
        rounds, failuresLocal);
}
void CheckFinalReleaseRace()
{
    constexpr int rounds = 1000;
    int badRundown = 0;
    int badFinalState = 0;
    int successfulPins = 0;
    Contender contender;

    for (int i = 0; i < rounds; ++i)
    {
        ApiRefState state;
        state.AddRef();

        std::atomic<bool> ready{ false };
        std::atomic<bool> go{ false };
        std::atomic<int> rundownCount{ 0 };
        std::atomic<bool> pinned{ false };

        contender.Post([&]
        {
            ready.store(true, std::memory_order_release);
            ready.notify_one();

            // Spin a bounded while so the pin races the final Release closely when the thread runs, then block: an
            // unbounded yield loop cost a scheduler quantum per round when every core was busy (ctest -j).
            for (int spin = 0; spin < (1 << 16) && !go.load(std::memory_order_acquire); ++spin)
                YieldProcessor();
            go.wait(false, std::memory_order_acquire);

            if (state.TryAddRef())
            {
                pinned.store(true, std::memory_order_relaxed);

                if (state.Release().startRundown)
                    rundownCount.fetch_add(1);
            }
        });

        ready.wait(false, std::memory_order_acquire);

        go.store(true, std::memory_order_release);
        go.notify_one();

        if (state.Release().startRundown)
            rundownCount.fetch_add(1);

        contender.Finish();

        if (pinned.load(std::memory_order_relaxed))
            ++successfulPins;

        if (rundownCount.load() != 1)
            ++badRundown;

        bool unexpectedPin = state.TryAddRef();

        if (unexpectedPin)
            state.Release();

        if (state.RefCount() != 0 ||
            !state.IsDying() ||
            unexpectedPin)
        {
            ++badFinalState;
        }
    }

    Check(badRundown == 0,
          "racing final Release starts rundown exactly once");

    Check(badFinalState == 0,
          "racing final Release preserves terminal state");

    std::printf(
        "race rounds: %d, successful pins: %d, rejected pins: %d\n",
        rounds, successfulPins, rounds - successfulPins);
}
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    // The 1200 race rounds hand off between two threads each round. With every core busy (ctest -j) each handoff waited
    // a scheduler quantum behind the other tests' threads, and the run took minutes (160 s measured with 20 CPU-bound
    // processes on 16 logical cores, 0.2 s idle); the races themselves need no particular timing, so the test's
    // threads run above normal priority and get the core when they are woken.
    RaisePriority();

    std::puts("api_ref_state_test");

    ApiRefState state;

    Check(!state.TryAddRef(),
          "cannot pin zero-ref object");

    Check(state.AddRef() == 0,
          "initial AddRef");

    Check(state.TryAddRef(),
          "can pin live object");

    Check(state.RefCount() == 2,
          "two references held");

    auto first = state.Release();

    Check(first.remaining == 1 &&
          !first.startRundown,
          "non-final Release");

    auto last = state.Release();

    Check(last.remaining == 0 &&
          last.startRundown,
          "final Release starts rundown");

    Check(state.IsDying(),
          "dying marker is permanent");

    Check(!state.TryAddRef(),
          "cannot pin after final Release");

    RaceContext race;

    race.rundownEntered =
        CreateEventW(nullptr, TRUE, FALSE, nullptr);

    race.allowRundownExit =
        CreateEventW(nullptr, TRUE, FALSE, nullptr);

    if (!race.rundownEntered || !race.allowRundownExit)
        return 2;

    HANDLE thread = CreateThread(
        nullptr, 0, RundownThread, &race, 0, nullptr);

    if (!thread)
        return 3;

    DWORD entered = WaitForSingleObject(
        race.rundownEntered, 3000);

    Check(entered == WAIT_OBJECT_0,
          "rundown thread reached checkpoint");

    if (entered == WAIT_OBJECT_0)
    {
        Check(race.state.IsDying(),
              "dying bit survives internal AddRef");

        Check(race.state.RefCount() == 1,
              "temporary internal reference exists");

        bool rejected = true;

        for (int i = 0; i < 10000; ++i)
        {
            if (race.state.TryAddRef())
            {
                rejected = false;
                race.state.Release();
                break;
            }
        }

        Check(rejected,
              "concurrent pins rejected during rundown");
    }

    SetEvent(race.allowRundownExit);

    DWORD joined = WaitForSingleObject(thread, 3000);

    Check(joined == WAIT_OBJECT_0,
          "rundown thread completed");

    if (joined == WAIT_OBJECT_0)
    {
        DWORD exitCode = 1;
        GetExitCodeThread(thread, &exitCode);

        Check(exitCode == 0,
              "internal Release did not restart rundown");
    }

    CloseHandle(thread);
    CloseHandle(race.rundownEntered);
    CloseHandle(race.allowRundownExit);

    CheckPinWinsBeforeFinalRelease();
    CheckFinalReleaseRace();

    std::printf("RESULT: %s (%d failures)\n",
        failures == 0 ? "PASS" : "FAIL", failures);

    return failures == 0 ? 0 : 1;
}
