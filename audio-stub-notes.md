# Audio boot stubs: AcpHal.dll, xapu.dll, chad.dll

Branch `audio-stub`, based on `exe-day-combined` head `a9eafc9b3`. Not merged or deployed. The retail package was read
only; the `gta6rt --dry-run` check used a temporary overlay, and the package has no new overlay folder.

## Why these are not plain policy stubs

The executable imports 19 names from the three modules. All are by name; none is an ordinal import (checked against
the PE import table). The loader's unimplemented thunk returns 0 and writes nothing. That is sufficient for 14 of
them, but five measured call sites need a value the caller then uses:

| Call site (RVA) | Import | What the caller does with the result |
|---|---|---|
| `0x43aed98` | `AcpHalCreate` | Reads `[out]` as an object and calls `[obj]`, `[obj+0x10]`; shutdown calls `[obj+8]`, `[obj+0x28]`. |
| `0x43aec73` | `AcpHalAllocateShapeContexts` | Reads `config+0x38` and memsets `count * 64` bytes (`count` is `config+0x10`). |
| `0x43aed13` | `ApuAlloc` | Reads the out pointer and memsets `0x1700` and `0x200` chunks of it. |
| `0x43ad933` | `ApuCreateHeap` | Tests the signed return; a negative result reaches the game's `int3` at `0x43ad94e`. |
| `0x43aee69` | `XApuConnect` | Tests the signed return; a negative result reaches the same `int3`. |

The other 14 are silent no-ops. `ApuMapApuAddress` and `ApuIsVirtualAddressValid` are the two that must **not** invent
an APU address: the measured callers use the null/false branch safely.

## Served names and behaviour

All 19 are registered in `runtime/shims/audio_stub.cpp`; none is left on the unimplemented thunk.

| Import | Return | Writes / side effect |
|---|---|---|
| `AcpHalAllocateShapeContexts` | `S_OK` (0), or `E_INVALIDARG` / `E_OUTOFMEMORY` | Allocates `count * 64` zeroed host bytes (count read from `config+0x10`) and writes the pointer at `config+0x38`, but only after `VirtualQuery` proves that slot writable. The array is process-lifetime; `AcpHalReleaseShapeContexts` is a no-op. |
| `AcpHalCreate` | `S_OK` (0), or `E_INVALIDARG` | Writes a pointer to a static 4 KiB zeroed object whose first qword is a 64-slot vtable; every slot returns `S_OK`. The write happens only after the out pointer is proven writable. |
| `AcpHalReleaseShapeContexts` | `S_OK` (0) | No-op; accepts any state. |
| `ApuAlloc` | `S_OK` (0), `E_INVALIDARG`, or `E_OUTOFMEMORY` | `(out, reserved, size, alignment, flags)` inferred from the measured wrapper. Allocates zeroed `VirtualAlloc` memory and writes the pointer only after the out pointer is proven writable. `size` is capped at 256 MiB; alignment must be a power of two up to 64 KiB. |
| `ApuFree` | `S_OK` (0) | Frees only a pointer previously handed out by `ApuAlloc`; null, fake and unknown pointers are accepted and ignored. |
| `ApuCreateHeap` | `S_OK` (0) | No-op; accepts any size pair. |
| `ApuIsVirtualAddressValid` | `FALSE` (0) | No write. TRUE would make the engine store a fabricated APU address. |
| `ApuMapApuAddress` | `nullptr` | No write. Both measured callers check null before adding an offset. |
| `ApuMapVirtualAddress` | `0` | No write. Only reached when `ApuIsVirtualAddressValid` is TRUE, which it never is here. |
| `XApuConnect` | `S_OK` (0) | No write. The game's zeroed state block is left zero; all XApu shims accept the zero handle. |
| `XApuDequeueResult` | `0x80070490` (`HRESULT_FROM_WIN32(ERROR_NOT_FOUND)`) | No write. The caller treats a negative return as "skip"; returning 0 would make it parse an unwritten buffer. |
| `XApuDisconnect` | `S_OK` (0) | No-op; accepts null/fake handles. |
| `XApuEnqueueCommand` | `S_OK` (0) | The command is not read or written. |
| `XApuGetResultCount` | `0` | No write; the caller's dequeue loop is skipped. |
| `XDspActivate` | `0` (success) | No-op. |
| `XDspConnectWithMaximumStreamLimit` | `0` (success) | No-op; the six arguments are ignored. |
| `XDspDeactivate` | `0` | No-op. |
| `XDspDisconnect` | `0` | No-op. |
| `XDspSubmitCommand` | `0` (success) | The command is not read or written. |

`ApuAlloc` deliberately does **not** return a fake pointer or null. Its measured callers dereference and memset the
result, so null would crash at `0x43aed5b`; a fake pointer would crash in the same memset. It returns real, zeroed,
page-aligned host memory instead. The same reasoning applies to the shape-context array and the `AcpHalCreate` object.

## The game's `int3` at `0x43ad94e`

The audio init wrapper checks three results before continuing:

1. `ApuCreateHeap` (import at `0x43ad933`) must be non-negative.
2. The HAL init wrapper (which calls `AcpHalAllocateShapeContexts`, `AcpHalCreate` and `ApuAlloc`) must return true.
3. `XApuConnect` (import at `0x43aee69`) must be non-negative.

Any failure reaches `0x43ad94e` (`int3`). With this shim all three succeed on the measured path, and the buffers that
the wrapper memsets are real. The minimal non-negotiable set to avoid the `int3` is therefore five functions:

- `ApuCreateHeap` -> 0
- `AcpHalAllocateShapeContexts` -> 0 and a real zeroed `count * 64` array at `config+0x38`
- `AcpHalCreate` -> 0 and a dereferenceable object with a no-op vtable at `*out`
- `ApuAlloc` -> 0 and a real zeroed buffer at `*out`
- `XApuConnect` -> 0

The remaining 14 can return 0 without crashing on the measured paths, but serving them explicitly gives the trace
and the stricter no-write behaviour. The only residual `int3` risk is allocation failure (22 KiB for the contexts,
2.2 MiB for the pool) or `VirtualQuery` refusing the game's own `.data`/stack pointer; both are reported rather than
silently handed a bad pointer.

## Verification

- `audio_stub` CTest: **114 checks, 0 failures**. It resolves every name through `rt::shims::find`, calls each with
  valid, null and fake arguments, exercises the fake vtable, and replays the measured boot sequence including the
  2.2 MiB memset.
- `xgameruntime` family: **8/8 passed** (`xgameruntime`, user, gamesave, store, offline, offline HDR, both gamertag cases).
- `run_tiny_frame`: **passed** (3.35 s).
- `xmem` and `unimplemented_policy`: **passed**.
- `python -I tests/test_ctest_registration.py`: **4 tests, OK**.
- `gta6rt --dry-run` on the retail executable: **5 modules mapped, 19/19 audio imports bound to `shim`**, and the
  remaining unimplemented set is exactly `xg_xs.dll!XGCreateTextureComputer`, `xg_xs.dll!XGSuggestSwizzleMode` and
  `XCRDAPI.dll!DStorageGetFactory` (3 imports).
- `gta6rt` rebuilt with the shim linked.

## Signatures that are measured vs inferred

Measured from the executable's call sites (RVA `0x43ad8b0` audio init, `0x43aec40` HAL wrapper, `0x43ae840` release,
`0x144eaa840` XApu result loop, `0x140078120` / `0x1400785d0` address mapping):

- All 19 imports are by name.
- `AcpHalCreate` takes one out pointer; the object is read at `[out]`; vtable slots 0, 1, 2 and 5 are called.
- `AcpHalAllocateShapeContexts` takes one config pointer; count at `+0x10`, output pointer at `+0x38`.
- `ApuAlloc` takes `(out, 0, size, 0x800, flags)`; out is the first argument, not the return value.
- `ApuCreateHeap` takes two 32-bit words; the return is checked with a signed test.
- `XApuConnect` takes three pointers; the third is the game's state block. The wrappers test the return with `setns`.
- `ApuMapApuAddress` returns a pointer and both call sites test it for null.
- `ApuIsVirtualAddressValid` returns a byte in AL (declared `BOOL` here); `ApuMapVirtualAddress` returns a 32-bit APU
  address in EAX.
- `XApuDequeueResult` and `XDspSubmitCommand` have two-pointer wrappers; the exact meaning of each pointer is not
  proven.

Not proven:

- The full `AcpHalCreate` interface, the real object layout, and any vtable slot beyond 0/1/2/5. The shim answers all
  64 slots with `S_OK`, so an unknown slot cannot fault.
- The full shape-context structure beyond `+0x10` and `+0x38`; only this one caller was measured. The allocated array
  is zeroed, so no pointer field starts with a stale value.
- The exact `ApuFree` signature (no direct call site in the executable); the shim reads only the first pointer.
- The exact `XDspConnectWithMaximumStreamLimit` parameter names; six argument positions are inferred from the wrapper.
- The real return codes: they are treated as signed 32-bit HRESULT-like values because every measured caller uses
  `js`/`setns`. `E_INVALIDARG`, `E_OUTOFMEMORY`, `E_FAIL` conventions are therefore chosen, not measured.
- `XApuDequeueResult`'s no-work HRESULT is a chosen value (`0x80070490`), not a measured one. It is never reached while
  `XApuGetResultCount` returns 0.
- Ordinal aliases: none are registered. The executable's import table names every function, so an ordinal alias would
  be dead code for this title.

## Follow-up

`docs/console_libraries.md` still lists `AcpHal`/`chad`/`xapu` as unhandled; that table is now stale and should be
updated separately. The shape-context array is intentionally leaked for process lifetime (about 22 KiB at the measured
count), and `ApuAlloc` blocks are tracked only until `ApuFree` supplies the exact base pointer.
