// Tiny fixture DLL that imports GUID data from d3d12_xs.dll, the way a title linked against the console runtime's
// import library gets its IIDs: the import address table slot holds the address of the GUID, not of a function. Each
// export returns the address its import resolved to, so pe_loader_test can read the 16 bytes the title would read.
// Test code, not game content.
#include <guiddef.h>

extern "C" {
__declspec(dllimport) extern const GUID IID_ID3D12Device;              // a standard interface id
__declspec(dllimport) extern const GUID IID_ID3D12ConsDmaCommandList;  // a console-only one
__declspec(dllimport) extern const GUID DXGI_DEBUG_ALL;                // a GUID that is not an interface id

__declspec(dllexport) const GUID* __cdecl tiny_iid_device() { return &IID_ID3D12Device; }
__declspec(dllexport) const GUID* __cdecl tiny_iid_dma_list() { return &IID_ID3D12ConsDmaCommandList; }
__declspec(dllexport) const GUID* __cdecl tiny_iid_debug_all() { return &DXGI_DEBUG_ALL; }
}
