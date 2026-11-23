// Link-time stand-in for the GUID data exports of the console's d3d12_xs.dll, so tiny_iid.dll gets real data imports
// of them. Under rt::Loader the shim registry answers the imports; the values here are zero, so if this stub were
// ever the one bound the test fails loudly. Test code, not game content.
#include <guiddef.h>

extern "C" {
__declspec(dllexport) extern const GUID IID_ID3D12Device = {};
__declspec(dllexport) extern const GUID IID_ID3D12ConsDmaCommandList = {};
__declspec(dllexport) extern const GUID DXGI_DEBUG_ALL = {};
}
