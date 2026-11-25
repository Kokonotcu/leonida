// Set while a guarded read of the title's pages runs (guarded_probe_hash and friends in shims/d3d12_vulkan_device.cpp):
// the exception loggers (main.cpp's crash logger, startup_trace's handler) leave those expected access violations to the
// read's __except without a report. A full stack report for each cost the texture sync tens of milliseconds a fault.
// A plain int: no thread_local with a destructor in gta6rt (its TLS callback hung the loader).
#pragma once

inline thread_local int leonida_quiet_av = 0;
