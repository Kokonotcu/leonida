// BCPack decoder: lossless packing of BC1-BC7 block streams, as the console's
// DirectStorage hardware unpacks them. Decode only; BC2 is left out (no title
// data uses it).
//
// Ported from VIctory (github.com/marvisak/victory,
// crates/victory-formats/src/bcpack.rs, GPL-3.0). The format's constant
// tables (bcpack-tables.bin, 77,982 bytes) are not part of this source and are
// supplied separately at run time; see parse_tables.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace leonida::bcpack_victory {

// Bytes of the table file.
inline constexpr std::size_t kTablesSize = 77982;

// The format's constant tables: opaque, defined in the .cpp.
struct Tables;

struct TablesDeleter {
    void operator()(Tables* t) const noexcept;
};
using TablesPtr = std::unique_ptr<Tables, TablesDeleter>;

// Parses the table file (VIctory's Tables::parse): 128 x 9 u16 model
// frequencies, the rank tables, 512 u16 BC7 control words, then 255
// dictionary entries per format. Returns null and sets *error (if error is
// not null) on failure. Never throws.
TablesPtr parse_tables(std::span<const std::uint8_t> bytes, const char** error);

// Same as TablesDeleter; for callers that hold a raw pointer.
void free_tables(Tables* t) noexcept;

// Unpacks `src` (BC number `bc`: 1, 3, 4, 5, 6 or 7) into `dst`, whose
// length gives the block count (VIctory's decode). The stream carries no size:
// bytes after the last one the lanes need are ignored, a stream that runs out
// is an error. Returns false and sets *error (if not null) on failure; dst may
// then hold partial output. Never throws. Safe to call concurrently with one
// shared const Tables.
bool decode(unsigned bc, std::span<const std::uint8_t> src, std::span<std::uint8_t> dst,
            const Tables& tables, const char** error);

}  // namespace leonida::bcpack_victory
