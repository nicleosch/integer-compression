#pragma once
//---------------------------------------------------------------------------
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bootstrap {
//---------------------------------------------------------------------------
struct CLIOptions {
  //===--------------------------------------------------------------------===//
  // Input Data Configuration
  //===--------------------------------------------------------------------===//
  /// Path to the data.
  string data;
  /// The column index.
  u16 column;
  /// The type of the column (int or bigint).
  string type;
  //===--------------------------------------------------------------------===//
  // Compression Configuration - Phase 1
  //===--------------------------------------------------------------------===//
  /// Compression scheme to be used.
  string scheme;
  /// Size of a small integer block.
  u16 block_size;
  //===--------------------------------------------------------------------===//
  // Compression Configuration - Phase 2 (Optional)
  //===--------------------------------------------------------------------===//
  /// The compression scheme to use on the compressed data (Phase 2), if given.
  string p2_scheme;
  /// Whether to apply the Phase 2 scheme only to the header.
  bool p2_header = false;
  /// Whether to apply the Phase 2 scheme only to the payload.
  bool p2_payload = false;
  //===--------------------------------------------------------------------===//
  // Runtime Options
  //===--------------------------------------------------------------------===//
  /// Whether to decompress into morsels.
  bool morsel = false;
  /// Whether the column should be divided into DataBlocks for (de)compression.
  bool blocks = false;
  /// Whether to log the first block of (de)compressed data.
  bool logging = false;
};
//---------------------------------------------------------------------------
CLIOptions parseCommandLine(int argc, char **argv);
//---------------------------------------------------------------------------
} // namespace bootstrap
//---------------------------------------------------------------------------
} // namespace compression