#pragma once
//---------------------------------------------------------------------------
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bootstrap {
//---------------------------------------------------------------------------
struct CLIOptions {
  /// Path to the data.
  string data;
  /// The column index.
  u16 column;
  /// The type of the column.
  string type;
  /// Compression scheme to be used.
  string scheme;
  /// Size of a small integer block.
  u16 block_size;
  /// Whether the column should be divided into DataBlocks for (de)compression.
  bool blocks = false;
  /// Whether to decompress into morsels.
  bool morsel = false;
  /// Whether to log the first block of (de)compressed data.
  bool logging = false;
};
//---------------------------------------------------------------------------
CLIOptions parseCommandLine(int argc, char **argv);
//---------------------------------------------------------------------------
} // namespace bootstrap
//---------------------------------------------------------------------------
} // namespace compression