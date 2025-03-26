#pragma once
//---------------------------------------------------------------------------
#include "common/Units.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bootstrap {
//---------------------------------------------------------------------------
struct CLIOptions {
  /// Path to the data.
  string data;
  /// Integer column.
  u16 column;
  /// Compression algorithm.
  string algorithm;
  /// Whether to decompress into morsels.
  bool morsel;
  /// Whether to log the first block of (de)compressed data.
  bool logging;
};
//---------------------------------------------------------------------------
CLIOptions parseCommandLine(int argc, char **argv);
//---------------------------------------------------------------------------
} // namespace bootstrap
//---------------------------------------------------------------------------
} // namespace compression