#include <iostream>
//---------------------------------------------------------------------------
#include "cli.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bootstrap {
//---------------------------------------------------------------------------
CLIOptions parseCommandLine(int argc, char **argv) {
  string usage = "Usage: ./ic --data <path> --column <index> --type <type> "
                 "--delimiter <delimiter>"
                 "--scheme <scheme> --size <block_size> --depth <depth> "
                 "--p2scheme <scheme> "
                 "[--p2header] [--p2payload] [--blocks] [--morsel] [--logging]";
  CLIOptions opts;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--data" && i + 1 < argc) {
      opts.data = argv[++i];
    } else if (arg == "--column" && i + 1 < argc) {
      opts.column = static_cast<u16>(std::atoi(argv[++i]));
    } else if (arg == "--type" && i + 1 < argc) {
      opts.type = argv[++i];
      if (opts.type != "int" && opts.type != "bigint") {
        std::cerr << "Unsupported column type \"" << opts.type
                  << "\". Only \"int\" and \"bigint\" are supported.";
      }
    } else if (arg == "--delimiter" && i + 1 < argc) {
      opts.delimiter = *argv[++i];
    } else if (arg == "--scheme" && i + 1 < argc) {
      opts.scheme = argv[++i];
    } else if (arg == "--size" && i + 1 < argc) {
      opts.block_size = static_cast<u16>(std::atoi(argv[++i]));
    } else if (arg == "--depth" && i + 1 < argc) {
      opts.depth = static_cast<u8>(std::atoi(argv[++i]));
    } else if (arg == "--p2scheme" && i + 1 < argc) {
      opts.p2_scheme = argv[++i];
    } else if (arg == "--p2header") {
      opts.p2_header = true;
    } else if (arg == "--p2payload") {
      opts.p2_payload = true;
    } else if (arg == "--blocks") {
      opts.blocks = true;
    } else if (arg == "--morsel") {
      opts.morsel = true;
    } else if (arg == "--logging") {
      opts.logging = true;
    } else {
      std::cerr << usage << std::endl;
      exit(1);
    }
  }

  if (opts.data.empty() || opts.type.empty() || opts.scheme.empty()) {
    std::cerr << usage << std::endl;
    exit(1);
  }

  return opts;
}
//---------------------------------------------------------------------------
} // namespace bootstrap
//---------------------------------------------------------------------------
} // namespace compression