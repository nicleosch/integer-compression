#include <iostream>
//---------------------------------------------------------------------------
#include "cli.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bootstrap {
//---------------------------------------------------------------------------
CLIOptions parseCommandLine(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: ./ic "
              << "--data <path> --column <index> --type <type> "
              << "--scheme <scheme> [--blocks] [--morsel] [--logging]\n";
    exit(1);
  }

  CLIOptions opts;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--data" && i + 1 < argc) {
      opts.data = argv[++i];
    } else if (arg == "--column" && i + 1 < argc) {
      opts.column = static_cast<uint16_t>(std::atoi(argv[++i]));
    } else if (arg == "--type" && i + 1 < argc) {
      opts.type = argv[++i];
      if (opts.type != "int" && opts.type != "bigint") {
        std::cerr << "Unsupported column type \"" << opts.type
                  << "\". Only \"int\" and \"bigint\" are supported.";
      }
    } else if (arg == "--scheme" && i + 1 < argc) {
      opts.scheme = argv[++i];
    } else if (arg == "--blocks") {
      opts.blocks = true;
    } else if (arg == "--morsel") {
      opts.morsel = true;
    } else if (arg == "--logging") {
      opts.logging = true;
    } else {
      std::cerr << "Usage: ./ic "
                << "--data <path> --column <index> --type <type> "
                << "--scheme <scheme> [--blocks] [--morsel] [--logging]\n";
      exit(1);
    }
  }

  return opts;
}
//---------------------------------------------------------------------------
} // namespace bootstrap
//---------------------------------------------------------------------------
} // namespace compression