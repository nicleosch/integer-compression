#include <iostream>
//---------------------------------------------------------------------------
#include "cli.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace bootstrap {
//---------------------------------------------------------------------------
CLIOptions parseCommandLine(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <path_to_data> <column> <algorithm> <datablocks> <morsel> "
                 "<logging>"
              << std::endl;
    exit(1);
  }

  CLIOptions opts;
  opts.data = argv[1];
  opts.column = std::stoi(argv[2]);
  opts.algorithm = argv[3];
  opts.blocks = std::stoi(argv[4]);
  opts.morsel = std::stoi(argv[5]);
  opts.logging = std::stoi(argv[6]);

  return opts;
}
//---------------------------------------------------------------------------
} // namespace bootstrap
//---------------------------------------------------------------------------
} // namespace compression