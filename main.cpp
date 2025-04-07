#include <iostream>
//---------------------------------------------------------------------------
#include "bootstrap/cli.hpp"
#include "common/Utils.hpp"
#include "core/Compressor.hpp"
#include "core/Decompressor.hpp"
#include "extern/BtrBlocks.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
int main(int argc, char **argv) {
  using namespace compression;

  // Parse input arguments
  auto cli = bootstrap::parseCommandLine(argc, argv);

  // Size of an integer block.
  constexpr u16 block_size = 64;
  // Size of a morsel.
  constexpr u16 morsel_size = 1024;

  // Read integer column
  auto column = storage::Column::fromFile(cli.data.c_str(), cli.column, '|');
  column.padToMultipleOf(morsel_size);

  u64 uncompressed_size = column.size() * sizeof(INTEGER);
  u64 compressed_size;

  // Compression & Decompression
  std::unique_ptr<compression::u8[]> compress_out;
  std::vector<compression::INTEGER> decompress_out;
  if (cli.algorithm == "uncompressed") {
    compressed_size = compressor::compressUncompressed(column, compress_out);

    if (cli.morsel) {
      utils::Timer timer;
      decompressor::morsel::decompressUncompressed<morsel_size>(
          compress_out.get(), column.size());
    } else {
      utils::Timer timer;
      decompressor::decompressUncompressed(decompress_out, column.size(),
                                           compress_out.get());
    }
  } else if (cli.algorithm == "for") {
    compressed_size = compressor::compressFOR(column, compress_out);

    if (cli.morsel) {
      utils::Timer timer;
      decompressor::morsel::decompressFOR<morsel_size>(compress_out.get(),
                                                       column.size());
    } else {
      utils::Timer timer;
      decompressor::decompressFOR(decompress_out, column.size(),
                                  compress_out.get());
    }
  } else if (cli.algorithm == "forn") {
    compressed_size =
        compressor::compressFORn<block_size>(column, compress_out);

    if (cli.morsel) {
      utils::Timer timer;
      decompressor::morsel::decompressFORn<morsel_size, block_size>(
          compress_out.get(), column.size());
    } else {
      utils::Timer timer;
      decompressor::decompressFORn<block_size>(decompress_out, column.size(),
                                               compress_out.get());
    }
  } else if (cli.algorithm == "tinyblocks") {
    compressed_size =
        compressor::compressTinyBlocks<block_size>(column, compress_out);

    if (cli.morsel) {
      utils::Timer timer;
      decompressor::morsel::decompressTinyBlocks<morsel_size, block_size>(
          compress_out.get(), column.size());
    } else {
      utils::Timer timer;
      decompressor::decompressTinyBlocks<block_size>(
          decompress_out, column.size(), compress_out.get());
    }
  } else if (cli.algorithm == "lz4") {
    compressed_size = compressor::compressLZ4(column, compress_out);

    {
      utils::Timer timer;
      decompressor::decompressLZ4(decompress_out, column.size(),
                                  compress_out.get(), compressed_size);
    }
  } else if (cli.algorithm == "btrblocks") {
    // setup
    btrblocks::BtrBlocksConfig::configure(
        [&](btrblocks::BtrBlocksConfig &config) {
          if (argc > 1) {
            auto max_depth = 5;
            config.integers.max_cascade_depth = max_depth;
          }
          config.integers.schemes.enableAll();
        });

    btrblocks::Vector<INTEGER> vec{column.size()};
    for (auto i = 0; i < vec.count; ++i) {
      vec[i] = column.data()[i];
    }

    // compress
    btrblocks::Relation to_compress;
    to_compress.addColumn({"ints", std::move(vec)});
    btrblocks::Range range(0, to_compress.tuple_count);
    btrblocks::Chunk input = to_compress.getChunk({range}, 0);
    btrblocks::Datablock compressor(to_compress);

    auto stats = compressor.compress(input, compress_out);
    compressed_size = stats.total_data_size;

    // decompress
    {
      utils::Timer timer;
      btrblocks::Chunk decompressed = compressor.decompress(compress_out);
    }

  } else {
    std::cout << "Unknown algorithm" << std::endl;
    return 1;
  }

  std::cout << "Before compression: " << uncompressed_size << " Byte"
            << std::endl;
  std::cout << "After compression : " << compressed_size << " Byte"
            << std::endl;

  // Log (de)compressed data.
  if (cli.logging) {
    if (cli.morsel || cli.algorithm == "btrblocks") {
      std::cout << "Logging not allowed when decompressing into morsels or "
                   "when using btrblocks."
                << std::endl;
      return 1;
    }

    std::cout << "Compressed" << std::endl;
    compression::utils::hex_dump(
        reinterpret_cast<const std::byte *>(compress_out.get()), block_size,
        std::cout);

    std::cout << "Decompressed" << std::endl;
    compression::utils::hex_dump(
        reinterpret_cast<const std::byte *>(decompress_out.data()), block_size,
        std::cout);
  }
}