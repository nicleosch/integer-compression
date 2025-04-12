#include <iostream>
//---------------------------------------------------------------------------
#include "bootstrap/cli.hpp"
#include "common/Utils.hpp"
#include "core/BlockCompressor.hpp"
#include "core/ColumnCompressor.hpp"
#include "core/Compressor.hpp"
#include "extern/BtrBlocks.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
int main(int argc, char **argv) {
  using namespace compression;

  // Parse input arguments
  auto cli = bootstrap::parseCommandLine(argc, argv);

  // Size of an integer block.
  constexpr u16 kBlockSize = 256;
  // Size of a morsel.
  constexpr u16 kMorselSize = 1024;

  // Read integer column
  auto column = storage::Column::fromFile(cli.data.c_str(), cli.column, '|');

  // Choose the compressor
  std::unique_ptr<Compressor> compressor;
  if (cli.blocks) {
    column.padToMultipleOf(kDefaultDataBlockSize);
    compressor =
        std::make_unique<BlockCompressor<kDefaultDataBlockSize, kBlockSize>>(
            column);
  } else {
    column.padToMultipleOf(kMorselSize);
    compressor = std::make_unique<ColumnCompressor<kBlockSize>>(column);
  }
  u64 uncompressed_size = column.size() * sizeof(INTEGER);

  // Compression & Decompression
  CompressionStats stats;
  std::unique_ptr<compression::u8[]> compress_out;
  std::vector<compression::INTEGER> decompress_out;

  // BtrBlocks is treated special
  if (cli.algorithm == "btrblocks") {
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

    auto btr_stats = compressor.compress(input, compress_out);
    stats.compressed_size = btr_stats.total_data_size;
    stats.uncompressed_size = uncompressed_size;
    stats.compression_rate =
        static_cast<double>(stats.uncompressed_size) / stats.compressed_size;

    // decompress
    {
      utils::Timer timer;
      btrblocks::Chunk decompressed = compressor.decompress(compress_out);
    }
  } else {
    if (cli.algorithm == "uncompressed") {
      compressor->setScheme(CompressionSchemeType::kUncompressed);
    } else if (cli.algorithm == "bitpacking") {
      compressor->setScheme(CompressionSchemeType::kBitPacking);
    } else if (cli.algorithm == "delta") {
      compressor->setScheme(CompressionSchemeType::kDelta);
    } else if (cli.algorithm == "for") {
      compressor->setScheme(CompressionSchemeType::kFOR);
    } else if (cli.algorithm == "forn") {
      compressor->setScheme(CompressionSchemeType::kFORn);
    } else if (cli.algorithm == "rle") {
      compressor->setScheme(CompressionSchemeType::kRLE);
    } else if (cli.algorithm == "tinyblocks") {
      compressor->setScheme(CompressionSchemeType::kTinyBlocks);
    } else if (cli.algorithm == "lz4") {
      compressor->setScheme(CompressionSchemeType::kLZ4);
    } else {
      std::cout << "Unknown algorithm" << std::endl;
      return 1;
    }

    // compress
    stats = compressor->compress(compress_out);

    // decompress
    if (cli.morsel) {
      utils::Timer timer;
      compressor->decompress(compress_out.get());
    } else {
      utils::Timer timer;
      compressor->decompress(decompress_out, compress_out.get());
    }
  }

  std::cout << "Uncompressed Size: " << stats.uncompressed_size << " Byte"
            << std::endl;
  std::cout << "Compressed Size: " << stats.compressed_size << " Byte"
            << std::endl;
  std::cout << "Compression Rate: " << stats.compression_rate << std::endl;

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
        reinterpret_cast<const std::byte *>(compress_out.get()), kBlockSize,
        std::cout);

    std::cout << "Decompressed" << std::endl;
    compression::utils::hex_dump(
        reinterpret_cast<const std::byte *>(decompress_out.data()), kBlockSize,
        std::cout);
  }
}