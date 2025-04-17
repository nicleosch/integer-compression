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
using namespace compression;
//---------------------------------------------------------------------------
template <typename DataType, u16 kBlockSize>
int compressionLogic(bootstrap::CLIOptions &cli) {
  // Size of a morsel.
  constexpr u16 kMorselSize = 1024;

  // Read integer column
  auto column =
      storage::Column<DataType>::fromFile(cli.data.c_str(), cli.column, '|');

  // Settings for compressing the already compressed data.
  Phase2CompressionSettings settings;

  // Choose the compressor
  std::unique_ptr<Compressor<DataType>> compressor;
  if (cli.blocks) {
    column.padToMultipleOf(kDefaultDataBlockSize);
    compressor = std::make_unique<
        BlockCompressor<DataType, kDefaultDataBlockSize, kBlockSize>>(column);
  } else {
    column.padToMultipleOf(kMorselSize);
    if (cli.p2_scheme.size() == 0) { // no additional compression
      compressor =
          std::make_unique<ColumnCompressor<DataType, kBlockSize>>(column);
    } else { // additional compression on compressed data
      CompressionSchemeType scheme;
      if (cli.p2_scheme == "lz4") {
        scheme = CompressionSchemeType::kLZ4;
      } else {
        std::cerr << "Scheme \"" << cli.p2_scheme
                  << "\" not supported for Phase2-Compression." << std::endl;
        exit(1);
      }
      settings = {scheme, cli.p2_header, cli.p2_payload};
      compressor = std::make_unique<ColumnCompressor<DataType, kBlockSize>>(
          column, &settings);
    }
  }
  u64 uncompressed_size = column.size() * sizeof(DataType);

  // Compression & Decompression
  CompressionStats stats;
  std::unique_ptr<u8[]> compress_out;
  std::vector<DataType> decompress_out;

  // BtrBlocks is treated special
  if (cli.scheme == "btrblocks") {
    btrblocks::BtrBlocksConfig::configure(
        [&](btrblocks::BtrBlocksConfig &config) {
          auto max_depth = 3;
          config.integers.max_cascade_depth = max_depth;
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
    stats.details = {0, static_cast<u32>(btr_stats.total_data_size)};
    stats.compression_rate =
        static_cast<double>(stats.uncompressed_size) / stats.compressed_size;

    // decompress
    {
      utils::Timer timer;
      btrblocks::Chunk decompressed = compressor.decompress(compress_out);
    }
  } else {
    if (cli.scheme == "uncompressed") {
      compressor->setScheme(CompressionSchemeType::kUncompressed);
    } else if (cli.scheme == "bitpacking") {
      compressor->setScheme(CompressionSchemeType::kBitPacking);
    } else if (cli.scheme == "delta") {
      compressor->setScheme(CompressionSchemeType::kDelta);
    } else if (cli.scheme == "for") {
      compressor->setScheme(CompressionSchemeType::kFOR);
    } else if (cli.scheme == "forn") {
      compressor->setScheme(CompressionSchemeType::kFORn);
    } else if (cli.scheme == "rle") {
      compressor->setScheme(CompressionSchemeType::kRLE);
    } else if (cli.scheme == "tinyblocks") {
      compressor->setScheme(CompressionSchemeType::kTinyBlocks);
    } else if (cli.scheme == "lz4") {
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

  std::cout << "Uncompressed Size: " << stats.uncompressed_size << " Bytes"
            << std::endl;
  std::cout << "Compressed Size: " << stats.compressed_size << " Bytes"
            << std::endl;
  std::cout << "Compressed Header Size: " << stats.details.header_size
            << " Bytes" << std::endl;
  std::cout << "Compressed Payload Size: " << stats.details.payload_size
            << " Bytes" << std::endl;
  std::cout << "Compression Rate: " << stats.compression_rate << std::endl;

  // Log (de)compressed data.
  if (cli.logging) {
    if (cli.morsel || cli.scheme == "btrblocks") {
      std::cout << "Logging not allowed when decompressing into morsels or "
                   "when using btrblocks."
                << std::endl;
      return 1;
    }

    std::cout << "Compressed" << std::endl;
    utils::hex_dump(reinterpret_cast<const std::byte *>(compress_out.get()),
                    kBlockSize, std::cout);

    std::cout << "Decompressed" << std::endl;
    utils::hex_dump(reinterpret_cast<const std::byte *>(decompress_out.data()),
                    kBlockSize, std::cout);
  }

  return 0;
}
//---------------------------------------------------------------------------
int main(int argc, char **argv) {

  // Parse input arguments
  auto cli = bootstrap::parseCommandLine(argc, argv);

  if (cli.type == "int") {
    if (cli.block_size == 64)
      return compressionLogic<INTEGER, 64>(cli);
    else if (cli.block_size == 128)
      return compressionLogic<INTEGER, 128>(cli);
    else if (cli.block_size == 256)
      return compressionLogic<INTEGER, 256>(cli);
    else if (cli.block_size == 512)
      return compressionLogic<INTEGER, 512>(cli);
    else
      std::cerr
          << "Unsupported size: \"" << cli.block_size
          << "\". Only \"64\", \"128\", \"256\" and \"512\" are supported."
          << std::endl;
  } else if (cli.type == "bigint") {
    if (cli.block_size == 64)
      return compressionLogic<BIGINT, 64>(cli);
    else if (cli.block_size == 128)
      return compressionLogic<BIGINT, 128>(cli);
    else if (cli.block_size == 256)
      return compressionLogic<BIGINT, 256>(cli);
    else if (cli.block_size == 512)
      return compressionLogic<BIGINT, 512>(cli);
    else
      std::cerr
          << "Unsupported size: \"" << cli.block_size
          << "\". Only \"64\", \"128\", \"256\" and \"512\" are supported."
          << std::endl;
  } else {
    std::cerr << "Unsupported column type \"" << cli.type
              << "\". Only \"int\" and \"bigint\" are supported." << std::endl;
    return 1;
  }
}