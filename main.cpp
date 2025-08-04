#include <iostream>
//---------------------------------------------------------------------------
#include "bootstrap/cli.hpp"
#include "common/Utils.hpp"
#include "core/BlockCompressor.hpp"
#include "core/ColumnCompressor.hpp"
#include "core/Compressor.hpp"
#include "extern/BtrBlocks.hpp"
#include "extern/FastLanes.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
template <typename DataType, u16 kBlockSize>
int compressionLogic(bootstrap::CLIOptions &cli) {
  // Size of a morsel.
  constexpr u16 kMorselSize = 1024;

  // Read integer column
  auto column = storage::Column<DataType>::fromFile(cli.data.c_str(),
                                                    cli.column, cli.delimiter);

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
      } else if (cli.p2_scheme == "zstd") {
        scheme = CompressionSchemeType::kZstd;
      } else if (cli.p2_scheme == "snappy") {
        scheme = CompressionSchemeType::kSnappy;
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

  utils::Timer timer;
  // BtrBlocks is treated special
  if (cli.scheme == "btrblocks") {
    // init
    btrblocks::BtrBlocksConfig::configure(
        [&](btrblocks::BtrBlocksConfig &config) {
          auto max_depth = cli.depth;
          config.integers.max_cascade_depth = max_depth;
          config.integers.schemes.enableAll();
        });
    btrblocks::Vector<INTEGER> vec{column.size()};
    for (auto i = 0; i < vec.count; ++i) {
      vec[i] = column.data()[i];
    }
    std::unordered_map<u8, u64> scheme_occurences{};
    //---------------------------------------------------------------------------
    // compress
    btrblocks::Relation relation;
    relation.addColumn({"ints", std::move(vec)});
    btrblocks::Datablock datablock(relation);

    vector<btrblocks::Range> ranges =
        relation.getRanges(btrblocks::SplitStrategy::SEQUENTIAL, 9999999);
    vector<btrblocks::BytesArray> compressed_chunks;
    compressed_chunks.resize(ranges.size());

    for (u32 chunk_i = 0; chunk_i < ranges.size(); chunk_i++) {
      auto chunk = relation.getChunk(ranges, chunk_i);
      auto db_meta = datablock.compress(chunk, compressed_chunks[chunk_i]);

      stats.uncompressed_size += chunk.size_bytes();
      stats.compressed_size += db_meta.total_data_size;

      auto &scheme = db_meta.used_compression_schemes[0];

      scheme_occurences[scheme] += 1;
    }

    stats.compression_rate =
        static_cast<double>(stats.uncompressed_size) / stats.compressed_size;
    stats.details = {0, static_cast<u32>(stats.compressed_size)};
    //---------------------------------------------------------------------------
    // decompress
    timer.start();
    for (u32 chunk_i = 0; chunk_i < ranges.size(); chunk_i++) {
      btrblocks::Chunk decompressed =
          datablock.decompress(compressed_chunks[chunk_i]);
    }
    timer.end();
  } else if (cli.scheme == "fastlanes") {
    using namespace fastlanes;
    const path fls_path = path{cli.data.c_str()} / "fastlanes.fls";

    Connection conn;
    auto fls_reader = conn.reset().read_fls(fls_path);
    auto table = fls_reader->materialize();
    auto n_row_groups = table->get_n_rowgroups();

    // Thrash the CPU caches to read compressed data from memory.
    utils::thrashCPUCaches();

    auto start = std::chrono::high_resolution_clock::now();
    for (u32 i{0}; i < n_row_groups; ++i) {
      auto first_rowgroup_reader = fls_reader->get_rowgroup_reader(i);
      for (const auto &col :
           first_rowgroup_reader->get_descriptor().m_column_descriptors) {
        std::cout << std::endl;
        for (const auto &token : col->encoding_rpn->operator_tokens) {
        }
      }
      for (u32 vec_idx{0};
           vec_idx < first_rowgroup_reader->get_descriptor().m_n_vec;
           vec_idx++) {
        first_rowgroup_reader->get_chunk(vec_idx);
      };
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed =
        end - start; // in milliseconds

    std::cout << "FastLanes Decompression Time: " << elapsed.count() << " ms"
              << std::endl;
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
    } else if (cli.scheme == "zstd") {
      compressor->setScheme(CompressionSchemeType::kZstd);
    } else if (cli.scheme == "snappy") {
      compressor->setScheme(CompressionSchemeType::kSnappy);
    } else if (cli.scheme == "datablock") {
      compressor->setScheme(CompressionSchemeType::kDataBlock);
    } else {
      std::cout << "Unknown algorithm" << std::endl;
      return 1;
    }

    // compress
    stats = compressor->compress(compress_out);

    // Thrash the CPU caches to read compressed data from memory.
    utils::thrashCPUCaches();

    // decompress
    if (cli.morsel) {
      timer.start();
      compressor->decompress(compress_out.get());
      timer.end();
    } else {
      timer.start();
      compressor->decompress(decompress_out, compress_out.get());
      timer.end();
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
  std::cout << "Decompression Time: " << timer.getMicroSeconds() << std::endl;
  std::cout << "Decompression Bandwidth (Compressed): "
            << (static_cast<double>(stats.compressed_size) / 1000000000) /
                   timer.getSeconds()
            << " GB/s" << std::endl;
  std::cout << "Decompression Bandwidth (Uncompressed): "
            << (static_cast<double>(stats.uncompressed_size) / 1000000000) /
                   timer.getSeconds()
            << " GB/s" << std::endl;

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