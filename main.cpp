#include <iostream>
//---------------------------------------------------------------------------
#include "bootstrap/cli.hpp"
#include "common/Utils.hpp"
#include "compression/Compressor.hpp"
#include "compression/Decompressor.hpp"
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
  auto column = storage::Column::fromFile(cli.data.c_str(), 0, '|');
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
    if (cli.morsel) {
      std::cout << "Logging not allowed when decompressing into morsels."
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