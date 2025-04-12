#pragma once
//---------------------------------------------------------------------------
#include <lz4.h>
//---------------------------------------------------------------------------
#include "core/Compressor.hpp"
#include "schemes/BitPacking.hpp"
#include "schemes/Delta.hpp"
#include "schemes/FOR.hpp"
#include "schemes/FORn.hpp"
#include "schemes/RLE.hpp"
#include "schemes/TinyBlocks.hpp"
#include "schemes/Uncompressed.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
/// This class compresses a the whole column without partitioning it into
/// DataBlocks.
/// Note: Some compression schemes work on smaller blocks either by construction
/// or because the implementation requires it. For these schemes, we include an
/// additional template parameter for efficiency reasons.
/// @tparam kTinyBlockSize The size of the small blocks.
template <const u16 kTinyBlockSize = 128, const u16 kMorselSize = 1024>
class ColumnCompressor : public Compressor {
public:
  //---------------------------------------------------------------------------
  explicit ColumnCompressor(const Column &column) : Compressor(column) {}
  //---------------------------------------------------------------------------
  ColumnCompressor(const Column &column, const CompressionSchemeType scheme)
      : Compressor(column, scheme) {}
  //---------------------------------------------------------------------------
  CompressionStats compress(std::unique_ptr<u8[]> &dest) override {
    // allocate space (overallocate by 2x to prevent UB)
    u64 uncompressed_size = this->column.size() * sizeof(INTEGER);
    dest = std::make_unique<u8[]>(uncompressed_size * 2);

    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      compressed_size = compress<BitPacking<kTinyBlockSize>>(dest.get());
      break;
    case CompressionSchemeType::kDelta:
      compressed_size = compress<Delta>(dest.get());
      break;
    case CompressionSchemeType::kFOR:
      compressed_size = compress<FOR>(dest.get());
      break;
    case CompressionSchemeType::kFORn:
      compressed_size = compress<FORn<kTinyBlockSize>>(dest.get());
      break;
    case CompressionSchemeType::kRLE:
      compressed_size = compress<RLE>(dest.get());
      break;
    case CompressionSchemeType::kTinyBlocks:
      compressed_size = compress<TinyBlocks<kTinyBlockSize>>(dest.get());
      break;
    case CompressionSchemeType::kUncompressed:
      compressed_size = compress<Uncompressed>(dest.get());
      break;
    case CompressionSchemeType::kLZ4:
      compressed_size = compressLZ4(dest.get());
      break;
    default:
      throw std::runtime_error("Compression not supported for this scheme.");
    }

    return {static_cast<double>(uncompressed_size) / compressed_size,
            uncompressed_size, compressed_size};
  }
  //---------------------------------------------------------------------------
  void decompress(vector<INTEGER> &dest, u8 *src) override {
    // allocate space
    dest.reserve(this->column.size());

    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      decompress<BitPacking<kTinyBlockSize>>(dest.data(), src);
      return;
    case CompressionSchemeType::kDelta:
      decompress<Delta>(dest.data(), src);
      return;
    case CompressionSchemeType::kFOR:
      decompress<FOR>(dest.data(), src);
      return;
    case CompressionSchemeType::kFORn:
      decompress<FORn<kTinyBlockSize>>(dest.data(), src);
      return;
    case CompressionSchemeType::kRLE:
      decompress<RLE>(dest.data(), src);
      return;
    case CompressionSchemeType::kTinyBlocks:
      decompress<TinyBlocks<kTinyBlockSize>>(dest.data(), src);
      return;
    case CompressionSchemeType::kUncompressed:
      decompress<Uncompressed>(dest.data(), src);
      return;
    case CompressionSchemeType::kLZ4:
      decompressLZ4(dest.data(), src);
      return;
    default:
      throw std::runtime_error(
          "Compression on DataBlocks not supported for this scheme.");
    }
  }
  //---------------------------------------------------------------------------
  void decompress(u8 *src) {
    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      decompress<BitPacking<kTinyBlockSize>>(src);
      return;
    case CompressionSchemeType::kFOR:
      decompress<FOR>(src);
      return;
    case CompressionSchemeType::kFORn:
      decompress<FORn<kTinyBlockSize>>(src);
      return;
    case CompressionSchemeType::kTinyBlocks:
      decompress<TinyBlocks<kTinyBlockSize>>(src);
      return;
    case CompressionSchemeType::kUncompressed:
      decompress<Uncompressed>(src);
      return;
    default:
      throw std::runtime_error(
          "Compression into a morsel not supported for this scheme.");
    }
  }

private:
  //---------------------------------------------------------------------------
  template <typename Scheme> u32 compress(u8 *dest) {
    u32 size = 0;

    Scheme scheme;
    if (scheme.isPartitioningScheme()) {
      vector<Statistics> stats;
      auto block_count = this->column.size() / kTinyBlockSize;
      for (size_t i = 0; i < block_count; ++i) {
        stats.push_back(Statistics::generateFrom(
            this->column.data() + i * kTinyBlockSize, kTinyBlockSize));
      }

      size = scheme.compress(this->column.data(), this->column.size(), dest,
                             stats.data());
    } else {
      auto stats =
          Statistics::generateFrom(this->column.data(), this->column.size());

      size = scheme.compress(this->column.data(), this->column.size(), dest,
                             &stats);
    }

    return size;
  }
  //---------------------------------------------------------------------------
  u32 compressLZ4(u8 *dest) {
    return static_cast<u32>(LZ4_compress_default(
        reinterpret_cast<const char *>(this->column.data()),
        reinterpret_cast<char *>(dest), this->column.size(),
        this->column.size()));
  }
  //---------------------------------------------------------------------------
  template <typename Scheme> void decompress(INTEGER *dest, u8 *src) {
    Scheme scheme;
    scheme.decompress(dest, this->column.size(), src);
  }
  //---------------------------------------------------------------------------
  template <typename Scheme> void decompress(u8 *src) {
    // create L1-resident buffer
    vector<INTEGER> dest(kMorselSize);

    auto morsel_count = this->column.size() / kMorselSize;

    Scheme scheme;
    if (scheme.isPartitioningScheme()) {
      auto blocks_per_morsel = kMorselSize / kTinyBlockSize;

      for (u32 i = 0; i < morsel_count; ++i) {
        scheme.decompress(dest.data(), kMorselSize, src, i * blocks_per_morsel);
      }
    } else {
      for (u32 i = 0; i < morsel_count; ++i) {
        scheme.decompress(dest.data(), kMorselSize, src, i * kMorselSize);
      }
    }
  }
  //---------------------------------------------------------------------------
  void decompressLZ4(INTEGER *dest, u8 *src) {
    LZ4_decompress_safe(reinterpret_cast<const char *>(src),
                        reinterpret_cast<char *>(dest), compressed_size,
                        this->column.size() * sizeof(INTEGER));
  }

  /// The size of the compressed data.
  u32 compressed_size;
};
//---------------------------------------------------------------------------
} // namespace compression