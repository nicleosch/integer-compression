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
/// @tparam kMorselSize The size of a morsel for decompression benchmarking.
template <typename T, const u16 kTinyBlockSize = 128,
          const u16 kMorselSize = 1024>
class ColumnCompressor : public Compressor<T> {
public:
  //---------------------------------------------------------------------------
  explicit ColumnCompressor(const Column<T> &column) : Compressor<T>(column) {}
  //---------------------------------------------------------------------------
  ColumnCompressor(const Column<T> &column, const CompressionSchemeType scheme)
      : Compressor<T>(column, scheme) {}
  //---------------------------------------------------------------------------
  CompressionStats compress(std::unique_ptr<u8[]> &dest) override {
    // allocate space (overallocate by 2x to prevent UB)
    u64 uncompressed_size = this->column.size() * sizeof(T);
    dest = std::make_unique<u8[]>(uncompressed_size * 2);

    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      compressed_size = compress<BitPacking<T, kTinyBlockSize>>(dest.get());
      break;
    case CompressionSchemeType::kDelta:
      compressed_size = compress<Delta<T>>(dest.get());
      break;
    case CompressionSchemeType::kFOR:
      compressed_size = compress<FOR<T>>(dest.get());
      break;
    case CompressionSchemeType::kFORn:
      compressed_size = compress<FORn<T, kTinyBlockSize>>(dest.get());
      break;
    case CompressionSchemeType::kRLE:
      compressed_size = compress<RLE<T>>(dest.get());
      break;
    case CompressionSchemeType::kTinyBlocks:
      compressed_size = compress<TinyBlocks<T, kTinyBlockSize>>(dest.get());
      break;
    case CompressionSchemeType::kUncompressed:
      compressed_size = compress<Uncompressed<T>>(dest.get());
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
  void decompress(vector<T> &dest, u8 *src) override {
    // allocate space
    dest.reserve(this->column.size());

    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      decompress<BitPacking<T, kTinyBlockSize>>(dest.data(), src);
      return;
    case CompressionSchemeType::kDelta:
      decompress<Delta<T>>(dest.data(), src);
      return;
    case CompressionSchemeType::kFOR:
      decompress<FOR<T>>(dest.data(), src);
      return;
    case CompressionSchemeType::kFORn:
      decompress<FORn<T, kTinyBlockSize>>(dest.data(), src);
      return;
    case CompressionSchemeType::kRLE:
      decompress<RLE<T>>(dest.data(), src);
      return;
    case CompressionSchemeType::kTinyBlocks:
      decompress<TinyBlocks<T, kTinyBlockSize>>(dest.data(), src);
      return;
    case CompressionSchemeType::kUncompressed:
      decompress<Uncompressed<T>>(dest.data(), src);
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
      decompress<BitPacking<T, kTinyBlockSize>>(src);
      return;
    case CompressionSchemeType::kFOR:
      decompress<FOR<T>>(src);
      return;
    case CompressionSchemeType::kFORn:
      decompress<FORn<T, kTinyBlockSize>>(src);
      return;
    case CompressionSchemeType::kTinyBlocks:
      decompress<TinyBlocks<T, kTinyBlockSize>>(src);
      return;
    case CompressionSchemeType::kUncompressed:
      decompress<Uncompressed<T>>(src);
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
      vector<Statistics<T>> stats;
      auto block_count = this->column.size() / kTinyBlockSize;
      for (size_t i = 0; i < block_count; ++i) {
        stats.push_back(Statistics<T>::generateFrom(
            this->column.data() + i * kTinyBlockSize, kTinyBlockSize));
      }

      size = scheme.compress(this->column.data(), this->column.size(), dest,
                             stats.data());
    } else {
      auto stats =
          Statistics<T>::generateFrom(this->column.data(), this->column.size());

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
  template <typename Scheme> void decompress(T *dest, u8 *src) {
    Scheme scheme;
    scheme.decompress(dest, this->column.size(), src);
  }
  //---------------------------------------------------------------------------
  template <typename Scheme> void decompress(u8 *src) {
    // create L1-resident buffer
    vector<T> dest(kMorselSize);

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
  void decompressLZ4(T *dest, u8 *src) {
    LZ4_decompress_safe(reinterpret_cast<const char *>(src),
                        reinterpret_cast<char *>(dest), compressed_size,
                        this->column.size() * sizeof(T));
  }

  /// The size of the compressed data.
  u32 compressed_size;
};
//---------------------------------------------------------------------------
} // namespace compression