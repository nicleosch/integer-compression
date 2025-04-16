#pragma once
//---------------------------------------------------------------------------
#include "core/Compressor.hpp"
#include "schemes/BitPacking.hpp"
#include "schemes/Delta.hpp"
#include "schemes/FOR.hpp"
#include "schemes/FORn.hpp"
#include "schemes/RLE.hpp"
#include "schemes/TinyBlocks.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
constexpr u32 kDefaultDataBlockSize = 65536;
//---------------------------------------------------------------------------
struct DataBlocksMeta {
  /// The number of blocks for the given column.
  u32 block_count;
  /// The block offsets in the compressed data.
  u32 block_offsets[];
};
//---------------------------------------------------------------------------
/// This class implements a DataBlocks abstraction as presented here:
/// https://db.in.tum.de/downloads/publications/datablocks.pdf
/// Note: Some compression schemes work on smaller blocks either by construction
/// or because the implementation requires it. For these schemes, we include an
/// additional template parameter for efficiency reasons.
/// @tparam kDataBlockSize The size of a DataBlock.
/// @tparam kTinyBlockSize The size of the small blocks.
/// @tparam kMorselSize The size of a morsel for decompression benchmarking.
template <typename T, const u32 kDataBlockSize = kDefaultDataBlockSize,
          const u16 kTinyBlockSize = 128, const u16 kMorselSize = 1024>
class BlockCompressor : public Compressor<T> {
public:
  //---------------------------------------------------------------------------
  explicit BlockCompressor(const Column<T> &column) : Compressor<T>(column) {}
  //---------------------------------------------------------------------------
  BlockCompressor(const Column<T> &column, const CompressionSchemeType scheme)
      : Compressor<T>(column, scheme) {}
  //---------------------------------------------------------------------------
  CompressionStats compress(std::unique_ptr<u8[]> &dest) override {
    // allocate space (overallocate by 2x to prevent UB)
    u64 uncompressed_size = this->column.size() * sizeof(T);
    dest = std::make_unique<u8[]>(uncompressed_size * 2);

    // write meta data
    auto &meta_data = *reinterpret_cast<DataBlocksMeta *>(dest.get());
    meta_data.block_count = this->column.size() / kDataBlockSize;
    auto write_ptr = dest.get() + sizeof(meta_data.block_count) +
                     sizeof(u32) * meta_data.block_count;

    u32 total_size = 0;
    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      total_size =
          compress<BitPacking<T, kTinyBlockSize>>(write_ptr, meta_data);
      break;
    case CompressionSchemeType::kDelta:
      total_size = compress<Delta<T>>(write_ptr, meta_data);
      break;
    case CompressionSchemeType::kFOR:
      total_size = compress<FOR<T>>(write_ptr, meta_data);
      break;
    case CompressionSchemeType::kFORn:
      total_size = compress<FORn<T, kTinyBlockSize>>(write_ptr, meta_data);
      break;
    case CompressionSchemeType::kRLE:
      total_size = compress<RLE<T>>(write_ptr, meta_data);
      break;
    case CompressionSchemeType::kTinyBlocks:
      total_size =
          compress<TinyBlocks<T, kTinyBlockSize>>(write_ptr, meta_data);
      break;
    default:
      throw std::runtime_error(
          "Compression on DataBlocks not supported for this scheme.");
    }

    return {static_cast<double>(uncompressed_size) / total_size,
            uncompressed_size, total_size};
  }
  //---------------------------------------------------------------------------
  void decompress(vector<T> &dest, u8 *src) override {
    // allocate space
    dest.reserve(this->column.size());

    // read meta data
    const auto &meta_data = *reinterpret_cast<DataBlocksMeta *>(src);
    auto read_ptr = src + sizeof(meta_data.block_count) +
                    sizeof(u32) * meta_data.block_count;

    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      decompress<BitPacking<T, kTinyBlockSize>>(dest.data(), read_ptr,
                                                meta_data);
      return;
    case CompressionSchemeType::kDelta:
      decompress<Delta<T>>(dest.data(), read_ptr, meta_data);
      return;
    case CompressionSchemeType::kFOR:
      decompress<FOR<T>>(dest.data(), read_ptr, meta_data);
      return;
    case CompressionSchemeType::kFORn:
      decompress<FORn<T, kTinyBlockSize>>(dest.data(), read_ptr, meta_data);
      return;
    case CompressionSchemeType::kRLE:
      decompress<RLE<T>>(dest.data(), read_ptr, meta_data);
      return;
    case CompressionSchemeType::kTinyBlocks:
      decompress<TinyBlocks<T, kTinyBlockSize>>(dest.data(), read_ptr,
                                                meta_data);
      return;
    default:
      throw std::runtime_error(
          "Compression on DataBlocks not supported for this scheme.");
    }
  }
  //---------------------------------------------------------------------------
  void decompress(u8 *src) {
    // read meta data
    const auto &meta_data = *reinterpret_cast<DataBlocksMeta *>(src);
    auto read_ptr = src + sizeof(meta_data.block_count) +
                    sizeof(u32) * meta_data.block_count;

    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      decompress<BitPacking<T, kTinyBlockSize>>(read_ptr, meta_data);
      return;
    case CompressionSchemeType::kFOR:
      decompress<FOR<T>>(read_ptr, meta_data);
      return;
    case CompressionSchemeType::kFORn:
      decompress<FORn<T, kTinyBlockSize>>(read_ptr, meta_data);
      return;
    case CompressionSchemeType::kTinyBlocks:
      decompress<TinyBlocks<T, kTinyBlockSize>>(read_ptr, meta_data);
      return;
    default:
      throw std::runtime_error(
          "Compression into a morsel not supported for this scheme.");
    }
  }

private:
  //---------------------------------------------------------------------------
  template <typename Scheme> u32 compress(u8 *dest, DataBlocksMeta &meta_data) {
    u32 size = 0;

    for (size_t i = 0; i < meta_data.block_count; ++i) {
      auto read_ptr = this->column.data() + i * kDataBlockSize;
      auto write_ptr = dest + size;
      meta_data.block_offsets[i] = size;

      // compress
      Scheme scheme;
      if (scheme.isPartitioningScheme()) {
        // calculate stats for each partition
        vector<Statistics<T>> stats;
        auto block_count = kDataBlockSize / kTinyBlockSize;
        for (size_t i = 0; i < block_count; ++i) {
          stats.push_back(Statistics<T>::generateFrom(
              read_ptr + i * kTinyBlockSize, kTinyBlockSize));
        }

        size +=
            scheme.compress(read_ptr, kDataBlockSize, write_ptr, stats.data());
      } else {
        // calculate stats over the whole block
        auto stats = Statistics<T>::generateFrom(read_ptr, kDataBlockSize);

        size += scheme.compress(read_ptr, kDataBlockSize, write_ptr, &stats);
      }
    }

    return size;
  }
  //---------------------------------------------------------------------------
  template <typename Scheme>
  void decompress(T *dest, u8 *src, const DataBlocksMeta &meta_data) {
    for (size_t i = 0; i < meta_data.block_count; ++i) {
      auto read_ptr = src + meta_data.block_offsets[i];
      auto write_ptr = dest + i * kDataBlockSize;

      // decompress
      Scheme scheme;
      scheme.decompress(write_ptr, kDataBlockSize, read_ptr);
    }
  }
  //---------------------------------------------------------------------------
  template <typename Scheme>
  void decompress(u8 *src, const DataBlocksMeta &meta_data) {
    // create L1-resident buffer
    vector<T> dest(kMorselSize);

    Scheme scheme;
    for (size_t i = 0; i < meta_data.block_count; ++i) {
      auto read_ptr = src + meta_data.block_offsets[i];

      auto morsel_count = kDataBlockSize / kMorselSize;

      // decompress
      if (scheme.isPartitioningScheme()) {
        auto blocks_per_morsel = kMorselSize / kTinyBlockSize;

        for (u32 i = 0; i < morsel_count; ++i) {
          scheme.decompress(dest.data(), kMorselSize, read_ptr,
                            i * blocks_per_morsel);
        }
      } else {
        for (u32 i = 0; i < morsel_count; ++i) {
          scheme.decompress(dest.data(), kMorselSize, read_ptr,
                            i * kMorselSize);
        }
      }
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression