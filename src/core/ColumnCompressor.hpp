#pragma once
//---------------------------------------------------------------------------
#include <cstring>
#include <lz4.h>
#include <snappy.h>
#include <zstd.h>
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
  ColumnCompressor(const Column<T> &column,
                   const Phase2CompressionSettings *settings)
      : Compressor<T>(column, settings) {}
  //---------------------------------------------------------------------------
  CompressionStats compress(std::unique_ptr<u8[]> &dest) override {
    // allocate space (overallocate by 2x to prevent UB)
    u64 uncompressed_size = this->column.size() * sizeof(T);
    dest = std::make_unique<u8[]>(uncompressed_size * 2);

    // Phase 1: Regular compression phase.
    this->details = phase1_compress(dest.get());

    // Phase 2: A second compression phase on the compressed data (optional).
    if (this->settings) {
      phase2_compress(dest.get());
    }

    this->compressed_size =
        this->details.payload_size + this->details.header_size;
    return {uncompressed_size, this->compressed_size,
            static_cast<double>(uncompressed_size) / this->compressed_size,
            this->details};
  }
  //---------------------------------------------------------------------------
  void decompress(vector<T> &dest, u8 *src) override {
    // allocate space
    dest.reserve(this->column.size());

    // Phase 2: Decompression
    std::unique_ptr<char[]> temp;
    // If Phase2-Settings specified and Phase2-Compression was successfull.
    if (this->settings && !this->p2_fail) {
      phase2_decompress(temp, src);

      // Read from temporary buffer.
      src = reinterpret_cast<u8 *>(temp.get());
    }

    // Phase1: Decompression
    phase1_decompress(dest.data(), src);
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
  /// @brief Compress the raw data using this compressor's compression scheme.
  /// Note: This is referred to as Phase 1.
  CompressionDetails phase1_compress(u8 *dest) {
    CompressionDetails details{};

    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      details = compress<BitPacking<T, kTinyBlockSize>>(dest);
      break;
    case CompressionSchemeType::kDelta:
      details = compress<Delta<T>>(dest);
      break;
    case CompressionSchemeType::kFOR:
      details = compress<FOR<T>>(dest);
      break;
    case CompressionSchemeType::kFORn:
      details = compress<FORn<T, kTinyBlockSize>>(dest);
      break;
    case CompressionSchemeType::kRLE:
      details = compress<RLE<T>>(dest);
      break;
    case CompressionSchemeType::kTinyBlocks:
      details = compress<TinyBlocks<T, kTinyBlockSize>>(dest);
      break;
    case CompressionSchemeType::kUncompressed:
      details = compress<Uncompressed<T>>(dest);
      break;
    case CompressionSchemeType::kLZ4:
      details = compressLZ4(dest);
      break;
    case CompressionSchemeType::kZstd:
      details = compressZstd(dest);
      break;
    case CompressionSchemeType::kSnappy:
      details = compressSnappy(dest);
      break;
    default:
      throw std::runtime_error("Compression not supported for this scheme.");
    }

    return details;
  }
  //---------------------------------------------------------------------------
  /// @brief Compress the compressed data again using this compressor's
  /// secondary compression scheme. Note: This is referred to as Phase 2.
  void phase2_compress(u8 *dest) {
    this->p2_fail = false;
    std::unique_ptr<char[]> temp;

    // 1. only the header should be compressed
    if (this->settings->header_only && !this->settings->payload_only) {
      u64 size = details.header_size;
      u64 capacity = std::max(snappy::MaxCompressedLength(size), size);
      temp = std::make_unique<char[]>(capacity);

      // compress
      u64 compressed_header = 0;
      switch (this->settings->scheme) {
      case CompressionSchemeType::kLZ4:
        compressed_header = static_cast<u64>(
            LZ4_compress_default(reinterpret_cast<const char *>(dest), // src
                                 temp.get(),                           // dest
                                 size,                                 // size
                                 capacity) // capacity
        );
        break;
      case CompressionSchemeType::kZstd:
        compressed_header = ZSTD_compress(temp.get(), // dest
                                          capacity,   // capacity
                                          dest,       // src
                                          size,       // size
                                          1           // level
        );
        if (ZSTD_isError(compressed_header))
          compressed_header = 0;
        break;
      case CompressionSchemeType::kSnappy:
        snappy::RawCompress(reinterpret_cast<const char *>(dest), // src
                            size,                                 // size
                            temp.get(),                           // dest
                            &compressed_header);
        break;
      default:
        throw std::runtime_error(
            "Phase2 Compression not supported for this scheme.");
      }

      if (compressed_header > 0 &&
          compressed_header < size) { // if it was compressed at all
        std::memcpy(dest, temp.get(), compressed_header);
        std::memmove(dest + compressed_header, dest + details.header_size,
                     details.payload_size);

        details.header_size = compressed_header;
      } else { // compression failed
        this->p2_fail = true;
      }

      // 2. only the payload should be compressed
    } else if (this->settings->payload_only && !this->settings->header_only) {
      u64 size = details.payload_size;
      u64 capacity = std::max(snappy::MaxCompressedLength(size), size);
      temp = std::make_unique<char[]>(capacity);

      // compress
      u8 *payload = dest + details.header_size;
      u64 compressed_payload = 0;
      switch (this->settings->scheme) {
      case CompressionSchemeType::kLZ4:
        compressed_payload = static_cast<u64>(
            LZ4_compress_default(reinterpret_cast<const char *>(payload), // src
                                 temp.get(), // dest
                                 size,       // size
                                 capacity)   // capacity
        );
        break;
      case CompressionSchemeType::kZstd:
        compressed_payload = ZSTD_compress(temp.get(), // dest
                                           capacity,   // capacity
                                           payload,    // src
                                           size,       // size
                                           1           // level
        );
        if (ZSTD_isError(compressed_payload))
          compressed_payload = 0;
        break;
      case CompressionSchemeType::kSnappy:
        snappy::RawCompress(reinterpret_cast<const char *>(payload), // src
                            size,                                    // size
                            temp.get(),                              // dest
                            &compressed_payload);
        break;
      default:
        throw std::runtime_error(
            "Phase2 Compression not supported for this scheme.");
      }

      if (compressed_payload > 0 &&
          compressed_payload < size) { // if it was compressed at all
        std::memcpy(payload, temp.get(), compressed_payload);
        details.payload_size = compressed_payload;
      } else { // compression failed
        this->p2_fail = true;
      }

      // 3. the whole compressed data should be compressed again
    } else {
      u64 size = details.payload_size + details.header_size;
      u64 capacity = std::max(snappy::MaxCompressedLength(size), size);
      temp = std::make_unique<char[]>(capacity);

      // compress
      u64 compressed = 0;
      switch (this->settings->scheme) {
      case CompressionSchemeType::kLZ4:
        compressed = static_cast<u64>(
            LZ4_compress_default(reinterpret_cast<const char *>(dest), // src
                                 temp.get(),                           // dest
                                 size,                                 // size
                                 capacity) // capacity
        );
        break;
      case CompressionSchemeType::kZstd:
        compressed = ZSTD_compress(temp.get(), // dest
                                   capacity,   // capacity
                                   dest,       // src
                                   size,       // size
                                   1           // level
        );
        if (ZSTD_isError(compressed))
          compressed = 0;
        break;
      case CompressionSchemeType::kSnappy:
        snappy::RawCompress(reinterpret_cast<const char *>(dest), // src
                            size,                                 // size
                            temp.get(),                           // dest
                            &compressed);
        break;
      default:
        throw std::runtime_error(
            "Phase2 Compression not supported for this scheme.");
      }

      if (compressed > 0 && compressed < size) { // if it was compressed at all
        std::memcpy(dest, temp.get(), compressed);
        details.header_size = 0;
        details.payload_size = compressed;
      } else { // compression failed
        this->p2_fail = true;
      }
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Generic compression implementation for all compression schemes.
  template <typename Scheme> CompressionDetails compress(u8 *dest) {
    CompressionDetails details{};

    Scheme scheme;
    if (scheme.isPartitioningScheme()) {
      vector<Statistics<T>> stats;
      auto block_count = this->column.size() / kTinyBlockSize;
      for (size_t i = 0; i < block_count; ++i) {
        stats.push_back(Statistics<T>::generateFrom(
            this->column.data() + i * kTinyBlockSize, kTinyBlockSize));
      }

      details = scheme.compress(this->column.data(), this->column.size(), dest,
                                stats.data());
    } else {
      auto stats =
          Statistics<T>::generateFrom(this->column.data(), this->column.size());

      details = scheme.compress(this->column.data(), this->column.size(), dest,
                                &stats);
    }

    return details;
  }
  //---------------------------------------------------------------------------
  /// @brief LZ4 compression.
  CompressionDetails compressLZ4(u8 *dest) {
    int capacity = this->column.size() * sizeof(T);
    return {0, static_cast<u64>(LZ4_compress_default(
                   reinterpret_cast<const char *>(this->column.data()),
                   reinterpret_cast<char *>(dest), capacity, capacity))};
  }
  //---------------------------------------------------------------------------
  /// @brief Zstd compression.
  CompressionDetails compressZstd(u8 *dest) {
    int capacity = this->column.size() * sizeof(T);
    u64 size = ZSTD_compress(dest, capacity, this->column.data(), capacity, 1);
    if (ZSTD_isError(size)) {
      throw std::runtime_error("Zstd-Compression failed.");
    }
    return {0, size};
  }
  //---------------------------------------------------------------------------
  /// @brief Snappy compression.
  CompressionDetails compressSnappy(u8 *dest) {
    size_t capacity = this->column.size() * sizeof(T);
    size_t compressed_length = 0;
    snappy::RawCompress(reinterpret_cast<const char *>(this->column.data()),
                        capacity, reinterpret_cast<char *>(dest),
                        &compressed_length);
    return {0, compressed_length};
  }
  //---------------------------------------------------------------------------
  /// @brief Generic decompression implementation for all compression schemes.
  template <typename Scheme> void decompress(T *dest, u8 *src) {
    Scheme scheme;
    scheme.decompress(dest, this->column.size(), src);
  }
  //---------------------------------------------------------------------------
  /// @brief Decompression of this compressor's compression scheme.
  void phase1_decompress(T *dest, u8 *src) {
    switch (this->scheme) {
    case CompressionSchemeType::kBitPacking:
      decompress<BitPacking<T, kTinyBlockSize>>(dest, src);
      return;
    case CompressionSchemeType::kDelta:
      decompress<Delta<T>>(dest, src);
      return;
    case CompressionSchemeType::kFOR:
      decompress<FOR<T>>(dest, src);
      return;
    case CompressionSchemeType::kFORn:
      decompress<FORn<T, kTinyBlockSize>>(dest, src);
      return;
    case CompressionSchemeType::kRLE:
      decompress<RLE<T>>(dest, src);
      return;
    case CompressionSchemeType::kTinyBlocks:
      decompress<TinyBlocks<T, kTinyBlockSize>>(dest, src);
      return;
    case CompressionSchemeType::kUncompressed:
      decompress<Uncompressed<T>>(dest, src);
      return;
    case CompressionSchemeType::kLZ4:
      decompressLZ4(dest, src);
      return;
    case CompressionSchemeType::kZstd:
      decompressZstd(dest, src);
      return;
    case CompressionSchemeType::kSnappy:
      decompressSnappy(dest, src);
      return;
    default:
      throw std::runtime_error(
          "Compression on DataBlocks not supported for this scheme.");
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Decompression of the compression layer on top of the compressed
  /// data.
  void phase2_decompress(std::unique_ptr<char[]> &temp, u8 *src) {
    // allocate temporary space (overallocate to prevent UB)
    auto allocated_size = this->column.size() * sizeof(T) * 2;
    temp = std::make_unique<char[]>(allocated_size);

    // 1. only the header should be decompressed
    if (this->settings->header_only && !this->settings->payload_only) {
      // decompress into temp buffer
      u64 decompressed_size = 0;
      switch (this->settings->scheme) {
      case CompressionSchemeType::kLZ4:
        decompressed_size =
            LZ4_decompress_safe(reinterpret_cast<const char *>(src), // src
                                temp.get(),                          // dest
                                this->details.header_size,           // size
                                allocated_size                       // capacity
            );
        break;
      case CompressionSchemeType::kZstd:
        decompressed_size = ZSTD_decompress(temp.get(),     // dest
                                            allocated_size, // capacity
                                            src,            // src
                                            this->details.header_size // size
        );
        if (ZSTD_isError(decompressed_size)) {
          throw std::runtime_error("Zstd-Decompression of the header failed.");
        }
        break;
      case CompressionSchemeType::kSnappy:
        snappy::RawUncompress(reinterpret_cast<const char *>(src), // src
                              this->details.header_size,           // size
                              temp.get()                           // dest
        );
        snappy::GetUncompressedLength(reinterpret_cast<const char *>(src),
                                      this->details.header_size,
                                      &decompressed_size);
        break;
      default:
        throw std::runtime_error(
            "Phase2-Compression not supported for this scheme.");
      }

      if (decompressed_size <= 0) {
        throw std::runtime_error("Phase2-Decompression of the header failed.");
      }

      // copy the rest of the data
      std::memcpy(temp.get() + decompressed_size,
                  src + this->details.header_size, this->details.payload_size);

      // 2. only the payload should be decompressed
    } else if (this->settings->payload_only && !this->settings->header_only) {
      // copy the header
      std::memcpy(temp.get(), src, this->details.header_size);

      // decompress into temp buffer
      u8 *payload = src + details.header_size;
      u64 decompressed_size = 0;
      switch (this->settings->scheme) {
      case CompressionSchemeType::kLZ4:
        decompressed_size = LZ4_decompress_safe(
            reinterpret_cast<const char *>(payload),   // src
            temp.get() + this->details.header_size,    // dest
            this->details.payload_size,                // size
            allocated_size - this->details.header_size // capacity
        );
        break;
      case CompressionSchemeType::kZstd:
        decompressed_size = ZSTD_decompress(
            temp.get() + this->details.header_size,     // dest
            allocated_size - this->details.header_size, // capacity
            payload,                                    // src
            this->details.payload_size                  // size
        );
        if (ZSTD_isError(decompressed_size)) {
          throw std::runtime_error("Zstd-Decompression of the payload failed.");
        }
        break;
      case CompressionSchemeType::kSnappy:
        snappy::RawUncompress(reinterpret_cast<const char *>(payload), // src
                              this->details.payload_size,              // size
                              temp.get() + this->details.header_size   // dest
        );
        snappy::GetUncompressedLength(reinterpret_cast<const char *>(payload),
                                      this->details.payload_size,
                                      &decompressed_size);
        break;
      default:
        throw std::runtime_error(
            "Phase2-Compression not supported for this scheme.");
      }

      if (decompressed_size <= 0) {
        throw std::runtime_error("Phase2-Decompression of the payload failed.");
      }
      // 3. the whole compressed data should be decompressed again
    } else {
      // decompress into temp buffer
      u64 decompressed_size = 0;
      switch (this->settings->scheme) {
      case CompressionSchemeType::kLZ4:
        decompressed_size =
            LZ4_decompress_safe(reinterpret_cast<const char *>(src), // src
                                temp.get(),                          // dest
                                this->compressed_size,               // size
                                allocated_size                       // capacity
            );
        break;
      case CompressionSchemeType::kZstd:
        decompressed_size = ZSTD_decompress(temp.get(),           // dest
                                            allocated_size,       // capacity
                                            src,                  // src
                                            this->compressed_size // size
        );
        if (ZSTD_isError(decompressed_size)) {
          throw std::runtime_error("Zstd-Decompression failed.");
        }
        break;
      case CompressionSchemeType::kSnappy:
        snappy::RawUncompress(reinterpret_cast<const char *>(src), // src
                              this->compressed_size,               // size
                              temp.get()                           // dest
        );
        snappy::GetUncompressedLength(reinterpret_cast<const char *>(src),
                                      this->compressed_size,
                                      &decompressed_size);
        break;
      default:
        throw std::runtime_error(
            "Phase2-Compression not supported for this scheme.");
      }

      if (decompressed_size <= 0) {
        throw std::runtime_error("Phase2-Decompression failed.");
      }
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Generic morsel-driven decompression implementation for all
  /// compression schemes.
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
  /// @brief LZ4 decompression.
  void decompressLZ4(T *dest, u8 *src) {
    LZ4_decompress_safe(reinterpret_cast<const char *>(src),
                        reinterpret_cast<char *>(dest), compressed_size,
                        this->column.size() * sizeof(T));
  }
  //---------------------------------------------------------------------------
  /// @brief Zstd decompression.
  void decompressZstd(T *dest, u8 *src) {
    u64 size = ZSTD_decompress(dest, this->column.size() * sizeof(T), src,
                               compressed_size);
    if (ZSTD_isError(size)) {
      throw std::runtime_error("Zstd-Decompression failed.");
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Snappy decompression.
  void decompressSnappy(T *dest, u8 *src) {
    snappy::RawUncompress(reinterpret_cast<const char *>(src), compressed_size,
                          reinterpret_cast<char *>(dest));
  }

  /// Details on the compressed data.
  CompressionDetails details;
  /// The size of the compressed data.
  u64 compressed_size;
  /// Whether the Phase2-Compression failed.
  bool p2_fail;
};
//---------------------------------------------------------------------------
} // namespace compression