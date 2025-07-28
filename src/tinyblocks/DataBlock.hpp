#pragma once
//---------------------------------------------------------------------------
#include "common/Types.hpp"
#include "statistics/Statistics.hpp"
#include "tinyblocks/TinyBlocks.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace tinyblocks {
//---------------------------------------------------------------------------
namespace datablock {
//---------------------------------------------------------------------------
constexpr u32 kDefaultSize = 65536;
//---------------------------------------------------------------------------
/// The compression scheme applied to a whole datablock.
enum class Scheme : u8 {
  NONE = 0,
  MONOTONIC = 1,
  TRUNCATION = 2,
};
//---------------------------------------------------------------------------
/// The tag wrapping a scheme code and its payload.
struct Tag {
  Scheme scheme;
  u8 payload;
};
static_assert(sizeof(Tag) == 2);
//---------------------------------------------------------------------------
/// A datablock abstraction for our tinyblocks.
template <typename T,                     // The data type to be compressed.
          const u16 kTinyBlockSize = 512, // The size of a tinyblocks block.
          const u32 kBlockSize =
              kDefaultSize, // The default size of a datablock.
          const bool kTinyBlocksActive =
              true> // Whether tinyblocks compression is active.
class DataBlock {
public:
  //---------------------------------------------------------------------------
  static_assert(kBlockSize % kTinyBlockSize == 0,
                "Datablock size must be multiple of Tinyblock size.");
  //---------------------------------------------------------------------------
  /// The header stored in front of each datablock.
  struct Header {
    /// The minimum in the block.
    T min;
    /// The maximum in the block.
    T max;
    /// The number of compressed bytes.
    u32 cbytes;
    /// The applied block size in the data block.
    u16 block_size;
    /// The scheme applied to the datablock and its payload, possibly none.
    Tag tag;
    /// The payload.
    u8 data[];
  };
  static_assert(sizeof(Header) % sizeof(T) == 0,
                "Data must be sizeof(T)-Byte aligned.");
  //---------------------------------------------------------------------------
  /// @brief Compress given data on a datablock granularity.
  /// @param src The data to be compressed.
  /// @param size The number of values to be compressed.
  /// @param dest The location to compress the data to.
  /// @param scheme The scheme to compress the data with, if given.
  /// @return Statistics on the compressed data.
  CompressionDetails compress(const T *src, const u32 size, u8 *dest,
                              const tinyblocks::Scheme *scheme = nullptr) {
    assert(size % kTinyBlockSize == 0);
    //---------------------------------------------------------------------------
    CompressionDetails details{};  // Full details.
    CompressionDetails pdetails{}; // Partial details.
    //---------------------------------------------------------------------------
    const u32 cblock = size / kBlockSize;
    auto read_ptr = src;
    auto write_ptr = dest;
    //---------------------------------------------------------------------------
    // Compress all datablocks.
    for (u32 i = 0; i < cblock; ++i, read_ptr += kBlockSize) {
      pdetails = compressImpl(read_ptr, kBlockSize, write_ptr, scheme);
      details.header_size += pdetails.header_size;
      details.payload_size += pdetails.payload_size;
      //---------------------------------------------------------------------------
      write_ptr += pdetails.header_size + pdetails.payload_size;
    }
    //---------------------------------------------------------------------------
    // Compress the rest.
    pdetails = compressImpl(read_ptr, size % kBlockSize, write_ptr, scheme);
    details.header_size += pdetails.header_size;
    details.payload_size += pdetails.payload_size;
    //---------------------------------------------------------------------------
    return details;
  }
  //---------------------------------------------------------------------------
  /// @brief Decompress given datablocks.
  /// @param dest The location to decompress the data to.
  /// @param size The number of compressed values.
  /// @param src The compressed datablocks.
  void decompress(T *dest, const u32 size, const u8 *src) {
    const u32 cblock = size / kBlockSize;
    auto read_ptr = src;
    auto write_ptr = dest;
    //---------------------------------------------------------------------------
    for (u32 i = 0; i < cblock; ++i, write_ptr += kBlockSize) {
      read_ptr +=
          sizeof(Header) + decompressImpl(write_ptr, kBlockSize, read_ptr);
    }
    //---------------------------------------------------------------------------
    decompressImpl(write_ptr, size % kBlockSize, read_ptr);
  }
  //---------------------------------------------------------------------------
  /// @brief Filter the datablocks.
  /// @param data The compressed data to be filtered.
  /// @param size The number of compressed values.
  /// @param predicate The predicate used for filtering.
  /// @param mv The vector to indicate the matching values in.
  void filter(const u8 *data, const u32 size, algebra::Predicate<T> &predicate,
              MatchVector &mv) {
    mv.resize(size);
    //---------------------------------------------------------------------------
    const u32 cblock = size / kBlockSize;
    auto read_ptr = data;
    auto match_ptr = mv.data();
    //---------------------------------------------------------------------------
    for (u32 i = 0; i < cblock; ++i) {
      auto header = *reinterpret_cast<const Header *>(read_ptr);
      read_ptr += sizeof(Header);
      filterImpl(read_ptr, kBlockSize, header, predicate, match_ptr);
      read_ptr += header.cbytes;
      match_ptr += kBlockSize;
    }
    //---------------------------------------------------------------------------
    auto header = *reinterpret_cast<const Header *>(read_ptr);
    filterImpl(read_ptr + sizeof(Header), size % kBlockSize, header, predicate,
               match_ptr);
  }

private:
  /// @brief Compress given datablock.
  /// Note: This function contains the actual compression logic.
  /// @param src The data to be compressed.
  /// @param size The number of values to be compressed.
  /// @param dest The location to compress the data to.
  /// @param scheme The scheme to compress the data with, if given.
  /// @return Statistics on the compressed data.
  CompressionDetails compressImpl(const T *src, const u32 size, u8 *dest,
                                  const tinyblocks::Scheme *scheme) {
    assert(size % kTinyBlockSize == 0);
    //---------------------------------------------------------------------------
    if (size == 0)
      return {0, 0};
    //---------------------------------------------------------------------------
    auto &header = *reinterpret_cast<Header *>(dest);
    header.block_size = kTinyBlockSize;
    //---------------------------------------------------------------------------
    // Calculate statistics on the whole datablock.
    auto db_stats = MiniStatistics<T>::generateFrom(src, size);
    header.min = db_stats.min;
    header.max = db_stats.max;
    header.cbytes = 0;
    //---------------------------------------------------------------------------
    // Apply MONOTONIC compression to the whole datablock, if possible.
    if (db_stats.step_size >= 0 && db_stats.step_size < 256) {
      header.tag = {Scheme::MONOTONIC, static_cast<u8>(db_stats.step_size)};
      return {sizeof(Header), 0};
    }
    //---------------------------------------------------------------------------
    if constexpr (!kTinyBlocksActive) {
      /// Apply TRUNCATION if tinyblocks is DEACTIVATED.
      if (db_stats.diff_bits <= 8) {
        header.tag = {Scheme::TRUNCATION, sizeof(u8)};
        header.cbytes = sizeof(u8) * size;
        return truncate<u8>(src, size, dest + sizeof(Header), header.min);
      } else if (db_stats.diff_bits <= 16) {
        header.tag = {Scheme::TRUNCATION, sizeof(u16)};
        header.cbytes = sizeof(u16) * size;
        return truncate<u16>(src, size, dest + sizeof(Header), header.min);
      } else if (db_stats.diff_bits <= 32) {
        header.tag = {Scheme::TRUNCATION, sizeof(u32)};
        header.cbytes = sizeof(u32) * size;
        return truncate<u32>(src, size, dest + sizeof(Header), header.min);
      } else {
        header.tag = {Scheme::TRUNCATION, sizeof(u64)};
        header.cbytes = sizeof(u64) * size;
        return truncate<u64>(src, size, dest + sizeof(Header), header.min);
      }
    } else {
      /// Apply TINYBLOCKS compression.
      //---------------------------------------------------------------------------
      // Prepare the header for tinyblocks compression.
      header.tag = {Scheme::NONE, 0};
      //---------------------------------------------------------------------------
      // Calculate statistics on each individual tinyblock.
      vector<Statistics<T>> stats;
      u32 block_count = size / kTinyBlockSize;
      auto read_ptr = src;
      for (u32 i = 0; i < block_count; ++i) {
        stats.push_back(Statistics<T>::generateFrom(
            read_ptr + i * kTinyBlockSize, kTinyBlockSize));
      }
      //---------------------------------------------------------------------------
      // Compress the tinyblocks.
      TinyBlocks<T, kTinyBlockSize> tb;
      CompressionDetails cd =
          scheme == nullptr
              ? tb.compress(src, size, dest + sizeof(Header), stats.data())
              : tb.compress(src, size, dest + sizeof(Header), stats.data(),
                            *scheme);
      header.cbytes = cd.header_size + cd.payload_size;
      cd.header_size += sizeof(Header);
      //---------------------------------------------------------------------------
      return cd;
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Decompress given datablock.
  /// Note: This function contains the actual decompression logic.
  /// @param dest The location to decompress the data to.
  /// @param size The number of compressed values.
  /// @param src The compressed datablocks.
  u32 decompressImpl(T *dest, const u32 size, const u8 *src) {
    assert(size % kTinyBlockSize == 0);
    //---------------------------------------------------------------------------
    if (size == 0)
      return 0;
    //---------------------------------------------------------------------------
    auto &header = *reinterpret_cast<const Header *>(src);
    //---------------------------------------------------------------------------
    if (header.tag.scheme == Scheme::MONOTONIC) [[unlikely]] {
      // Datablock decompression.
      for (u32 i = 0; i < size; ++i) {
        dest[i] = header.min + i * header.tag.payload;
      }
    } else {
      if constexpr (!kTinyBlocksActive) {
        /// Apply EXTENSION if tinyblocks is DEACTIVATED.
        assert(header.tag.scheme == Scheme::TRUNCATION);
        switch (header.tag.payload) {
        case 1:
          extend<u8>(dest, size, src + sizeof(Header), header.min);
          break;
        case 2:
          extend<u16>(dest, size, src + sizeof(Header), header.min);
          break;
        case 4:
          extend<u32>(dest, size, src + sizeof(Header), header.min);
          break;
        case 8:
          extend<u64>(dest, size, src + sizeof(Header), header.min);
          break;
        default:
          assert(false);
        }
      } else {
        /// Apply TINYBLOCKS compression.
        TinyBlocks<T, kTinyBlockSize> tb;
        tb.decompress(dest, size, src + sizeof(Header));
      }
    }
    //---------------------------------------------------------------------------
    return header.cbytes;
  }
  //---------------------------------------------------------------------------
  /// @brief Filter the datablock.
  /// @param data The compressed datablock to be filtered.
  /// @param size The number of compressed values.
  /// @param header The header for given datablock.
  /// @param predicate The predicate used for filtering.
  /// @param mv The buffer to indicate the matching values in.
  void filterImpl(const u8 *data, const u32 size, const Header &header,
                  algebra::Predicate<T> &predicate, Match *matches) {
    // Pre-Filter
    auto value = predicate.getValue();
    switch (predicate.getType()) {
    case algebra::PredicateType::EQ: {
      if (value < header.min || value > header.max)
        return;
      break;
    }
    case algebra::PredicateType::GT: {
      if (value > header.max)
        return;
      else if (value < header.min) {
        std::fill(matches, matches + size, 1);
        return;
      }
      break;
    }
    case algebra::PredicateType::LT: {
      if (value < header.min)
        return;
      else if (value > header.max) {
        std::fill(matches, matches + size, 1);
        return;
      }
      break;
    }
    case algebra::PredicateType::INEQ: {
      if (value == header.min && value == header.max)
        return;
      break;
    }
    }
    //---------------------------------------------------------------------------
    // Filter
    if (header.tag.scheme == Scheme::MONOTONIC) [[unlikely]] {
      // Datablock filter.
      switch (predicate.getType()) {
      case algebra::PredicateType::EQ: { // Equality Predicate
        if (header.tag.payload > 0) {
          if ((value - header.min) % header.tag.payload == 0)
            ++matches[(value - header.min) / header.tag.payload];
        } else {
          std::fill(matches, matches + size, 1);
        }
        return;
      }
      case algebra::PredicateType::GT: { // GreaterThan Predicate
        /// If payload is 0, all values MUST be equal to the predicate,
        /// thus no tuples qualify
        if (!header.tag.payload == 0) {
          u32 min = (value - header.min) / header.tag.payload + 1;
          for (u32 i = min; i < size; ++i) {
            ++matches[i];
          }
        }
        return;
      }
      case algebra::PredicateType::LT: { // LessThan Predicate
        /// If payload is 0, all values MUST be equal to the predicate,
        /// thus no tuples qualify
        if (!header.tag.payload == 0) {
          u32 numerator = value - header.min;
          u32 denominator = header.tag.payload;
          u32 max = (numerator + denominator - 1) / denominator;
          for (u32 i = 0; i < max; ++i) {
            ++matches[i];
          }
        }
        return;
      }
      case algebra::PredicateType::INEQ: { // Inequality Predicate
        assert(!(header.tag.payload == 0));
        std::fill(matches, matches + size, 1);
        if ((value - header.min) % header.tag.payload == 0)
          matches[(value - header.min) / header.tag.payload] = 0;
        return;
      }
      }
    } else {
      // Tinyblock filter.
      TinyBlocks<T, kTinyBlockSize> tb;
      tb.filter(data, size, predicate, matches);
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Compression-Implementation of the TRUNCATION scheme.
  /// @tparam D The datatype to truncate to.
  /// @param src The data to be compressed.
  /// @param size The number of values to be compressed.
  /// @param dest The location to compress the data to.
  /// @param reference The FOR reference.
  template <typename D>
  CompressionDetails truncate(const T *src, const u32 size, u8 *dest,
                              const u32 reference) {
    auto data = reinterpret_cast<D *>(dest);
    for (u32 i = 0; i < size; ++i) {
      data[i] = static_cast<D>(src[i] - reference);
    }
    return {sizeof(Header), sizeof(D) * size};
  }
  //---------------------------------------------------------------------------
  /// @brief Decompression-Implementation of the TRUNCATION scheme.
  /// @tparam D The datatype to extend from.
  /// @param dest The location to decompress the data to.
  /// @param size The number of compressed values.
  /// @param src The compressed datablocks.
  /// @param reference The FOR reference.
  template <typename D>
  void extend(T *dest, const u32 size, const u8 *src, const u32 reference) {
    const auto &data = reinterpret_cast<const D *>(src);
    for (u32 i = 0; i < size; ++i) {
      dest[i] = src[i] + reference;
    }
  }
};
//---------------------------------------------------------------------------
} // namespace datablock
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression