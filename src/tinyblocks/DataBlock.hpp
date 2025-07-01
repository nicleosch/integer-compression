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
template <typename T, const u16 kTinyBlockSize,
          const u32 kBlockSize = kDefaultSize>
class DataBlock {
public:
  //---------------------------------------------------------------------------
  static_assert(kBlockSize % kTinyBlockSize == 0,
                "Datablock size must be multiple of Tinyblock size.");
  //---------------------------------------------------------------------------
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
  /// Compress the datablocks.
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
  /// Decompress the datablocks.
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
  /// Filter the datablock.
  void filter(const T *data, const u32 size,
              const algebra::Predicate<T> &predicate, MatchVector &mv) {
    mv.resize(size);
    //---------------------------------------------------------------------------
    const u32 cblock = size / kBlockSize;
    auto read_ptr = data;
    auto match_ptr = mv.data();
    //---------------------------------------------------------------------------
    for (u32 i = 0; i < cblock; ++i) {
      auto header = *reinterpret_cast<Header *>(read_ptr);
      filterImpl(read_ptr + sizeof(Header), kBlockSize, predicate, match_ptr);
      read_ptr += header.cbytes;
      match_ptr += kBlockSize;
    }
    //---------------------------------------------------------------------------
    filterImpl(read_ptr + sizeof(Header), size % kBlockSize, predicate,
               match_ptr);
  }

private:
  //---------------------------------------------------------------------------
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
    // Apply scheme to the whole datablock, if possible.
    if (db_stats.step_size >= 0 && db_stats.step_size < 256) {
      header.tag = {Scheme::MONOTONIC, static_cast<u8>(db_stats.step_size)};
      return {sizeof(Header), 0};
    }
    //---------------------------------------------------------------------------
    // Prepare the header for tinyblocks compression.
    header.tag = {Scheme::NONE, 0};
    //---------------------------------------------------------------------------
    // Calculate statistics on each individual tinyblock.
    vector<Statistics<T>> stats;
    u32 block_count = size / kTinyBlockSize;
    auto read_ptr = src;
    for (u32 i = 0; i < block_count; ++i) {
      stats.push_back(Statistics<T>::generateFrom(read_ptr + i * kTinyBlockSize,
                                                  kTinyBlockSize));
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
  //---------------------------------------------------------------------------
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
      // Tinyblock decompression.
      TinyBlocks<T, kTinyBlockSize> tb;
      tb.decompress(dest, size, src + sizeof(Header));
    }
    //---------------------------------------------------------------------------
    return header.cbytes;
  }
  //---------------------------------------------------------------------------
  void filterImpl(const T *data, const u32 size, const Header &header,
                  const algebra::Predicate<T> predicate, u8 *matches) {
    // Pre-Filter
    auto value = predicate.getValue();
    switch (predicate.getType()) {
    case algebra::PredicateType::EQ: {
      if (value < header.min || value > header.max)
        return;
    }
    case algebra::PredicateType::GT: {
      if (value > header.max)
        return;
    }
    case algebra::PredicateType::LT: {
      if (value < header.min)
        return;
    }
    case algebra::PredicateType::INEQ: {
      if (value == header.min && value == header.max)
        return;
    }
    }
    //---------------------------------------------------------------------------
    // Filter
    if (header.tag.scheme == Scheme::MONOTONIC) [[unlikely]] {
      // Datablock filter.
      for (u32 i = 0; i < size; ++i) {
        switch (predicate.getType()) {
        case algebra::PredicateType::EQ: { // Equality Predicate
          if (header.tag.payload > 0 &&
              (value - header.min) % header.tag.payload == 0) {
            ++matches[(value - header.min) / header.tag.payload];
          } else {
            std::fill(matches, matches + size, 1);
          }
        }
        case algebra::PredicateType::GT: { // GreaterThan Predicate
          u32 min = (value - header.min) / header.tag.payload + 1;
          for (u32 i = min; i < size; ++i) {
            ++matches[i];
          }
        }
        case algebra::PredicateType::LT: { // LessThan Predicate
          u32 numerator = value - header.min;
          u32 denominator = header.tag.payload;
          u32 max = (numerator + denominator - 1) / denominator;
          for (u32 i = 0; i < max; ++i) {
            ++matches[i];
          }
        }
        case algebra::PredicateType::INEQ: { // Inequality Predicate
          assert(!(header.tag.payload == 0));
          std::fill(matches, matches + size, 1);
          if ((value - header.min) % header.tag.payload == 0)
            matches[(value - header.min) / header.tag.payload] = 0;
        }
        }
      }
    } else {
      // Tinyblock filter.
      TinyBlocks<T, kTinyBlockSize> tb;
      tb.filter(data, size, predicate, matches);
    }
  }
};
//---------------------------------------------------------------------------
} // namespace datablock
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression