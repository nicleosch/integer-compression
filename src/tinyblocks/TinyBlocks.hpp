#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "algebra/Predicate.hpp"
#include "schemes/CompressionScheme.hpp"
#include "tinyblocks/Metadata.hpp"
#include "tinyblocks/schemes/Delta.hpp"
#include "tinyblocks/schemes/FrameOfReference.hpp"
#include "tinyblocks/schemes/Monotonic.hpp"
#include "tinyblocks/schemes/PatchedFrameOfReference.hpp"
#include "tinyblocks/schemes/RLE.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace tinyblocks {
//---------------------------------------------------------------------------
/// The class wrapping tinyblocks compression.
template <typename T, const u16 kBlockSize> class TinyBlocks {
public:
  //---------------------------------------------------------------------------
  /// A tinyblocks alignment.
  static constexpr u32 kAlignment = sizeof(T);
  //---------------------------------------------------------------------------
  /// @brief Compress given data.
  /// @param src The data to be compressed.
  /// @param size The number of values to be compressed.
  /// @param dest The location to compress the data to.
  /// @param stats Statistics on the data to be compressed.
  /// @return Statistics on the compressed data.
  CompressionDetails compress(const T *src, const u32 size, u8 *dest,
                              const Statistics<T> *stats) {
    return compressImpl(src, size, dest, stats,
                        [this](const T *src, const Statistics<T> &stat) {
                          return chooseScheme(src, stat);
                        });
  }
  //---------------------------------------------------------------------------
  /// @brief Compress given data using given scheme.
  /// @param src The data to be compressed.
  /// @param size The number of values to be compressed.
  /// @param dest The location to compress the data to.
  /// @param stats Statistics on the data to be compressed.
  /// @param scheme The scheme to compress the data with.
  /// @return Statistics on the compressed data.
  CompressionDetails compress(const T *src, const u32 size, u8 *dest,
                              const Statistics<T> *stats, const Scheme scheme) {
    return compressImpl(
        src, size, dest, stats,
        [scheme](const T *, const Statistics<T> &) { return scheme; });
  }
  //---------------------------------------------------------------------------
  /// @brief Decompress given data.
  /// @param dest The location to decompress the data to.
  /// @param size The number of compressed values.
  /// @param src The compressed data.
  void decompress(T *dest, const u32 size, const u8 *src) {
    decompress(dest, size, src, 0);
  }
  //---------------------------------------------------------------------------
  /// @brief Decompress given data starting at given offset.
  /// @param dest The location to decompress the data to.
  /// @param size The number of compressed values.
  /// @param src The compressed data.
  /// @param block_offset The offset into the header.
  void decompress(T *dest, const u32 size, const u8 *src,
                  const u32 block_offset) {
    const u32 block_count = size / kBlockSize;
    const u8 *header_ptr = src + block_offset * sizeof(Slot<T>);
    const u8 *data_ptr = src;
    //---------------------------------------------------------------------------
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      // Read header
      auto &slot = *reinterpret_cast<const Slot<T> *>(header_ptr);
      data_ptr = src + slot.offset * sizeof(T);
      //---------------------------------------------------------------------------
      // Decompress payload
      decompressDispatch(dest, data_ptr, slot);
      //---------------------------------------------------------------------------
      // Update iterators
      dest += kBlockSize;
      header_ptr += sizeof(Slot<T>);
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Filter the data based on a predicate.
  /// @param data The data to be filtered.
  /// @param size The number of compressed values.
  /// @param predicate The predicate used for filtering.
  /// @param matches The buffer to indicate the matching values in.
  void filter(const u8 *data, const u32 size,
              const algebra::Predicate<T> &predicate, Match *matches) {
    const u32 block_count = size / kBlockSize;
    //---------------------------------------------------------------------------
    const u8 *header_ptr = data;
    const u8 *data_ptr = data;
    const Match *match_ptr = matches;
    //---------------------------------------------------------------------------
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      auto &slot = *reinterpret_cast<const Slot<T> *>(header_ptr);
      data_ptr = data + slot.offset * sizeof(T);
      header_ptr = data + slot.offset * sizeof(Slot<T>);
      match_ptr = matches + slot.offset * kBlockSize;
      //---------------------------------------------------------------------------
      // Pre-Filter
      auto value = predicate.getValue();
      switch (predicate.getType()) {
      case algebra::PredicateType::EQ: {
        if (value < slot.min || value > slot.max)
          continue;
      }
      case algebra::PredicateType::GT: {
        if (value > slot.max)
          continue;
      }
      case algebra::PredicateType::LT: {
        if (value < slot.min)
          continue;
      }
      case algebra::PredicateType::INEQ: {
        if (value == slot.min && value == slot.max)
          continue;
      }
      }
      // Filter
      filterDispatch(data, slot, predicate, match_ptr);
    }
  }

private:
  /// @brief Compress given data.
  /// Note: This function contains the actual compression logic.
  /// @param src The data to be compressed.
  /// @param size The number of values to be compressed.
  /// @param dest The location to compress the data to.
  /// @param stats Statistics on the data to be compressed.
  /// @param chooseSchemeFn A function to determine the compression scheme.
  /// @return Statistics on the compressed data.
  CompressionDetails compressImpl(const T *src, const u32 size, u8 *dest,
                                  const Statistics<T> *stats,
                                  auto &&chooseSchemeFn) {
    assert(reinterpret_cast<uintptr_t>(dest) % 4 == 0 &&
           "TinyBlock destination is not 4-byte aligned!");
    assert(size <= (1ULL << (sizeof(Slot<T>::offset) * 8)) &&
           "Block is too large to reliably be addressed by offset!");
    //---------------------------------------------------------------------------
    const u32 block_count = size / kBlockSize;
    u8 *header_ptr = dest;
    u8 *data_ptr = dest;
    // TODO: Data offset should be 0 at the end of the header. Because right
    // now, if there is no compression, the header + payload are larger than
    // 2^16 * sizeof(T).
    // -> serious bug, because end is not addressable
    u32 data_offset = block_count * sizeof(Slot<T>);
    //---------------------------------------------------------------------------
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      data_ptr = dest + data_offset;
      //---------------------------------------------------------------------------
      // Update header
      auto &slot = *reinterpret_cast<Slot<T> *>(header_ptr);
      slot.reference = stats[block_i].min;
      slot.max = stats[block_i].max;
      slot.offset = data_offset / kAlignment;
      //---------------------------------------------------------------------------
      // Compress
      Scheme scheme = chooseSchemeFn(src, stats[block_i]);
      u32 compressed_size =
          compressDispatch(src, data_ptr, stats[block_i], slot, scheme);
      //---------------------------------------------------------------------------
      // Align the offset.
      data_offset += (compressed_size + (kAlignment - 1)) & ~(kAlignment - 1);
      //---------------------------------------------------------------------------
      // Update iterators
      src += kBlockSize;
      header_ptr += sizeof(Slot<T>);
    }
    //---------------------------------------------------------------------------
    u64 header_size = header_ptr - dest;
    u64 payload_size = data_offset - header_size;
    return {header_size, payload_size};
  }
  //---------------------------------------------------------------------------
  /// @brief Choose the scheme that yields the smallest size.
  /// Note: This is a naive approach, which will be improved in the future.
  /// @param src The tinyblock to be compressed.
  /// @param stats Statistics on the given tinyblock.
  /// @return The chosen scheme.
  Scheme chooseScheme(const T *src, const Statistics<T> &stats) {
    if (stats.step_size >= 0 && stats.step_size < 256) {
      return Scheme::MONOTONIC;
    }
    //---------------------------------------------------------------------------
    // The buffer to compress into.
    auto dest = std::make_unique<u8[]>(kBlockSize * sizeof(T) * 2);
    //---------------------------------------------------------------------------
    // A dummy slot.
    Slot<T> slot{stats.min, {}, {}};
    //---------------------------------------------------------------------------
    // Compress to find best compressing scheme.
    unordered_map<Scheme, u32> scheme2size;
    scheme2size[Scheme::FOR] =
        frameofreference::compress<T, kBlockSize>(src, dest.get(), stats, slot);
    scheme2size[Scheme::RLE4] =
        rle::compress<T, kBlockSize, 4>(src, dest.get(), stats, slot);
    scheme2size[Scheme::RLE8] =
        rle::compress<T, kBlockSize, 8>(src, dest.get(), stats, slot);
    scheme2size[Scheme::PFOR] =
        pframeofreference::compress<T, kBlockSize, Scheme::PFOR>(
            src, dest.get(), slot);
    scheme2size[Scheme::PFOR_EBP] =
        pframeofreference::compress<T, kBlockSize, Scheme::PFOR_EBP>(
            src, dest.get(), slot);
    scheme2size[Scheme::PFOR_EP] =
        pframeofreference::compress<T, kBlockSize, Scheme::PFOR_EP>(
            src, dest.get(), slot);
    scheme2size[Scheme::PFOR_LEMIRE] =
        pframeofreference::compress<T, kBlockSize, Scheme::PFOR_LEMIRE>(
            src, dest.get(), slot);
    if (stats.delta) {
      scheme2size[Scheme::DELTA] =
          delta::compress<T, kBlockSize>(src, dest.get(), slot);
      scheme2size[Scheme::PFOR_DELTA] =
          pframeofreference::compress<T, kBlockSize, Scheme::PFOR_DELTA>(
              src, dest.get(), slot);
    }
    //---------------------------------------------------------------------------
    // Find minimum size.
    auto min = std::min_element(
        scheme2size.begin(), scheme2size.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; });
    //---------------------------------------------------------------------------
    return min->first;
  }
  //---------------------------------------------------------------------------
  /// @brief Dispatch the compression functions based on given scheme.
  /// @param src The tinyblock to be compressed.
  /// @param dest The location to compress the tinyblock to.
  /// @param stats Statistics on the tinyblock to be compressed.
  /// @param slot The header for given tinyblock.
  /// @param scheme The scheme to compress the tinyblock with.
  /// @return The size of the compressed data in bytes.
  u16 compressDispatch(const T *src, u8 *dest, const Statistics<T> &stats,
                       Slot<T> &slot, Scheme scheme) {
    switch (scheme) {
    case Scheme::MONOTONIC:
      return monotonic::compress<T, kBlockSize>(src, dest, stats, slot);
    case Scheme::FOR:
      return frameofreference::compress<T, kBlockSize>(src, dest, stats, slot);
    case Scheme::RLE4:
      return rle::compress<T, kBlockSize, 4>(src, dest, stats, slot);
    case Scheme::RLE8:
      return rle::compress<T, kBlockSize, 8>(src, dest, stats, slot);
    case Scheme::DELTA:
      return delta::compress<T, kBlockSize>(src, dest, slot);
    case Scheme::PFOR:
      return pframeofreference::compress<T, kBlockSize, Scheme::PFOR>(src, dest,
                                                                      slot);
    case Scheme::PFOR_EBP:
      return pframeofreference::compress<T, kBlockSize, Scheme::PFOR_EBP>(
          src, dest, slot);
    case Scheme::PFOR_EP:
      return pframeofreference::compress<T, kBlockSize, Scheme::PFOR_EP>(
          src, dest, slot);
    case Scheme::PFOR_DELTA:
      return pframeofreference::compress<T, kBlockSize, Scheme::PFOR_DELTA>(
          src, dest, slot);
    case Scheme::PFOR_LEMIRE:
      return pframeofreference::compress<T, kBlockSize, Scheme::PFOR_LEMIRE>(
          src, dest, slot);
    default:
      throw std::runtime_error(
          "Compression failed: Provided scheme does not exist.");
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Dispatch the decompression functions based on the encoding scheme.
  /// @param dest The location to decompress the tinyblock to.
  /// @param src The compressed tinyblock.
  /// @param slot The header for given tinyblock.
  void decompressDispatch(T *dest, const u8 *src, const Slot<T> &slot) {
    switch (slot.opcode.scheme) {
    case Scheme::MONOTONIC:
      monotonic::decompress<T, kBlockSize>(dest, slot);
      return;
    case Scheme::FOR:
      frameofreference::decompress<T, kBlockSize>(dest, src, slot);
      return;
    case Scheme::RLE4:
      rle::decompress<T, kBlockSize, 4>(dest, src, slot);
      return;
    case Scheme::RLE8:
      rle::decompress<T, kBlockSize, 8>(dest, src, slot);
      return;
    case Scheme::DELTA:
      delta::decompress<T, kBlockSize>(dest, src, slot);
      return;
    case Scheme::PFOR:
      pframeofreference::decompress<T, kBlockSize, Scheme::PFOR>(dest, src,
                                                                 slot);
      return;
    case Scheme::PFOR_EBP:
      pframeofreference::decompress<T, kBlockSize, Scheme::PFOR_EBP>(dest, src,
                                                                     slot);
      return;
    case Scheme::PFOR_EP:
      pframeofreference::decompress<T, kBlockSize, Scheme::PFOR_EP>(dest, src,
                                                                    slot);
      return;
    case Scheme::PFOR_DELTA:
      pframeofreference::decompress<T, kBlockSize, Scheme::PFOR_DELTA>(
          dest, src, slot);
      return;
    case Scheme::PFOR_LEMIRE:
      pframeofreference::decompress<T, kBlockSize, Scheme::PFOR_LEMIRE>(
          dest, src, slot);
      return;
    default:
      throw std::runtime_error(
          "Decompression failed: Provided opcode does not exist.");
    }
  }
  //---------------------------------------------------------------------------
  /// @brief Dispatch the filtering functions based on the encoding scheme.
  /// @param data The tinyblock to be filtered.
  /// @param slot The header for given tinyblock.
  /// @param predicate The predicate used for filtering.
  /// @param matches The buffer to indicate the matching values in.
  void filterDispatch(const u8 *data, const Slot<T> &slot,
                      const algebra::Predicate<T> predicate, Match *matches) {
    switch (slot.opcode.scheme) {
    case Scheme::MONOTONIC:
      monotonic::filter<T, kBlockSize>(data, slot, predicate, matches);
      return;
    case Scheme::FOR:
      frameofreference::filter<T, kBlockSize>(data, predicate, matches);
      return;
    case Scheme::RLE4:
      rle::filter<T, kBlockSize, 4>(data, predicate, matches);
      return;
    case Scheme::RLE8:
      rle::filter<T, kBlockSize, 8>(data, predicate, matches);
      return;
    case Scheme::DELTA:
      delta::filter<T, kBlockSize>(data, predicate, matches);
      return;
    case Scheme::PFOR:
      pframeofreference::filter<T, kBlockSize, Scheme::PFOR>(data, predicate,
                                                             matches);
      return;
    case Scheme::PFOR_EBP:
      pframeofreference::filter<T, kBlockSize, Scheme::PFOR_EBP>(
          data, predicate, matches);
      return;
    case Scheme::PFOR_EP:
      pframeofreference::filter<T, kBlockSize, Scheme::PFOR_EP>(data, predicate,
                                                                matches);
      return;
    case Scheme::PFOR_DELTA:
      pframeofreference::filter<T, kBlockSize, Scheme::PFOR_DELTA>(
          data, predicate, matches);
      return;
    case Scheme::PFOR_LEMIRE:
      pframeofreference::filter<T, kBlockSize, Scheme::PFOR_LEMIRE>(
          data, predicate, matches);
      return;
    default:
      throw std::runtime_error(
          "Filtering failed: Provided opcode does not exist.");
    }
  }
  //---------------------------------------------------------------------------
}; // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression