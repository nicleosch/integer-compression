#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
#include "extern/fastpfor/BitPacking.hpp"
#include "extern/fastpfor/Delta.hpp"
#include "extern/fastpfor/PFOR.hpp"
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace tinyblocks {
//---------------------------------------------------------------------------
/// The scheme used for compression within a block.
enum class Scheme : u8 {
  MONOTONIC = 0,
  FOR = 1,
  RLE4 = 2,
  RLE8 = 3,
  DELTA = 4,
  PFOR = 5,
  PFOR_EBP = 6,
  PFOR_EP = 7,
  PFOR_DELTA = 8,
  PFOR_LEMIRE = 9,
};
//---------------------------------------------------------------------------
/// The opcode stored in the header slot.
struct Opcode {
  /// The compression scheme.
  Scheme scheme;
  /// Additional meta data required for the scheme.
  u8 payload;
};
//---------------------------------------------------------------------------
/// An abstraction for a block's 3 byte offset.
struct Offset {
  /// The raw offset.
  u8 value[3];
  /// Get the offset as 32-bit integer.
  u32 get() const {
    return (static_cast<u32>(value[0]) << 16) |
           (static_cast<u32>(value[1]) << 8) | value[2];
  }
  /// Set the offset to provided value.
  void set(u32 v) {
    value[0] = (v >> 16) & 0xFF;
    value[1] = (v >> 8) & 0xFF;
    value[2] = v & 0xFF;
  }
};
//---------------------------------------------------------------------------
/// The class wrapping tinyblocks compression.
template <typename DataType, const u16 kBlockSize> class TinyBlocks {
public:
  //---------------------------------------------------------------------------
  using RunLength = u8;
  //---------------------------------------------------------------------------
  /// The slot stored in the header per block.
  struct __attribute__((packed)) Slot {
    /// The reference of the corresponding frame.
    DataType reference;
    /// The offset into the data array.
    Offset offset;
    /// The number of bits used to store an integer in corresponding frame.
    Opcode opcode;
  };

  //---------------------------------------------------------------------------
  // COMPRESSION
  //---------------------------------------------------------------------------
  CompressionDetails compress(const DataType *src, const u32 size, u8 *dest,
                              const Statistics<DataType> *stats) {
    return compressImpl(
        src, size, dest, stats,
        [this](const DataType *src, const Statistics<DataType> &stat) {
          return chooseScheme(src, stat);
        });
  }
  //---------------------------------------------------------------------------
  // Compress given a specific opcode.
  CompressionDetails compress(const DataType *src, const u32 size, u8 *dest,
                              const Statistics<DataType> *stats,
                              const Scheme scheme) {
    return compressImpl(
        src, size, dest, stats,
        [scheme](const DataType *, const Statistics<DataType> &) {
          return scheme;
        });
  }

  //---------------------------------------------------------------------------
  // DECOMPRESSION
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src) {
    decompress(dest, size, src, 0);
  }
  //---------------------------------------------------------------------------
  // Decompress starting from a given block offset.
  void decompress(DataType *dest, const u32 size, const u8 *src,
                  const u32 block_offset) {
    const u32 block_count = size / kBlockSize;
    const u8 *header_ptr = src + block_offset * sizeof(Slot);
    const u8 *data_ptr = src;
    //---------------------------------------------------------------------------
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      // Read header
      auto &slot = *reinterpret_cast<const Slot *>(header_ptr);
      data_ptr = src + slot.offset.get();
      //---------------------------------------------------------------------------
      // Decompress payload
      decompressDispatch(dest, data_ptr, slot);
      //---------------------------------------------------------------------------
      // Update iterators
      dest += kBlockSize;
      header_ptr += sizeof(Slot);
    }
  }

private:
  //---------------------------------------------------------------------------
  // COMPRESSION HELPERS
  //---------------------------------------------------------------------------
  CompressionDetails compressImpl(const DataType *src, const u32 size, u8 *dest,
                                  const Statistics<DataType> *stats,
                                  auto &&chooseSchemeFn) {
    const u32 block_count = size / kBlockSize;
    u8 *header_ptr = dest;
    u8 *data_ptr = dest;
    u32 data_offset = block_count * sizeof(Slot);
    //---------------------------------------------------------------------------
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      data_ptr = dest + data_offset;
      //---------------------------------------------------------------------------
      // Update header
      auto &slot = *reinterpret_cast<Slot *>(header_ptr);
      slot.reference = stats[block_i].min;
      slot.offset.set(data_offset);
      //---------------------------------------------------------------------------
      // Compress
      Scheme scheme = chooseSchemeFn(src, stats[block_i]);
      data_offset +=
          compressDispatch(src, data_ptr, stats[block_i], slot, scheme);
      //---------------------------------------------------------------------------
      // Update iterators
      src += kBlockSize;
      header_ptr += sizeof(Slot);
    }
    //---------------------------------------------------------------------------
    u64 header_size = header_ptr - dest;
    u64 payload_size = data_offset - header_size;
    return {header_size, payload_size};
  }
  //---------------------------------------------------------------------------
  // Choose the scheme that yields the smallest size.
  // Note: This is a naive approach, which will be improved in the future.
  Scheme chooseScheme(const DataType *src, const Statistics<DataType> &stats) {
    if (stats.step_size >= 0 && stats.step_size < 256) {
      return Scheme::MONOTONIC;
    }
    //---------------------------------------------------------------------------
    // The buffer to compress into.
    auto dest = std::make_unique<u8[]>(kBlockSize * sizeof(DataType) * 2);
    //---------------------------------------------------------------------------
    // A dummy slot.
    Slot slot{stats.min, {}, {}};
    //---------------------------------------------------------------------------
    // Compress to find best compressing scheme.
    unordered_map<Scheme, u32> scheme2size;
    scheme2size[Scheme::FOR] = compressFOR(src, dest.get(), stats, slot);
    scheme2size[Scheme::RLE4] = compressRLE<4>(src, dest.get(), stats, slot);
    scheme2size[Scheme::RLE8] = compressRLE<8>(src, dest.get(), stats, slot);
    scheme2size[Scheme::PFOR] = compressPFOR(src, dest.get(), slot);
    scheme2size[Scheme::PFOR_EBP] = compressPFOREBP(src, dest.get(), slot);
    scheme2size[Scheme::PFOR_EP] = compressPFOREP(src, dest.get(), slot);
    scheme2size[Scheme::PFOR_LEMIRE] =
        compressPFORLemire(src, dest.get(), slot);
    if (stats.delta) {
      scheme2size[Scheme::DELTA] = compressDelta(src, dest.get(), slot);
      scheme2size[Scheme::PFOR_DELTA] =
          compressPFORDelta(src, dest.get(), slot);
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
  u16 compressDispatch(const DataType *src, u8 *dest,
                       const Statistics<DataType> &stats, Slot &slot,
                       Scheme scheme) {
    switch (scheme) {
    case Scheme::MONOTONIC:
      return compressMonotonic(src, dest, stats, slot);
    case Scheme::FOR:
      return compressFOR(src, dest, stats, slot);
    case Scheme::RLE4:
      return compressRLE<4>(src, dest, stats, slot);
    case Scheme::RLE8:
      return compressRLE<8>(src, dest, stats, slot);
    case Scheme::DELTA:
      return compressDelta(src, dest, slot);
    case Scheme::PFOR:
      return compressPFOR(src, dest, slot);
    case Scheme::PFOR_EBP:
      return compressPFOREBP(src, dest, slot);
    case Scheme::PFOR_EP:
      return compressPFOREP(src, dest, slot);
    case Scheme::PFOR_DELTA:
      return compressPFORDelta(src, dest, slot);
    case Scheme::PFOR_LEMIRE:
      return compressPFORLemire(src, dest, slot);
    default:
      throw std::runtime_error(
          "Compression failed: Provided scheme does not exist.");
    }
  }

  //---------------------------------------------------------------------------
  // DECOMPRESSION HELPERS
  //---------------------------------------------------------------------------
  void decompressDispatch(DataType *dest, const u8 *src, const Slot &slot) {
    switch (slot.opcode.scheme) {
    case Scheme::MONOTONIC:
      decompressMonotonic(dest, slot);
      return;
    case Scheme::FOR:
      decompressFOR(dest, src, slot);
      return;
    case Scheme::RLE4:
      decompressRLE<4>(dest, src, slot);
      return;
    case Scheme::RLE8:
      decompressRLE<8>(dest, src, slot);
      return;
    case Scheme::DELTA:
      decompressDelta(dest, src, slot);
      return;
    case Scheme::PFOR:
      decompressPFOR(dest, src, slot);
      return;
    case Scheme::PFOR_EBP:
      decompressPFOREBP(dest, src, slot);
      return;
    case Scheme::PFOR_EP:
      decompressPFOREP(dest, src, slot);
      return;
    case Scheme::PFOR_DELTA:
      decompressPFORDelta(dest, src, slot);
      return;
    case Scheme::PFOR_LEMIRE:
      decompressPFORLemire(dest, src, slot);
      return;
    default:
      throw std::runtime_error(
          "Decompression failed: Provided opcode does not exist.");
    }
  }

  //---------------------------------------------------------------------------
  // MONOTONIC
  //---------------------------------------------------------------------------
  u32 compressMonotonic(const DataType *src, u8 *dest,
                        const Statistics<DataType> &stats, Slot &slot) {
    assert(stats.step_size >= 0 && stats.step_size < 256);
    slot.opcode = {Scheme::MONOTONIC, static_cast<u8>(stats.step_size)};
    return 0;
  }
  //---------------------------------------------------------------------------
  void decompressMonotonic(DataType *dest, const Slot &slot) {
    for (u16 i = 0; i < kBlockSize; ++i) {
      dest[i] = slot.reference + i * slot.opcode.payload;
    }
  }

  //---------------------------------------------------------------------------
  // FOR
  //---------------------------------------------------------------------------
  u32 compressFOR(const DataType *src, u8 *dest,
                  const Statistics<DataType> &stats, Slot &slot) {
    assert(stats.diff_bits >= 0 && stats.diff_bits <= sizeof(DataType) * 8);
    slot.opcode = {Scheme::FOR, stats.diff_bits};
    //---------------------------------------------------------------------------
    // Normalize
    vector<DataType> normalized(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      normalized[i] = src[i] - slot.reference;
    }
    //---------------------------------------------------------------------------
    // Compress
    u8 pack_size = slot.opcode.payload;
    return bitpacking::pack<DataType, kBlockSize>(normalized.data(), dest,
                                                  pack_size);
  }
  //---------------------------------------------------------------------------
  void decompressFOR(DataType *dest, const u8 *src, const Slot &slot) {
    // Decompress
    bitpacking::unpack<DataType, kBlockSize>(dest, src, slot.opcode.payload);
    //---------------------------------------------------------------------------
    // Denormalize
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  }

  //---------------------------------------------------------------------------
  // RLE
  //---------------------------------------------------------------------------
  template <u8 kLengthBits>
  u32 compressRLE(const DataType *src, u8 *dest,
                  const Statistics<DataType> &stats, Slot &slot) {
    static_assert(kLengthBits == 4 || kLengthBits == 8,
                  "Run-Lengths must be encoded in either 4 or 8 bits.");
    assert(stats.diff_bits >= 0 && stats.diff_bits <= sizeof(DataType) * 8);
    //---------------------------------------------------------------------------
    slot.opcode = {kLengthBits == 4 ? Scheme::RLE4 : Scheme::RLE8,
                   stats.diff_bits};
    //---------------------------------------------------------------------------
    constexpr RunLength max_length = (kLengthBits == 4) ? 15 : 255;
    //---------------------------------------------------------------------------
    // Data to be stored
    u16 run_count;
    vector<DataType> values;
    vector<RunLength> lengths;
    //---------------------------------------------------------------------------
    // Iterators
    DataType cur = src[0];
    RunLength run_length = 1;
    //---------------------------------------------------------------------------
    // Helpers for bit manipulation
    u8 shift = 0;
    u8 length_byte = 0;
    auto appendLength = [&]() {
      if constexpr (kLengthBits == 8) {
        lengths.push_back(run_length);
      } else {
        length_byte |= (run_length << (4 - shift));
        shift = (shift + 4) % 8;
        if (shift == 0) {
          lengths.push_back(length_byte);
          length_byte = 0;
        }
      }
    };
    //---------------------------------------------------------------------------
    // Fill length & value arrays
    for (u32 i = 1; i < kBlockSize; ++i) {
      if (src[i] == cur) {
        if (run_length == max_length) {
          appendLength();
          values.push_back(cur - slot.reference);
          run_length = 0;
        }
        ++run_length;
      } else {
        appendLength();
        values.push_back(cur - slot.reference);
        run_length = 1;
        cur = src[i];
      }
    }
    if constexpr (kLengthBits == 8) {
      lengths.push_back(run_length);
    } else {
      lengths.push_back(length_byte | (run_length << (4 - shift)));
    }
    values.push_back(cur - slot.reference);
    run_count = values.size();
    assert(((kLengthBits == 4) ? (values.size() + 1) / 2 : values.size()) ==
           lengths.size());
    //---------------------------------------------------------------------------
    // Write lengths
    auto write_ptr = dest;
    std::memcpy(write_ptr, &run_count, sizeof(run_count));
    write_ptr += sizeof(run_count);
    std::memcpy(write_ptr, lengths.data(), lengths.size() * sizeof(RunLength));
    //---------------------------------------------------------------------------
    // Write values
    u16 value_offset = lengths.size() * sizeof(RunLength);
    write_ptr += value_offset;
    u32 values_size = pfor::packAdaptive(
        values, reinterpret_cast<u32 *>(write_ptr), slot.opcode.payload);
    //---------------------------------------------------------------------------
    return sizeof(run_count) + value_offset + values_size;
  }
  //---------------------------------------------------------------------------
  template <u8 kLengthBits>
  void decompressRLE(DataType *dest, const u8 *src, const Slot &slot) {
    static_assert(kLengthBits == 4 || kLengthBits == 8,
                  "Run-Lengths must be encoded in either 4 or 8 bits.");
    //---------------------------------------------------------------------------
    // Decode meta data
    auto run_count = utils::unalignedLoad<u16>(src);
    //---------------------------------------------------------------------------
    u16 length_bytes;
    if constexpr (kLengthBits == 4) {
      length_bytes = ((run_count + 1) / 2);
    } else {
      length_bytes = run_count;
    }
    u16 value_offset = sizeof(run_count) + length_bytes;
    //---------------------------------------------------------------------------
    // Decompress lengths & values
    auto read_ptr = src + value_offset;
    // Unpack
    vector<DataType> values(run_count);
    pfor::unpackAdaptive<DataType>(
        values, reinterpret_cast<const u32 *>(read_ptr), slot.opcode.payload);
    // Initialize lengths
    // TODO: Can be improved probably, currently done to prevent UB
    vector<RunLength> lengths(length_bytes);
    std::memcpy(lengths.data(), src + sizeof(run_count), length_bytes);
    //---------------------------------------------------------------------------
    // Decode RLE using AVX2
    auto write_ptr = dest;
    u16 shift = 0;
    for (u16 i = 0; i < run_count; ++i) {
      RunLength length;
      if constexpr (kLengthBits == 4) {
        length = (lengths[i / 2] >> (4 - (shift % 8))) & 0x0F;
      } else {
        length = lengths[i];
      }
      auto end_ptr = write_ptr + length;
      DataType value = values[i] + slot.reference;

      if constexpr (std::is_same_v<DataType, INTEGER>) {
        __m256i broadcast = _mm256_set1_epi32(value);
        while (write_ptr + 8 <= end_ptr) {
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(write_ptr),
                              broadcast);
          write_ptr += 8;
        }
      } else if constexpr (std::is_same_v<DataType, BIGINT>) {
        __m256i broadcast = _mm256_set1_epi64x(value);
        while (write_ptr + 4 <= end_ptr) {
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(write_ptr),
                              broadcast);
          write_ptr += 4;
        }
      } else {
        static_assert(false,
                      "Unsupported DataType for TinyBlocks RLE decompression.");
      }
      while (write_ptr < end_ptr) {
        *write_ptr++ = value;
      }
      shift += 4;
    }
  }

  //---------------------------------------------------------------------------
  // DELTA
  //---------------------------------------------------------------------------
  u32 compressDelta(const DataType *src, u8 *dest, Slot &slot) {
    // FOR
    auto buffer = std::make_unique<DataType[]>(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      buffer[i] = src[i] - slot.reference;
    }
    //---------------------------------------------------------------------------
    // Delta
    pfor::delta::compress<DataType, kBlockSize>(buffer.get());
    //---------------------------------------------------------------------------
    // Bitpacking
    //---------------------------------------------------------------------------
    // Prepare
    DataType min = 0;
    DataType max = 0;
    for (u16 i = 0; i < kBlockSize; ++i) {
      if (buffer[i] < min)
        min = buffer[i];
      if (buffer[i] > max)
        max = buffer[i];
    }
    u8 pack_size = utils::requiredBits(max - min);
    //---------------------------------------------------------------------------
    slot.opcode = {Scheme::DELTA, pack_size};
    //---------------------------------------------------------------------------
    // Compress
    u32 compressed_size =
        bitpacking::pack<DataType, kBlockSize>(buffer.get(), dest, pack_size);
    //---------------------------------------------------------------------------
    return compressed_size;
  }
  //---------------------------------------------------------------------------
  void decompressDelta(DataType *dest, const u8 *src, const Slot &slot) {
    // Bitpacking
    bitpacking::unpack<DataType, kBlockSize>(dest, src, slot.opcode.payload);
    //---------------------------------------------------------------------------
    // Delta
    pfor::delta::decompress<DataType, kBlockSize>(dest);
    //---------------------------------------------------------------------------
    // FOR
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  }

  //---------------------------------------------------------------------------
  // PFOR
  //---------------------------------------------------------------------------
  u32 compressPFOR(const DataType *src, u8 *dest, Slot &slot) {
    u8 pack_size;
    slot.opcode = {Scheme::PFOR, pack_size};
    //---------------------------------------------------------------------------
    // Compress
    return pfor::compressPFOR<DataType, kBlockSize>(src, dest, slot.reference,
                                                    slot.opcode.payload);
  }
  //---------------------------------------------------------------------------
  void decompressPFOR(DataType *dest, const u8 *src, const Slot &slot) {
    pfor::decompressPFOR<DataType, kBlockSize>(dest, src, slot.reference,
                                               slot.opcode.payload);
  }

  //---------------------------------------------------------------------------
  // PFOR_EBP
  //---------------------------------------------------------------------------
  u32 compressPFOREBP(const DataType *src, u8 *dest, Slot &slot) {
    u8 pack_size;
    slot.opcode = {Scheme::PFOR_EBP, pack_size};
    //---------------------------------------------------------------------------
    // Compress
    return pfor::compressPFOREBP<DataType, kBlockSize>(
        src, dest, slot.reference, slot.opcode.payload);
  }
  //---------------------------------------------------------------------------
  void decompressPFOREBP(DataType *dest, const u8 *src, const Slot &slot) {
    pfor::decompressPFOREBP<DataType, kBlockSize>(dest, src, slot.reference,
                                                  slot.opcode.payload);
  }

  //---------------------------------------------------------------------------
  // PFOR_EP
  //---------------------------------------------------------------------------
  u32 compressPFOREP(const DataType *src, u8 *dest, Slot &slot) {
    u8 pack_size;
    slot.opcode = {Scheme::PFOR_EP, pack_size};
    //---------------------------------------------------------------------------
    // Compress
    return pfor::compressPFOREP<DataType, kBlockSize>(src, dest, slot.reference,
                                                      slot.opcode.payload);
  }
  //---------------------------------------------------------------------------
  void decompressPFOREP(DataType *dest, const u8 *src, const Slot &slot) {
    pfor::decompressPFOREP<DataType, kBlockSize>(dest, src, slot.reference,
                                                 slot.opcode.payload);
  }
  //---------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // PFOR_DELTA
  //---------------------------------------------------------------------------
  u32 compressPFORDelta(const DataType *src, u8 *dest, Slot &slot) {
    slot.opcode = {Scheme::PFOR_DELTA, 0};
    //---------------------------------------------------------------------------
    // Normalize
    vector<DataType> normalized(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      normalized[i] = src[i] - slot.reference;
    }
    //---------------------------------------------------------------------------
    // Compress Delta
    pfor::delta::compress<DataType, kBlockSize>(normalized.data());
    //---------------------------------------------------------------------------
    // Prepare
    DataType min = 0;
    DataType max = 0;
    for (u16 i = 0; i < kBlockSize; ++i) {
      if (normalized[i] < min)
        min = normalized[i];
      if (normalized[i] > max)
        max = normalized[i];
    }
    u8 pack_size = utils::requiredBits(max - min);
    //---------------------------------------------------------------------------
    // Compress PFOR
    return pfor::compressPFORLemire<DataType, kBlockSize>(
        normalized.data(), dest, slot.opcode.payload);
  }
  //---------------------------------------------------------------------------
  void decompressPFORDelta(DataType *dest, const u8 *src, const Slot &slot) {
    // Decompress
    pfor::decompressPFORLemire<DataType, kBlockSize>(dest, src,
                                                     slot.opcode.payload);
    //---------------------------------------------------------------------------
    // Delta
    pfor::delta::decompress<DataType, kBlockSize>(dest);
    //---------------------------------------------------------------------------
    // Denormalize
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  }

  //---------------------------------------------------------------------------
  // PFOR_Lemire
  //---------------------------------------------------------------------------
  u32 compressPFORLemire(const DataType *src, u8 *dest, Slot &slot) {
    u8 pack_size;
    slot.opcode = {Scheme::PFOR_LEMIRE, pack_size};
    //---------------------------------------------------------------------------
    // Normalize
    vector<DataType> normalized(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      normalized[i] = src[i] - slot.reference;
    }
    //---------------------------------------------------------------------------
    // Compress
    return pfor::compressPFORLemire<DataType, kBlockSize>(
        normalized.data(), dest, slot.opcode.payload);
  }
  //---------------------------------------------------------------------------
  void decompressPFORLemire(DataType *dest, const u8 *src, const Slot &slot) {
    // Decompress
    pfor::decompressPFORLemire<DataType, kBlockSize>(dest, src,
                                                     slot.opcode.payload);
    //---------------------------------------------------------------------------
    // Denormalize
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  }
  //---------------------------------------------------------------------------
}; // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression