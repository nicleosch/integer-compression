#pragma once
//---------------------------------------------------------------------------
#include <cmath>
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
#include "schemes/CompressionScheme.hpp"
#include "simd/delta.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
template <typename DataType, const u16 kBlockSize>
class TinyBlocks : public CompressionScheme<DataType> {
public:
  //---------------------------------------------------------------------------
  struct __attribute__((packed)) Slot {
    /// The reference of the corresponding frame.
    DataType reference;
    /// The offset into the data array.
    u32 offset;
    /// The number of bits used to store an integer in corresponding frame.
    u8 opcode;
  };
  //---------------------------------------------------------------------------
  enum class Opcode : u8 {
    ONE_VALUE = 0,
    BIT_PACKING = 1,
    RLE = 65,
    DELTA = 66,
    MONOTONICALLY_INCREASING = 67
  };
  //---------------------------------------------------------------------------
  CompressionDetails compress(const DataType *src, const u32 size, u8 *dest,
                              const Statistics<DataType> *stats) override {
    const u32 block_count = size / kBlockSize;

    // Layout: HEADER [kBlockSize] | COMPRESSED DATA
    u8 *header_ptr = dest;
    u8 *data_ptr = dest;

    u32 data_offset = block_count * sizeof(Slot);
    for (u32 block_i = 0; block_i < block_count; ++block_i) {
      data_ptr = dest + data_offset;

      // Update header
      auto &slot = *reinterpret_cast<Slot *>(header_ptr);
      slot.reference = stats[block_i].min;
      slot.offset = data_offset;

      // Compress block
      Opcode opcode = chooseOpcode(src, stats[block_i]);
      data_offset +=
          compressDispatch(src, data_ptr, stats[block_i], slot, opcode);

      // Update iterators
      src += kBlockSize;
      header_ptr += sizeof(Slot);
    }

    u64 header_size = header_ptr - dest;
    u64 payload_size = data_offset - header_size;
    return {header_size, payload_size};
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src) override {
    decompress(dest, size, src, 0);
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src,
                  const u32 block_offset) {
    const u32 block_count = size / kBlockSize;

    // Layout: HEADER [kBlockSize] | COMPRESSED DATA
    const u8 *header_ptr = src + block_offset * sizeof(Slot);
    const u8 *data_ptr = src;

    for (u32 block_i = 0; block_i < block_count; ++block_i) {

      // Read header
      auto &slot = *reinterpret_cast<const Slot *>(header_ptr);
      data_ptr = src + slot.offset;

      // Decompress payload
      Opcode opcode = decodeOpcode(slot.opcode);
      decompressDispatch(dest, data_ptr, slot, opcode);

      // Update iterators
      dest += kBlockSize;
      header_ptr += sizeof(Slot);
    }
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return true; }

private:
  //---------------------------------------------------------------------------
  // COMPRESSION
  //---------------------------------------------------------------------------
  Opcode chooseOpcode(const DataType *src, const Statistics<DataType> &stats) {
    if (stats.max == stats.min) {
      // When min and max are the same, we can use OneValue encoding.
      return Opcode::ONE_VALUE;
    } else if (stats.step_size > 0 && stats.step_size < 16) {
      // When the step size is smaller than 16, we can use MonoInc encoding.
      // We have 4 bit to represent the step size, thus smaller than 16.
      return Opcode::MONOTONICALLY_INCREASING;
    }

    // Choose the compression scheme that yields the smallest size
    auto dest = std::make_unique<u8[]>(kBlockSize * sizeof(DataType) * 2);

    const u32 bp_size =
        std::ceil(static_cast<double>(stats.diff_bits * kBlockSize) / 8);
    const u32 rle_size =
        compressRLE(src, dest.get(), stats.min, stats.diff_bits);

    if (stats.delta) {
      const u32 delta_size = compressDelta(src, dest.get(), stats.min);
      if (delta_size < std::min(rle_size, bp_size)) {
        return Opcode::DELTA;
      }
    }

    return (bp_size <= rle_size) ? Opcode::BIT_PACKING : Opcode::RLE;
  }
  //---------------------------------------------------------------------------
  u16 compressDispatch(const DataType *src, u8 *dest,
                       const Statistics<DataType> &stats, Slot &slot,
                       Opcode opcode) {
    switch (opcode) {
    case Opcode::ONE_VALUE:
      slot.opcode = static_cast<u8>(opcode);
      return 0;
    case Opcode::BIT_PACKING:
      // Use the opcode byte to store the pack size for bitpacking.
      slot.opcode = stats.diff_bits;
      return compressImpl(src, dest, slot);
    case Opcode::RLE:
      slot.opcode = static_cast<u8>(opcode);
      return compressRLE(src, dest, stats.min, stats.diff_bits);
    case Opcode::DELTA:
      slot.opcode = static_cast<u8>(opcode);
      return compressDelta(src, dest, slot.reference);
    case Opcode::MONOTONICALLY_INCREASING:
      slot.opcode = static_cast<u8>(opcode) | (stats.step_size << 2);
      return 0;
    default:
      throw std::runtime_error(
          "Compression failed: Provided opcode does not exist.");
    }
  }
  //---------------------------------------------------------------------------
  u32 compressImpl(const DataType *src, u8 *dest, const Slot &slot) {
    // Normalize
    vector<DataType> normalized(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      normalized[i] = src[i] - slot.reference;
    }

    // Compress
    u8 pack_size = slot.opcode;
    return bitpacking::pack<DataType, kBlockSize>(normalized.data(), dest,
                                                  pack_size);
  }
  //---------------------------------------------------------------------------
  u32 compressRLE(const DataType *src, u8 *dest, const DataType reference,
                  const u8 pack_size) {
    /// Layout:
    /// Data         | Size (Bytes)
    /// ---------------------------
    /// run_count    | 2
    /// lengths[]    | 1 * size
    ///
    /// pack_size    | 1
    /// u32_count    | 1           -- only required for adaptive bitpacking
    /// .... padding to 4 Byte ... -- only required for adaptive bitpacking
    /// values[]     | x
    ///
    //---------------------------------------------------------------------------
    using LengthType = u8;
    //---------------------------------------------------------------------------
    u16 run_count;
    vector<DataType> values;
    vector<LengthType> lengths;
    LengthType max_length = 15;
    //---------------------------------------------------------------------------
    // Encode data
    DataType cur = src[0];
    LengthType run_length = 1;
    //---------------------------------------------------------------------------
    // Helpers for bit manipulation
    u8 shift = 0;
    u8 two_lengths = 0;
    auto appendLength = [&]() {
      two_lengths |= (run_length << (4 - shift));
      if (shift == 4) {
        lengths.push_back(two_lengths);
        two_lengths = 0;
      }
      shift = (shift + 4) % 8;
    };
    //---------------------------------------------------------------------------
    for (u32 i = 1; i < kBlockSize; ++i) {
      if (src[i] == cur) {
        if (run_length == max_length) {
          appendLength();
          values.push_back(cur - reference);
          run_length = 0;
        }
        ++run_length;
      } else {
        appendLength();
        values.push_back(cur - reference);
        run_length = 1;
        cur = src[i];
      }
    }
    appendLength();
    if (shift != 0)
      lengths.push_back(two_lengths);
    values.push_back(cur - reference);
    run_count = values.size();
    assert((values.size() + 1) / 2 == lengths.size());
    //---------------------------------------------------------------------------
    u16 value_offset = sizeof(run_count) + lengths.size() * sizeof(LengthType);
    //---------------------------------------------------------------------------
    // Write lengths
    auto write_ptr = dest;
    std::memcpy(write_ptr, &run_count, sizeof(run_count));
    std::memcpy(write_ptr + sizeof(run_count), lengths.data(),
                lengths.size() * sizeof(LengthType));
    write_ptr += value_offset;
    //---------------------------------------------------------------------------
    // Write pack size
    *write_ptr = pack_size;
    write_ptr += sizeof(pack_size);
    //---------------------------------------------------------------------------
    // Write u32_count
    // u8 &u32_count = *write_ptr;
    // u32_count = 0;
    // write_ptr += sizeof(u32_count);
    //---------------------------------------------------------------------------
    // Compress & write values
    // Pad write_ptr, as values need to be 4-Byte aligned for BitPacking
    // u64 padding = 0;
    // write_ptr = reinterpret_cast<u8 *>(utils::align<u8>(write_ptr, 4,
    // padding));
    u32 values_size = bitpacking::pack<DataType>(values.data(), values.size(),
                                                 write_ptr, pack_size);
    // Include meta data
    // values_size += padding;
    // values_size += sizeof(u32_count);
    values_size += sizeof(pack_size);
    //---------------------------------------------------------------------------
    return value_offset + values_size;
  }
  //---------------------------------------------------------------------------
  u32 compressDelta(const DataType *src, u8 *dest, const DataType reference) {
    // FOR
    auto buffer = std::make_unique<DataType[]>(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      buffer[i] = src[i] - reference;
    }
    //---------------------------------------------------------------------------
    // Delta
    simd::delta::compress<DataType, kBlockSize>(buffer.get());
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
    *dest++ = pack_size;
    //---------------------------------------------------------------------------
    // Compress
    u32 compressed_size =
        bitpacking::pack<DataType, kBlockSize>(buffer.get(), dest, pack_size);
    //---------------------------------------------------------------------------
    return sizeof(pack_size) + compressed_size;
  }

  //---------------------------------------------------------------------------
  // DECOMPRESSION
  //---------------------------------------------------------------------------
  Opcode decodeOpcode(const u8 opcode) {
    // Handle special cases
    if (opcode >= 67)
      return Opcode::MONOTONICALLY_INCREASING;
    if (opcode > 0 && opcode < 65)
      return Opcode::BIT_PACKING;

    return static_cast<Opcode>(opcode);
  }
  //---------------------------------------------------------------------------
  void decompressDispatch(DataType *dest, const u8 *src, const Slot &slot,
                          Opcode opcode) {
    switch (opcode) {
    case Opcode::ONE_VALUE:
      decompressBroadcast(dest, slot.reference);
      break;
    case Opcode::BIT_PACKING:
      decompressImpl(dest, src, slot);
      break;
    case Opcode::RLE:
      decompressRLE(dest, src, slot);
      break;
    case Opcode::DELTA:
      decompressDelta(dest, src, slot);
      break;
    case Opcode::MONOTONICALLY_INCREASING:
      decompressMonoInc(dest, slot);
      break;
    default:
      throw std::runtime_error(
          "Decompression failed: Provided opcode does not exist.");
    }
  }
  //---------------------------------------------------------------------------
  void decompressBroadcast(DataType *dest, DataType value) {
    for (u16 i = 0; i < kBlockSize; ++i) {
      dest[i] = value;
    }
  }
  //---------------------------------------------------------------------------
  void decompressImpl(DataType *dest, const u8 *src, const Slot &slot) {
    // Decompress
    bitpacking::unpack<DataType, kBlockSize>(dest, src, slot.opcode);

    // Denormalize
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  }
  //---------------------------------------------------------------------------
  void decompressRLE(DataType *dest, const u8 *src, const Slot &slot) {
    //---------------------------------------------------------------------------
    /// Layout:
    /// Data         | Size (Bytes)
    /// ---------------------------
    /// run_count    | 2
    /// lengths[]    | 1 * size
    ///
    /// pack_size    | 1
    /// u32_count    | 1            -- only required for adaptive bitpacking
    /// .... padding to 4 Byte ...  -- only required for adaptive bitpacking
    /// values[]     | x
    ///
    //---------------------------------------------------------------------------
    using LengthType = u8;
    //---------------------------------------------------------------------------
    // Decode meta data
    auto run_count = utils::unalignedLoad<u16>(src);
    u16 length_bytes = ((run_count + 1) / 2) * sizeof(LengthType);
    u16 value_offset = sizeof(run_count) + length_bytes;
    auto pack_size = *reinterpret_cast<const u8 *>(src + value_offset);
    // auto u32_count =
    //     *reinterpret_cast<const u8 *>(src + value_offset +
    //     sizeof(pack_size));
    //---------------------------------------------------------------------------
    // Decompress lengths & values
    auto read_ptr = src + value_offset + sizeof(pack_size);
    // Pad read_ptr, as values are 4-Byte aligned for BitPacking
    // u64 padding = 0;
    // read_ptr = reinterpret_cast<const u8 *>(
    //     utils::align<const u8>(read_ptr, 4, padding));
    // Unpack
    vector<DataType> values(run_count * 2);
    bitpacking::unpack<DataType>(values.data(), run_count, read_ptr, pack_size);
    // Initialize lengths
    // TODO: Can be improved probably, currently done to prevent UB
    vector<LengthType> lengths(length_bytes);
    std::memcpy(lengths.data(), src + sizeof(run_count), length_bytes);
    //---------------------------------------------------------------------------
    // Decode RLE using AVX2
    auto write_ptr = dest;
    u16 shift = 0;
    for (u16 i = 0; i < run_count; ++i) {
      LengthType length = (lengths[i / 2] >> (4 - (shift % 8))) & 0x0F;
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
    //---------------------------------------------------------------------------
  }
  //---------------------------------------------------------------------------
  void decompressDelta(DataType *dest, const u8 *src, const Slot &slot) {
    // Bitpacking
    u8 pack_size = *src++;
    bitpacking::unpack<DataType, kBlockSize>(dest, src, pack_size);
    //---------------------------------------------------------------------------
    // Delta
    simd::delta::decompress<DataType, kBlockSize>(dest);
    //---------------------------------------------------------------------------
    // FOR
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  }
  //---------------------------------------------------------------------------
  void decompressMonoInc(DataType *dest, const Slot &slot) {
    // Reveal all but the 0- and 1-bit and shift to origin -> reveals 4 bit
    // step size.
    u8 step_size = (slot.opcode & (0x40 - 4)) >> 2;

    for (u16 i = 0; i < kBlockSize; ++i) {
      dest[i] = slot.reference + i * step_size;
    }
  }
  //---------------------------------------------------------------------------
};
//---------------------------------------------------------------------------
} // namespace compression