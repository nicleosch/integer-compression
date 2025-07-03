#pragma once
//---------------------------------------------------------------------------
#include <immintrin.h>
//---------------------------------------------------------------------------
#include "algebra/Predicate.hpp"
#include "extern/fastpfor/BitPacking.hpp"
#include "statistics/Statistics.hpp"
#include "tinyblocks/Metadata.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace tinyblocks {
//---------------------------------------------------------------------------
namespace rle {
//---------------------------------------------------------------------------
using RunLengthT = u8;
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize, u8 kLengthBits>
u32 compress(const T *src, u8 *dest, const Statistics<T> &stats,
             Slot<T> &slot) {
  static_assert(kLengthBits == 4 || kLengthBits == 8,
                "Run-Lengths must be encoded in either 4 or 8 bits.");
  assert(stats.diff_bits >= 0 && stats.diff_bits <= sizeof(T) * 8);
  //---------------------------------------------------------------------------
  slot.opcode = {kLengthBits == 4 ? Scheme::RLE4 : Scheme::RLE8,
                 stats.diff_bits};
  //---------------------------------------------------------------------------
  constexpr RunLengthT max_length = (kLengthBits == 4) ? 15 : 255;
  //---------------------------------------------------------------------------
  // Data to be stored
  u16 run_count;
  vector<T> values;
  vector<RunLengthT> lengths;
  //---------------------------------------------------------------------------
  // Iterators
  T cur = src[0];
  RunLengthT run_length = 1;
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
  std::memcpy(write_ptr, lengths.data(), lengths.size() * sizeof(RunLengthT));
  //---------------------------------------------------------------------------
  // Write values
  u16 value_offset = lengths.size() * sizeof(RunLengthT);
  write_ptr += value_offset;
  u32 values_size = external::fastpfor::bitpacking::packAdaptive(
      values, reinterpret_cast<u32 *>(write_ptr), slot.opcode.payload);
  //---------------------------------------------------------------------------
  return sizeof(run_count) + value_offset + values_size;
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize, u8 kLengthBits>
void decompress(T *dest, const u8 *src, const Slot<T> &slot) {
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
  vector<T> values(run_count);
  external::fastpfor::bitpacking::unpackAdaptive<T>(
      values, reinterpret_cast<const u32 *>(read_ptr), slot.opcode.payload);
  // Initialize lengths
  // TODO: Can be improved probably, currently done to prevent UB
  vector<RunLengthT> lengths(length_bytes);
  std::memcpy(lengths.data(), src + sizeof(run_count), length_bytes);
  //---------------------------------------------------------------------------
  // Decode RLE using AVX2
  auto write_ptr = dest;
  u16 shift = 0;
  for (u16 i = 0; i < run_count; ++i) {
    RunLengthT length;
    if constexpr (kLengthBits == 4) {
      length = (lengths[i / 2] >> (4 - (shift % 8))) & 0x0F;
    } else {
      length = lengths[i];
    }
    auto end_ptr = write_ptr + length;
    T value = values[i] + slot.reference;

    if constexpr (std::is_same_v<T, INTEGER>) {
      __m256i broadcast = _mm256_set1_epi32(value);
      while (write_ptr + 8 <= end_ptr) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(write_ptr), broadcast);
        write_ptr += 8;
      }
    } else if constexpr (std::is_same_v<T, BIGINT>) {
      __m256i broadcast = _mm256_set1_epi64x(value);
      while (write_ptr + 4 <= end_ptr) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(write_ptr), broadcast);
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
template <typename T, const u16 kBlockSize, u8 kLengthBits>
void filter(const T *data, algebra::Predicate<T> &predicate, Match *matches) {
  // TODO: Implement filter
  throw std::runtime_error("Not implemented yet.");
}
//---------------------------------------------------------------------------
} // namespace rle
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression