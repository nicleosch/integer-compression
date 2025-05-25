#pragma once
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
#include "common/Types.hpp"
#include "common/Utils.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace pfor {
//---------------------------------------------------------------------------
/// This function is derived from "getBestBFromData" from the simdfastpfor.h
/// header in https://github.com/fast-pack/FastPFOR. To be precise, symbol names
/// are changed and the cost function adjusted, as well as some more specific
/// edge cases considered.
template <typename T, const u16 kBlockSize>
void getBestPackSize(const T *src, u8 &best_pack_size, u8 &max_pack_size) {
  const u32 expression_overhead = sizeof(T);
  //---------------------------------------------------------------------------
  u32 bit_freqs[sizeof(T) * 8 + 1];
  for (u32 k = 0; k <= sizeof(T) * 8; ++k)
    bit_freqs[k] = 0;
  for (u32 k = 0; k < kBlockSize; ++k) {
    bit_freqs[utils::requiredBits<T>(src[k])]++;
  }
  //---------------------------------------------------------------------------
  best_pack_size = sizeof(T) * 8;
  while (bit_freqs[best_pack_size] == 0)
    best_pack_size--;
  max_pack_size = best_pack_size;
  // The second best pack size, required to handle below's edge case.
  u8 sbest_pack_size = best_pack_size + 1;
  //---------------------------------------------------------------------------
  u32 bestcost = best_pack_size * kBlockSize / 8;
  u32 cexcept = 0;
  for (u32 b = best_pack_size - 1; b > 0; --b) {
    cexcept += bit_freqs[b + 1];
    u32 thiscost = cexcept * expression_overhead + b * kBlockSize / 8;
    if (thiscost < bestcost) {
      bestcost = thiscost;
      sbest_pack_size = best_pack_size;
      best_pack_size = static_cast<uint8_t>(b);
    }
  }
  //---------------------------------------------------------------------------
  // Edge case handling:
  // As we reserve the highest value to be represented with k bits as our
  // exception value, we have to increase the pack size if that value appears in
  // the data.
  T max_value = (1 << best_pack_size) - 1;
  for (u32 k = 0; k < kBlockSize; ++k) {
    if (src[k] == max_value) {
      best_pack_size = sbest_pack_size;
      break;
    }
  }
};
//---------------------------------------------------------------------------
/// This function is derived from "__encodeArray" from the simdfastpfor.h
/// header in https://github.com/fast-pack/FastPFOR.
template <typename T, const u16 kBlockSize>
u32 compress(const T *src, u8 *dest, const T &reference, u8 &pack_size) {
  // Normalize
  vector<T> buffer(kBlockSize);
  for (u32 i = 0; i < kBlockSize; ++i) {
    buffer[i] = src[i] - reference;
  }
  //---------------------------------------------------------------------------
  // Determine pack sizes.
  u8 max_pack_size;
  getBestPackSize<T, kBlockSize>(buffer.data(), pack_size, max_pack_size);
  //---------------------------------------------------------------------------
  // Write data & exceptions.
  vector<T> exceptions;
  T max_value = (1U << pack_size) - 1;
  for (u16 i = 0; i < kBlockSize; ++i) {
    assert(buffer[i] != max_value);
    if (buffer[i] > max_value) {
      exceptions.push_back(src[i]);
      buffer[i] = max_value;
    }
  }
  //---------------------------------------------------------------------------
  // Compress & serialize data.
  auto write_ptr = dest;
  u32 payload_size =
      bitpacking::pack<T, kBlockSize>(buffer.data(), write_ptr, pack_size);
  write_ptr += payload_size;
  // auto exc_size = static_cast<u16>(exceptions.size());
  // std::memcpy(write_ptr, &exc_size, sizeof(exc_size));
  // write_ptr += sizeof(exc_size);
  if (exceptions.size() > 0)
    std::memcpy(write_ptr, exceptions.data(), exceptions.size() * sizeof(T));
  //---------------------------------------------------------------------------
  return payload_size + exceptions.size() * sizeof(T);
};
//---------------------------------------------------------------------------
/// This function is derived from "__decodeArray" from the simdfastpfor.h
/// header in https://github.com/fast-pack/FastPFOR.
template <typename T, const u16 kBlockSize>
void decompress(T *dest, const u8 *src, const T &reference,
                const u8 &pack_size) {
  // Decompress
  bitpacking::unpack<T, kBlockSize>(dest, src, pack_size);
  //---------------------------------------------------------------------------
  // Payload padded to the next SIMD register offset.
  u32 register_size = sizeof(__m128i);
  u32 payload_size =
      (kBlockSize * pack_size / 8 + register_size) & ~register_size;
  auto exception_ptr = reinterpret_cast<const T *>(src + payload_size);
  //---------------------------------------------------------------------------
  T max_value = (1U << pack_size) - 1;
  u16 cexcept;
  for (u32 i = 0; i < kBlockSize; ++i) {
    if (dest[i] == max_value) {
      dest[i] = exception_ptr[cexcept++];
    }
    dest[i] += reference;
  }
};
//---------------------------------------------------------------------------
} // namespace pfor
//---------------------------------------------------------------------------
} // namespace compression