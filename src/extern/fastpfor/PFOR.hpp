#pragma once
//---------------------------------------------------------------------------
#include "bitpacking/BitPacking.hpp"
#include "common/Types.hpp"
#include "common/Utils.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
/// Functions in this namespace implement FastPFOR functionality as pioneered by
/// Daniel Lemire and Leonid Boytsov (https://github.com/fast-pack/FastPFOR).
namespace pfor {
//---------------------------------------------------------------------------
using SIMDRegister = __m128i;
//---------------------------------------------------------------------------
/// This namespace contains internal functions used within the "pfor" namespace.
namespace internal {
//---------------------------------------------------------------------------
/// @brief Determines the cheapest distribution of regular- and exception-values
/// based on given cost-function.
///
/// @tparam T: They type of the values to be compressed.
/// @tparam kBlockSize: The size of a block of values.
///
/// @param src The integers to be compressed.
/// @param best_pack_size The number of bits required to pack the regular
/// values, excluding the exceptions.
/// @param max_pack_size The number of bits required to pack all values,
/// including the exceptions.
/// @param costFn The cost function based on which the exception values are
/// determined.
///
/// Note: This function is derived from "getBestBFromData" from the
/// simdfastpfor.h header in https://github.com/fast-pack/FastPFOR. To be
/// precise, symbol names are changed and the cost function adjusted, as well as
/// some more specific edge cases considered.
template <typename T, const u16 kBlockSize>
void getBestPackSize(const T *src, u8 &best_pack_size, u8 &max_pack_size,
                     auto &&costFn) {
  const u8 maxb = sizeof(T) * 8;
  //---------------------------------------------------------------------------
  u32 bit_freqs[maxb + 1];
  bool max_values[maxb + 1];
  for (u32 k = 0; k <= maxb; ++k) {
    bit_freqs[k] = 0;
    max_values[k] = false;
  }
  for (u32 k = 0; k < kBlockSize; ++k) {
    u8 b = utils::requiredBits<T>(src[k]);
    bit_freqs[b]++;
    T max_value = (1 << b) - 1;
    if (src[k] == max_value)
      max_values[b] = true;
  }
  //---------------------------------------------------------------------------
  best_pack_size = maxb;
  while (bit_freqs[best_pack_size] == 0)
    best_pack_size--;
  // We use the max value as exception value so it mustn't appear in the data.
  while (best_pack_size <= maxb && max_values[best_pack_size])
    best_pack_size++;
  max_pack_size = best_pack_size;
  //---------------------------------------------------------------------------
  u32 bestcost = best_pack_size * kBlockSize / 8;
  u32 cexcept = 0;
  for (u32 b = best_pack_size - 1; b > 0; --b) {
    cexcept += bit_freqs[b + 1];
    u32 thiscost = costFn(cexcept, b, (max_pack_size - b));
    // Update Pack-Size if cost smaller AND exception value not present.
    if (thiscost < bestcost && !max_values[b]) {
      bestcost = thiscost;
      best_pack_size = static_cast<uint8_t>(b);
    }
  }
};
//---------------------------------------------------------------------------
/// @brief Compresses given data using PFOR.
///
/// @tparam T: They type of the values to be compressed.
/// @tparam kBlockSize: The size of a block of values.
///
/// @param src The integers to be compressed.
/// @param dest The destination to compress the data to.
/// @param reference The reference value used for FOR compression.
/// @param best_pack_size The number of bits required to pack the regular
/// values, excluding the exceptions.
/// @param max_pack_size The number of bits required to pack all values,
/// including the exceptions.
/// @param exceptions The exception values.
/// @param costFn The cost function based on which the exception values are
/// determined.
///
/// Note: This function is derived from "__encodeArray" from the
/// simdfastpfor.h header in https://github.com/fast-pack/FastPFOR.
template <typename T, const u16 kBlockSize>
u32 compress(const T *src, u8 *dest, const T &reference, u8 &best_pack_size,
             u8 &max_pack_size, vector<T> &exceptions, auto &&costFn) {
  // Normalize
  vector<T> buffer(kBlockSize);
  for (u32 i = 0; i < kBlockSize; ++i) {
    buffer[i] = src[i] - reference;
  }
  //---------------------------------------------------------------------------
  // Determine pack sizes.
  getBestPackSize<T, kBlockSize>(buffer.data(), best_pack_size, max_pack_size,
                                 costFn);
  // No compression possible.
  if (best_pack_size == sizeof(T) * 8) {
    std::memcpy(dest, src, sizeof(T) * kBlockSize);
    return sizeof(T) * kBlockSize;
  }
  //---------------------------------------------------------------------------
  // Write data & exceptions.
  T max_value = (1U << best_pack_size) - 1;
  for (u16 i = 0; i < kBlockSize; ++i) {
    assert(buffer[i] != max_value);
    if (buffer[i] > max_value) {
      exceptions.push_back(buffer[i]);
      buffer[i] = max_value;
    }
  }
  //---------------------------------------------------------------------------
  // Bitpack payload.
  return bitpacking::pack<T, kBlockSize>(buffer.data(), dest, best_pack_size);
};
//---------------------------------------------------------------------------
/// @brief Decompresses given data using PFOR.
///
/// @tparam T: They type of the values to be decompressed.
/// @tparam ExceptionT: The type of the exceptions to be decompressed.
/// @tparam kBlockSize: The size of a block of values.
///
/// @param dest The decompressed integers.
/// @param src The compressed data.
/// @param reference The reference value used for FOR decompression.
/// @param pack_size The number of bits required to unpack the regular
/// values, excluding the exceptions.
/// @param exceptions The exception values.
///
/// Note: This function is derived from "__decodeArray" from the simdfastpfor.h
/// header in https://github.com/fast-pack/FastPFOR.
template <typename T, typename ExceptionT, const u16 kBlockSize>
void decompress(T *dest, const u8 *src, const T &reference, const u8 &pack_size,
                const ExceptionT *exceptions) {
  assert(sizeof(T) >= sizeof(ExceptionT));
  //---------------------------------------------------------------------------
  // Unpack the payload.
  bitpacking::unpack<T, kBlockSize>(dest, src, pack_size);
  //---------------------------------------------------------------------------
  // Decompress the payload.
  T max_value = (1U << pack_size) - 1;
  u16 cexcept = 0;
  for (u32 i = 0; i < kBlockSize; ++i) {
    if (dest[i] == max_value) {
      dest[i] = utils::unalignedLoad<ExceptionT>(&exceptions[cexcept++]);
    }
    dest[i] += reference;
  }
}
//---------------------------------------------------------------------------
/// @brief Determines the offset of the exceptions values based on given
/// pack-size. Since we use SIMD BitPacking, the offset is aligned to the size
/// of the smallest SIMD register.
template <const u16 kBlockSize> u32 getExceptionOffset(const u8 pack_size) {
  return (kBlockSize * pack_size / 8 + (sizeof(SIMDRegister) - 1)) &
         ~(sizeof(SIMDRegister) - 1);
}
//---------------------------------------------------------------------------
/// @brief Write "size" values of type "SourceType" to "TargetType".
template <typename SourceType, typename TargetType>
u32 intpack(const SourceType *src, const u32 size, u8 *dest) {
  for (u32 i = 0; i < size; ++i) {
    utils::unalignedStore<TargetType>(dest + i * sizeof(TargetType),
                                      static_cast<TargetType>(src[i]));
  }
  return sizeof(TargetType) * size;
}
//---------------------------------------------------------------------------
} // namespace internal

/// This namespace contains reimplementations of Lemire's PFOR that usually only
/// differ from the original regarding symbol naming.
namespace lemire {
//---------------------------------------------------------------------------
/// Per exception, we store an exception's location in the block.
/// Note: Currently naively set to 16 bit, although 8 bits is enough for smaller
/// block sizes.
using ExceptionLocation = u16;
const u8 kExceptionOverhead = sizeof(ExceptionLocation) * 8;
//---------------------------------------------------------------------------
/// @brief Determines the cheapest distribution of regular- and
/// exception-values.
///
/// @tparam T: They type of the values to be compressed.
/// @tparam kBlockSize: The size of a block of values.
///
/// @param src The integers to be compressed.
/// @param best_pack_size The number of bits required to pack the regular
/// values, excluding the exceptions.
/// @param max_pack_size The number of bits required to pack all values,
/// including the exceptions.
///
/// Note: This function is derived from "getBestBFromData" from the
/// simdfastpfor.h header in https://github.com/fast-pack/FastPFOR.
/// Only symbol names are changed.
template <typename T, const u16 kBlockSize>
void getBestPackSize(const T *src, u8 &best_pack_size, u8 &max_pack_size) {
  const u8 maxb = sizeof(T) * 8;
  //---------------------------------------------------------------------------
  u32 bit_freqs[maxb + 1];
  for (u32 k = 0; k <= maxb; ++k)
    bit_freqs[k] = 0;
  for (u32 k = 0; k < kBlockSize; ++k)
    bit_freqs[utils::requiredBits<T>(src[k])]++;
  //---------------------------------------------------------------------------
  best_pack_size = maxb;
  while (bit_freqs[best_pack_size] == 0)
    best_pack_size--;
  max_pack_size = best_pack_size;
  //---------------------------------------------------------------------------
  u32 bestcost = best_pack_size * kBlockSize;
  u32 cexcept = 0;
  for (u32 b = best_pack_size - 1; b > 0; --b) {
    cexcept += bit_freqs[b + 1];
    u32 thiscost =
        cexcept * kExceptionOverhead // Constant Exception Overhead
        + ((cexcept * (max_pack_size - b) + (sizeof(SIMDRegister) * 8 - 1)) &
           ~(sizeof(SIMDRegister) * 8 - 1)) // Packed Exceptions
        + b * kBlockSize;                   // Packed Payload
    if (max_pack_size - b == 1)
      thiscost -= cexcept;
    if (thiscost < bestcost) {
      bestcost = thiscost;
      best_pack_size = static_cast<uint8_t>(b);
    }
  }
};
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
u32 compress(const T *src, u8 *dest, u8 &best_pack_size) {
  // Determine pack sizes.
  u8 max_pack_size;
  getBestPackSize<T, kBlockSize>(src, best_pack_size, max_pack_size);
  //---------------------------------------------------------------------------
  // Build buffers.
  vector<T> exceptions;
  vector<ExceptionLocation> locations;
  T max_value = (1U << best_pack_size) - 1;
  for (u16 i = 0; i < kBlockSize; ++i) {
    if (src[i] >= max_value) {
      exceptions.push_back(src[i] >> best_pack_size);
      locations.push_back(i);
    }
  }
  //---------------------------------------------------------------------------
  // Serialize the data.
  auto write_ptr = dest;
  // Payload.
  write_ptr += bitpacking::pack<T, kBlockSize>(src, write_ptr, best_pack_size);
  // Meta data.
  assert(locations.size() < (1ULL << 24));
  u32 meta = ((static_cast<u32>(max_pack_size) << 24) |
              static_cast<u32>(locations.size()));
  *reinterpret_cast<u32 *>(write_ptr) = meta;
  write_ptr += sizeof(meta);
  // Locations.
  auto locations_size = locations.size() * sizeof(ExceptionLocation);
  std::memcpy(write_ptr, locations.data(), locations_size);
  write_ptr += locations_size;
  // Exceptions.
  u64 padding = 0;
  utils::align<u8>(write_ptr, 4, padding);
  write_ptr += bitpacking::pack<T>(exceptions.data(), exceptions.size(),
                                   write_ptr, max_pack_size - best_pack_size);
  //---------------------------------------------------------------------------
  return write_ptr - dest;
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize>
void decompress(T *dest, const u8 *src, const u8 &pack_size) {
  // Unpack the payload.
  bitpacking::unpack<T, kBlockSize>(dest, src, pack_size);
  //---------------------------------------------------------------------------
  auto read_ptr = reinterpret_cast<const u8 *>(
      src + internal::getExceptionOffset<kBlockSize>(pack_size));
  //---------------------------------------------------------------------------
  // Deserialize meta data.
  auto meta = *reinterpret_cast<const u32 *>(read_ptr);
  u8 max_pack_size = static_cast<u8>(meta >> 24);
  u32 exceptions_size = meta & 0xFFFFFF;
  read_ptr += sizeof(meta);
  //---------------------------------------------------------------------------
  // Deserialize locations.
  auto locations = reinterpret_cast<const ExceptionLocation *>(read_ptr);
  read_ptr += exceptions_size * sizeof(ExceptionLocation);
  //---------------------------------------------------------------------------
  // Unpack exceptions.
  vector<T> exceptions(exceptions_size);
  u64 padding = 0;
  utils::align<const u8>(read_ptr, 4, padding);
  bitpacking::unpack<T>(exceptions.data(), exceptions_size, read_ptr,
                        max_pack_size - pack_size);
  //---------------------------------------------------------------------------
  // Decompress the payload.
  for (u16 i = 0; i < exceptions_size; ++i) {
    dest[locations[i]] |= (exceptions[i] << pack_size);
  }
}
//---------------------------------------------------------------------------
} // namespace lemire

//---------------------------------------------------------------------------
/// @brief This function compresses given data using PFOR and leaves the
/// exception-values uncompressed.
template <typename T, const u16 kBlockSize>
u32 compressPFOR(const T *src, u8 *dest, const T &reference, u8 &pack_size) {
  // Compress payload using PFOR.
  u8 max_pack_size;
  vector<T> exceptions;
  u32 payload_size = internal::compress<T, kBlockSize>(
      src, dest, reference, pack_size, max_pack_size, exceptions,
      [&](const u32 cexcept, const u32 b, const u32) {
        return b * kBlockSize / 8 + // Payload
               cexcept * sizeof(T); // Exceptions
      });
  //---------------------------------------------------------------------------
  // Leave the exceptions uncompressed.
  if (exceptions.size() > 0)
    std::memcpy(dest + payload_size, exceptions.data(),
                exceptions.size() * sizeof(T));
  //---------------------------------------------------------------------------
  return payload_size + exceptions.size() * sizeof(T);
};
//---------------------------------------------------------------------------
/// @brief This function compresses given data using PFOR and bitpacks the
/// exception-values.
template <typename T, const u16 kBlockSize>
u32 compressPFOREBP(const T *src, u8 *dest, const T &reference, u8 &pack_size) {
  auto write_ptr = dest;
  //---------------------------------------------------------------------------
  // Compress payload using PFOR.
  u8 exc_pack_size;
  vector<T> exceptions;
  write_ptr += internal::compress<T, kBlockSize>(
      src, dest, reference, pack_size, exc_pack_size, exceptions,
      [&](const u32 cexcept, const u32 b, const u32 eb) {
        return b * kBlockSize / 8 + // Payload
                   sizeof(u32) +    // Exceptions-Length
                   (cexcept * eb / 8 + (sizeof(SIMDRegister) - 1)) &
               ~(sizeof(SIMDRegister) - 1); // Exceptions
      });
  //---------------------------------------------------------------------------
  // Write meta data.
  auto exceptions_length = static_cast<u32>(exceptions.size());
  assert(exceptions_length < (1U << 24));
  // | PS | EL | EL | EL |
  exceptions_length |= (static_cast<u32>(exc_pack_size) << 24);
  std::memcpy(write_ptr, &exceptions_length, sizeof(exceptions_length));
  write_ptr += sizeof(exceptions_length);
  //---------------------------------------------------------------------------
  // Pack the exceptions, if there are any.
  if (exceptions_length > 0) {
    write_ptr += bitpacking::pack<T>(exceptions.data(), exceptions.size(),
                                     write_ptr, exc_pack_size);
  }
  //---------------------------------------------------------------------------
  return write_ptr - dest;
};
//---------------------------------------------------------------------------
/// @brief This function compresses given data using PFOR and packs the
/// exception-values into the minimum possible integer type (1,2,4,8 bytes).
template <typename T, const u16 kBlockSize>
u32 compressPFOREP(const T *src, u8 *dest, const T &reference, u8 &pack_size) {
  // Compress payload using PFOR.
  u8 max_pack_size;
  vector<T> exceptions;
  u32 payload_size = internal::compress<T, kBlockSize>(
      src, dest, reference, pack_size, max_pack_size, exceptions,
      [&](const u32 cexcept, const u32 b, const u32 eb) {
        u32 req_bytes = sizeof(u64);
        if (eb <= 8)
          req_bytes = sizeof(u32);
        else if (eb <= 16)
          req_bytes = sizeof(u16);
        else if (eb <= 8)
          req_bytes = sizeof(u8);
        return b * kBlockSize / 8 + // Payload
               cexcept * req_bytes; // Exceptions
      });
  auto write_ptr = dest + payload_size;
  //---------------------------------------------------------------------------
  // Write the exceptions, if there are any.
  assert(exceptions.size() < (1U << 24));
  auto cexc = static_cast<u16>(exceptions.size());
  u32 exceptions_size = 0; // The number of bytes of compressed exceptions.
  u8 byte_code = 0;        // The number of required bytes for the exceptions.
  if (max_pack_size <= 8) {
    exceptions_size +=
        internal::intpack<T, u8>(exceptions.data(), cexc, write_ptr);
  } else if (max_pack_size <= 16) {
    exceptions_size +=
        internal::intpack<T, u16>(exceptions.data(), cexc, write_ptr);
    byte_code = 1;
  } else if (max_pack_size <= 32) {
    exceptions_size +=
        internal::intpack<T, u32>(exceptions.data(), cexc, write_ptr);
    byte_code = 2;
  } else {
    assert(max_pack_size <= 64);
    exceptions_size +=
        internal::intpack<T, u64>(exceptions.data(), cexc, write_ptr);
    byte_code = 3;
  }
  //---------------------------------------------------------------------------
  // Write byte-code into the-pack size. (Use the upper two bits)
  if (max_pack_size == 64) [[unlikely]]
    pack_size = 0; // special value
  else
    pack_size |= (byte_code << 6);
  //---------------------------------------------------------------------------
  return payload_size + exceptions_size;
};
//---------------------------------------------------------------------------
/// @brief This function compresses given data using Lemire's original PFOR
/// implementation, packing the payload and packing only the upper exception
/// bits.
template <typename T, const u16 kBlockSize>
u32 compressPFORLemire(const T *src, u8 *dest, u8 &pack_size) {
  return lemire::compress<T, kBlockSize>(src, dest, pack_size);
}

//---------------------------------------------------------------------------
/// @brief Decompress PFOR with uncompressed exception values.
template <typename T, const u16 kBlockSize>
void decompressPFOR(T *dest, const u8 *src, const T &reference,
                    const u8 &pack_size) {
  // Determine exception offset.
  auto exceptions = reinterpret_cast<const T *>(
      src + internal::getExceptionOffset<kBlockSize>(pack_size));
  //---------------------------------------------------------------------------
  // Decompress the payload.
  internal::decompress<T, T, kBlockSize>(dest, src, reference, pack_size,
                                         exceptions);
};
//---------------------------------------------------------------------------
/// @brief Decompress PFOR with bitpacked exception values.
template <typename T, const u16 kBlockSize>
void decompressPFOREBP(T *dest, const u8 *src, const T &reference,
                       const u8 &pack_size) {
  // Determine exception offset.
  auto exception_ptr =
      src + internal::getExceptionOffset<kBlockSize>(pack_size);
  //---------------------------------------------------------------------------
  // Unpack exceptions.
  auto exceptions_meta = utils::unalignedLoad<u32>(exception_ptr);
  exception_ptr += sizeof(exceptions_meta);
  const u32 exceptions_length = exceptions_meta & ((1U << 24) - 1);
  const u8 exc_pack_size = exceptions_meta >> 24;
  // TODO: Move such allocations one level up as it is not necessary to
  // allocate for each block.
  auto exceptions = std::make_unique<T[]>(exceptions_length);
  bitpacking::unpack<T>(exceptions.get(), exceptions_length, exception_ptr,
                        exc_pack_size);
  //---------------------------------------------------------------------------
  // Decompress the payload.
  internal::decompress<T, T, kBlockSize>(dest, src, reference, pack_size,
                                         exceptions.get());
};
//---------------------------------------------------------------------------
/// @brief Decompress PFOR with "integer-packed" exceptions values.
template <typename T, const u16 kBlockSize>
void decompressPFOREP(T *dest, const u8 *src, const T &reference,
                      const u8 &pack_size) {
  // Determine required bytes and bits.
  u8 req_bits = 0;
  u8 byte_code = 0;
  if (pack_size == 0) [[unlikely]] {
    req_bits = 64;
    byte_code = 3;
  } else {
    req_bits = pack_size & 63;
    byte_code = pack_size >> 6;
  }
  //---------------------------------------------------------------------------
  // Determine exception offset.
  auto exceptions = src + internal::getExceptionOffset<kBlockSize>(req_bits);
  //---------------------------------------------------------------------------
  // Decompress the payload.
  switch (byte_code) {
  case 0:
    internal::decompress<T, u8, kBlockSize>(
        dest, src, reference, req_bits,
        reinterpret_cast<const u8 *>(exceptions));
    return;
  case 1:
    internal::decompress<T, u16, kBlockSize>(
        dest, src, reference, req_bits,
        reinterpret_cast<const u16 *>(exceptions));
    return;
  case 2:
    internal::decompress<T, u32, kBlockSize>(
        dest, src, reference, req_bits,
        reinterpret_cast<const u32 *>(exceptions));
    return;
  case 3:
    internal::decompress<T, u64, kBlockSize>(
        dest, src, reference, req_bits,
        reinterpret_cast<const u64 *>(exceptions));
    return;
  default:
    throw std::runtime_error("Invalid byte code: " + byte_code);
  }
};
//---------------------------------------------------------------------------
/// @brief This function decompresses given data using Lemire's original PFOR
/// implementation, unpacking the payload and unpacking only the upper exception
/// bits.
template <typename T, const u16 kBlockSize>
void decompressPFORLemire(T *dest, const u8 *src, const u8 &pack_size) {
  return lemire::decompress<T, kBlockSize>(dest, src, pack_size);
};
//---------------------------------------------------------------------------
} // namespace pfor
//---------------------------------------------------------------------------
} // namespace compression