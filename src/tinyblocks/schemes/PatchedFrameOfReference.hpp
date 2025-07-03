#pragma once
//---------------------------------------------------------------------------
#include "algebra/Predicate.hpp"
#include "extern/fastpfor/PFOR.hpp"
#include "statistics/Statistics.hpp"
#include "tinyblocks/Metadata.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace tinyblocks {
//---------------------------------------------------------------------------
namespace pframeofreference {
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize, Scheme kScheme>
u32 compress(const T *src, u8 *dest, Slot<T> &slot) {
  slot.opcode = {kScheme, 0};
  //---------------------------------------------------------------------------
  if constexpr (kScheme == Scheme::PFOR) {
    return external::fastpfor::compressPFOR<T, kBlockSize>(
        src, dest, slot.reference, slot.opcode.payload);
  } else if constexpr (kScheme == Scheme::PFOR_EBP) {
    return external::fastpfor::compressPFOREBP<T, kBlockSize>(
        src, dest, slot.reference, slot.opcode.payload);
  } else if constexpr (kScheme == Scheme::PFOR_EP) {
    return external::fastpfor::compressPFOREP<T, kBlockSize>(
        src, dest, slot.reference, slot.opcode.payload);
  } else if constexpr (kScheme == Scheme::PFOR_DELTA) {
    // Normalize
    vector<T> normalized(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      normalized[i] = src[i] - slot.reference;
    }
    // Delta
    external::fastpfor::delta::compress<T, kBlockSize>(normalized.data());
    // PFOR
    T min = 0;
    T max = 0;
    for (u16 i = 0; i < kBlockSize; ++i) {
      if (normalized[i] < min)
        min = normalized[i];
      if (normalized[i] > max)
        max = normalized[i];
    }
    return external::fastpfor::compressPFORLemire<T, kBlockSize>(
        normalized.data(), dest, slot.opcode.payload);
  } else {
    static_assert(kScheme == Scheme::PFOR_LEMIRE);
    // Normalize
    vector<T> normalized(kBlockSize);
    for (u32 i = 0; i < kBlockSize; ++i) {
      normalized[i] = src[i] - slot.reference;
    }
    // Compress
    return external::fastpfor::compressPFORLemire<T, kBlockSize>(
        normalized.data(), dest, slot.opcode.payload);
  }
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize, Scheme kScheme>
void decompress(T *dest, const u8 *src, const Slot<T> &slot) {
  if constexpr (kScheme == Scheme::PFOR) {
    external::fastpfor::decompressPFOR<T, kBlockSize>(dest, src, slot.reference,
                                                      slot.opcode.payload);
  } else if constexpr (kScheme == Scheme::PFOR_EBP) {
    external::fastpfor::decompressPFOREBP<T, kBlockSize>(
        dest, src, slot.reference, slot.opcode.payload);
  } else if constexpr (kScheme == Scheme::PFOR_EP) {
    external::fastpfor::decompressPFOREP<T, kBlockSize>(
        dest, src, slot.reference, slot.opcode.payload);
  } else if constexpr (kScheme == Scheme::PFOR_DELTA) {
    // Decompress
    external::fastpfor::decompressPFORLemire<T, kBlockSize>(
        dest, src, slot.opcode.payload);
    // Delta
    external::fastpfor::delta::decompress<T, kBlockSize>(dest);
    // Denormalize
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  } else {
    static_assert(kScheme == Scheme::PFOR_LEMIRE);
    // Decompress
    external::fastpfor::decompressPFORLemire<T, kBlockSize>(
        dest, src, slot.opcode.payload);
    // Denormalize
    for (u32 i = 0; i < kBlockSize; ++i) {
      dest[i] += slot.reference;
    }
  }
}
//---------------------------------------------------------------------------
template <typename T, const u16 kBlockSize, Scheme kScheme>
void filter(const T *data, algebra::Predicate<T> &predicate, Match *matches) {
  // TODO: Implement filter
  throw std::runtime_error("Not implemented yet.");
}
//---------------------------------------------------------------------------
} // namespace pframeofreference
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression