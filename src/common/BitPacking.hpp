#pragma once
//---------------------------------------------------------------------------
#include <cassert>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
template <typename T>
struct BitPackingLayout {
  T data[];
};
//---------------------------------------------------------------------------
class BitPacking {
  public:
    /// @brief Bitpack the given data.
    /// @param src The integers to be packed.
    /// @param dest The destination to pack the data to.
    /// @param size The amount of integers to be packed.
    /// @param diff The difference between min and max in the source data. 
    /// @return The amount of bits used to pack the data.
    static u8 pack(const INTEGER* src, u8* dest, const u32 size, INTEGER diff) {
      auto pack_size = packSizeFor(diff);

      if(pack_size <= 8) {
        compress<u8>(src, dest, size, pack_size);
        return 8;
      } else if(pack_size <= 16) {
        compress<u16>(src, dest, size, pack_size);
        return 16;
      }

      return 32;
    }
    //---------------------------------------------------------------------------
    /// @brief Unpack the given data.
    /// @param dest The resulting integers.
    /// @param src The data to unpack.
    /// @param size The amount of integers to be unpacked.
    /// @param pack_size The amount of bits used for packing.
    static void unpack(INTEGER* dest, const u8* src, const u32 size, const u8 pack_size) {
      if(pack_size <= 8) {
        decompress<u8>(dest, src, size);
      } else if(pack_size <= 16) {
        decompress<u16>(dest, src, size);
      }
    }
  private:
    static u8 packSizeFor(INTEGER value) {
      if(value < 0) return 32;  // TODO: Improve handling negative integers
      if(value == 0) value += 1;
      return static_cast<u8>(sizeof(INTEGER) * 8) - __builtin_clz(static_cast<u32>(value));
    }
    //---------------------------------------------------------------------------
    template <typename T>
    static void compress(const INTEGER* src, u8* dest, const u32 size, const u8 pack_size) {
      auto& layout = *reinterpret_cast<BitPackingLayout<T>*>(dest);

      for(u32 i = 0; i < size; ++i) {
        layout.data[i] = static_cast<T>(src[i]);
      }
    }
    //---------------------------------------------------------------------------
    template <typename T>
    static void decompress(INTEGER* dest, const u8* src, const u32 size) {
      const auto& layout = *reinterpret_cast<const BitPackingLayout<T>*>(src);

      for(u32 i = 0; i < size; ++i) {
        dest[i] = static_cast<INTEGER>(layout.data[i]);
      }
    }
};
//---------------------------------------------------------------------------
}  // namespace compression