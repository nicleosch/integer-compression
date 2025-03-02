#pragma once
//---------------------------------------------------------------------------
#include <cassert>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
u8 getPackSize(INTEGER value) {
  if(value < 0) return 32;  // TODO: Improve handling negative integers
  if(value == 0) value += 1;
  return static_cast<u8>(sizeof(INTEGER) * 8) - __builtin_clz(static_cast<u32>(value));
}
//---------------------------------------------------------------------------
template <typename T>
struct BitPackingLayout {
  u8 pack_size;  // TODO: Currently unused
  T data[];
};
//---------------------------------------------------------------------------
template <typename T>
void pack(const INTEGER* src, u8* dest, const u32 size, INTEGER diff) {
  assert(diff <= std::numeric_limits<T>::max());

  auto& layout = *reinterpret_cast<BitPackingLayout<T>*>(dest);
  layout.pack_size = getPackSize(diff);

  for(u32 i = 0; i < size; ++i) {
    layout.data[i] = static_cast<T>(src[i]);
  }
};
//---------------------------------------------------------------------------
template <typename T>
void unpack(INTEGER* dest, const u8* src, const u32 size) {
  const auto& layout = *reinterpret_cast<const BitPackingLayout<T>*>(src);

  for(u32 i = 0; i < size; ++i) {
    dest[i] = static_cast<INTEGER>(layout.data[i]);
  }
};
//---------------------------------------------------------------------------
}  //namespace compression