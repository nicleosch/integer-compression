#pragma once
//---------------------------------------------------------------------------
#include "common/Types.hpp"
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
/// The slot stored in the header per block.
template <typename T> struct alignas(T) Slot {
  /// The reference, i.e. min, of the corresponding frame.
  T reference;
  /// The maximum value within the frame.
  T max;
  /// The offset into the data array (in sizeof(T)-Byte steps).
  u16 offset;
  /// The number of bits used to store an integer in corresponding frame.
  Opcode opcode;
};
//---------------------------------------------------------------------------
} // namespace tinyblocks
//---------------------------------------------------------------------------
} // namespace compression