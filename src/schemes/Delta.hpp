#pragma once
//---------------------------------------------------------------------------
#include "common/Types.hpp"
#include "statistics/Statistics.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct DeltaLayout {
  u8 pack_size;
  u8 data[];
};
//---------------------------------------------------------------------------
class Delta {
public:
  //---------------------------------------------------------------------------
  u32 compress(const INTEGER *src, const u32 total_size, u8 *dest,
               const Statistics *stats);
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 total_size, const u8 *src);

private:
  //---------------------------------------------------------------------------
  u8 compressDispatch(const INTEGER *src, const u32 size, u8 *dest,
                      const Statistics *stats);
  //---------------------------------------------------------------------------
  void decompressDispatch(INTEGER *dest, const u32 size, const u8 *src,
                          const u8 pack_size);
  //---------------------------------------------------------------------------
  template <typename T>
  void compressImpl(const INTEGER *src, const u32 size, u8 *dest,
                    const Statistics *stats) {
    auto data = reinterpret_cast<T *>(dest);

    data[0] = static_cast<T>(src[0]);
    for (u32 i = 1; i < size; ++i) {
      data[i] = static_cast<T>(src[i] - src[i - 1]);
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void decompressImpl(INTEGER *dest, const u32 size, const u8 *src) {
    const auto &data = reinterpret_cast<const T *>(src);

    dest[0] = data[0];
    for (u32 i = 1; i < size; ++i) {
      dest[i] = data[i] + dest[i - 1];
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression