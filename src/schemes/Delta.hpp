#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct DeltaLayout {
  INTEGER base;
  u8 pack_size;
  u8 data[];
};
//---------------------------------------------------------------------------
class Delta : public CompressionScheme {
public:
  //---------------------------------------------------------------------------
  u32 compress(const INTEGER *src, const u32 size, u8 *dest,
               const Statistics *stats) override;
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 size, const u8 *src) override;
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override;

private:
  //---------------------------------------------------------------------------
  u8 compressDispatch(const INTEGER *src, const u32 size, u8 *dest,
                      const Statistics *stats);
  //---------------------------------------------------------------------------
  void decompressDispatch(INTEGER *dest, const u32 size,
                          const DeltaLayout &layout);
  //---------------------------------------------------------------------------
  template <typename T>
  void compressImpl(const INTEGER *src, const u32 size, u8 *dest,
                    const INTEGER base) {
    auto data = reinterpret_cast<T *>(dest);

    data[0] = 0;
    for (u32 i = 1; i < size; ++i) {
      data[i] = static_cast<T>(src[i] - src[i - 1]);
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void decompressImpl(INTEGER *dest, const u32 size, const u8 *src,
                      const INTEGER base) {
    dest[0] = base;

    const auto &data = reinterpret_cast<const T *>(src);
    for (u32 i = 1; i < size; ++i) {
      dest[i] = data[i] + dest[i - 1];
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression