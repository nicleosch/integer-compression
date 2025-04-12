#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct FORLayout {
  INTEGER reference;
  u8 pack_size;
  u8 data[];
};
//---------------------------------------------------------------------------
class FOR : public CompressionScheme {
public:
  //---------------------------------------------------------------------------
  u32 compress(const INTEGER *src, const u32 size, u8 *dest,
               const Statistics *stats) override;
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 size, const u8 *src) override;
  //---------------------------------------------------------------------------
  void decompress(INTEGER *dest, const u32 size, const u8 *src,
                  const u32 offset);
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override;

private:
  //---------------------------------------------------------------------------
  u8 compressDispatch(const INTEGER *src, const u32 size, u8 *dest,
                      const Statistics *stats);
  //---------------------------------------------------------------------------
  void decompressDispatch(INTEGER *dest, const u32 size, const u8 *src,
                          const u32 reference, const u8 pack_size);
  //---------------------------------------------------------------------------
  template <typename T>
  void compressImpl(const INTEGER *src, const u32 size, u8 *dest,
                    const Statistics *stats) {
    auto data = reinterpret_cast<T *>(dest);
    for (u32 i = 0; i < size; ++i) {
      data[i] = static_cast<T>(src[i] - stats->min);
    }
  }
  //---------------------------------------------------------------------------
  template <typename T>
  void decompressImpl(INTEGER *dest, const u32 size, const u8 *src,
                      const u32 reference) {
    const auto &data = reinterpret_cast<const T *>(src);
    for (u32 i = 0; i < size; ++i) {
      dest[i] = data[i] + reference;
    }
  }
};
//---------------------------------------------------------------------------
} // namespace compression