#pragma once
//---------------------------------------------------------------------------
#include "schemes/CompressionScheme.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
struct RLELayout {
  u32 value_offset;
  u8 data[];
};
//---------------------------------------------------------------------------
template <typename DataType> class RLE : public CompressionScheme<DataType> {
public:
  //---------------------------------------------------------------------------
  CompressionDetails compress(const DataType *src, const u32 size, u8 *dest,
                              const Statistics<DataType> *stats) override {
    auto &layout = *reinterpret_cast<RLELayout *>(dest);

    // Array of run-lengths
    auto *run_lengths = reinterpret_cast<u32 *>(layout.data);
    // Array of values
    vector<DataType> values;

    DataType cur = src[0];
    u32 run_length = 1;
    for (u32 i = 1; i < size; ++i) {
      if (src[i] == cur) {
        ++run_length;
      } else {
        run_lengths[values.size()] = run_length;
        values.push_back(cur);
        run_length = 1;
        cur = src[i];
      }
    }
    if (run_length > 1) {
      run_lengths[values.size()] = run_length;
      values.push_back(cur);
    }

    layout.value_offset = values.size() * sizeof(run_length);

    auto *varr =
        reinterpret_cast<DataType *>(layout.data + layout.value_offset);
    for (u32 i = 0; i < values.size(); ++i) {
      varr[i] = values[i];
    }

    u32 header_size = sizeof(RLELayout);
    u32 payload_size = layout.value_offset + values.size() * sizeof(DataType);
    return {header_size, payload_size};
  }
  //---------------------------------------------------------------------------
  void decompress(DataType *dest, const u32 size, const u8 *src) override {
    const auto &layout = *reinterpret_cast<const RLELayout *>(src);

    const auto *run_lengths = reinterpret_cast<const u32 *>(layout.data);
    const auto *values =
        reinterpret_cast<const DataType *>(layout.data + layout.value_offset);

    u32 pos = 0;
    for (u32 i = 0; i < layout.value_offset / sizeof(u32); ++i) {
      const u32 run_length = run_lengths[i];
      const DataType value = values[i];

      for (u32 j = 0; j < run_length; ++j) {
        dest[pos++] = value;
      }
    }
  }
  //---------------------------------------------------------------------------
  bool isPartitioningScheme() override { return false; }
};
//---------------------------------------------------------------------------
} // namespace compression