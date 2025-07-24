#include <gtest/gtest.h>
//---------------------------------------------------------------------------
#include "schemes/Truncation.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after columnar compression and
// decompression.
TEST(TruncationTest, ColumnDecompressionInvariant32bit) {
  constexpr uint16_t kBlockSize = 512;
  //---------------------------------------------------------------------------
  /// Data.
  vector<INTEGER> data(kBlockSize << 4);
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] = i % 4;
  }
  //---------------------------------------------------------------------------
  /// Statistics.
  auto stats = Statistics<INTEGER>::generateFrom(data.data(), data.size());
  //---------------------------------------------------------------------------
  auto compression_out = std::make_unique<compression::u8[]>(data.size() * 8);
  Truncation<INTEGER, kBlockSize> trunc;
  trunc.compress(data.data(), data.size(), compression_out.get(), &stats);
  //---------------------------------------------------------------------------
  vector<compression::INTEGER> decompression_out(data.size());
  trunc.decompress(decompression_out.data(), data.size(),
                   compression_out.get());
  //---------------------------------------------------------------------------
  // verify
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(data.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after columnar compression and
// decompression.
TEST(TruncationTest, ColumnDecompressionInvariant64bit) {
  constexpr uint16_t kBlockSize = 512;
  //---------------------------------------------------------------------------
  /// Data.
  vector<BIGINT> data(kBlockSize << 4);
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] = i % 4;
  }
  //---------------------------------------------------------------------------
  /// Statistics.
  auto stats = Statistics<BIGINT>::generateFrom(data.data(), data.size());
  //---------------------------------------------------------------------------
  auto compression_out = std::make_unique<compression::u8[]>(data.size() * 8);
  Truncation<BIGINT, kBlockSize> trunc;
  trunc.compress(data.data(), data.size(), compression_out.get(), &stats);
  //---------------------------------------------------------------------------
  vector<compression::BIGINT> decompression_out(data.size());
  trunc.decompress(decompression_out.data(), data.size(),
                   compression_out.get());
  //---------------------------------------------------------------------------
  // verify
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(data.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after columnar compression and
// decompression.
TEST(TruncationTest, ColumnDecompressionInvariant32bitCbytes) {
  constexpr uint16_t kBlockSize = 512;
  //---------------------------------------------------------------------------
  /// Data.
  vector<INTEGER> data(kBlockSize << 4);
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] = i % 4;
  }
  //---------------------------------------------------------------------------
  /// Statistics.
  auto stats = Statistics<INTEGER>::generateFrom(data.data(), data.size());
  //---------------------------------------------------------------------------
  auto compression_out = std::make_unique<compression::u8[]>(data.size() * 8);
  Truncation<INTEGER, kBlockSize> trunc;
  u32 cbytes = 2;
  trunc.compress(data.data(), data.size(), compression_out.get(), &stats,
                 &cbytes);
  //---------------------------------------------------------------------------
  // The header contains the information about provided cbytes.
  ASSERT_EQ(*reinterpret_cast<u32 *>(compression_out.get()), cbytes);
  //---------------------------------------------------------------------------
  vector<compression::INTEGER> decompression_out(data.size());
  trunc.decompress(decompression_out.data(), data.size(),
                   compression_out.get());
  //---------------------------------------------------------------------------
  // verify
  for (size_t i = 0; i < data.size(); ++i) {
    ASSERT_EQ(data.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
/// Compile-Time configuration parameters.
template <typename T, u32 kBytes> struct Config {
  using DataType = T;
  //---------------------------------------------------------------------------
  static constexpr u32 cbytes = kBytes;
};
//---------------------------------------------------------------------------
template <typename Config> class TruncationFilter : public ::testing::Test {};
//---------------------------------------------------------------------------
using C0 = Config<INTEGER, 1>;
using C1 = Config<INTEGER, 2>;
using C2 = Config<INTEGER, 4>;
using C3 = Config<BIGINT, 8>;
using Configs = ::testing::Types<C0, C1, C2, C3>;
//---------------------------------------------------------------------------
TYPED_TEST_CASE(TruncationFilter, Configs);
TYPED_TEST(TruncationFilter, FilterEQSingle) {
  const u32 kSize = 512;
  const typename TypeParam::DataType comp = 42;
  //---------------------------------------------------------------------------
  algebra::Predicate<typename TypeParam::DataType> pred(
      algebra::PredicateType::EQ, 1);
  //---------------------------------------------------------------------------
  vector<typename TypeParam::DataType> vec(kSize);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = 0;
  }
  vec[comp] = 1;
  auto match_bitmap = std::make_unique<u8[]>(1024);
  //---------------------------------------------------------------------------
  /// Statistics.
  auto stats = Statistics<typename TypeParam::DataType>::generateFrom(
      vec.data(), vec.size());
  //---------------------------------------------------------------------------
  auto compression_out = std::make_unique<compression::u8[]>(vec.size() * 16);
  Truncation<typename TypeParam::DataType, kSize> trunc;
  //---------------------------------------------------------------------------
  // Compression.
  trunc.compress(vec.data(), vec.size(), compression_out.get(), &stats,
                 &TypeParam::cbytes);
  //---------------------------------------------------------------------------
  // Decompression.
  trunc.filter(compression_out.get(), vec.size(), match_bitmap.get(), pred);
  //---------------------------------------------------------------------------
  // Check results.
  alignas(64) u32 ints[16];
  _mm512_store_si512((__m512i *)ints,
                     *reinterpret_cast<__m512i *>(match_bitmap.get()));
  for (u32 i = 0; i < 16; ++i) {
    if (i == comp / 32) {
      EXPECT_EQ(ints[i], 1 << (comp % 32));
    } else {
      EXPECT_EQ(ints[i], 0);
    }
  }
}
TYPED_TEST(TruncationFilter, FilterEQMultiple) {
  const u32 kSize = 512;
  const typename TypeParam::DataType comp = 42;
  //---------------------------------------------------------------------------
  algebra::Predicate<typename TypeParam::DataType> pred(
      algebra::PredicateType::EQ, 1);
  //---------------------------------------------------------------------------
  vector<typename TypeParam::DataType> vec(kSize);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = 1;
  }
  vec[comp] = 0;
  auto match_bitmap = std::make_unique<u8[]>(1024);
  //---------------------------------------------------------------------------
  /// Statistics.
  auto stats = Statistics<typename TypeParam::DataType>::generateFrom(
      vec.data(), vec.size());
  //---------------------------------------------------------------------------
  auto compression_out = std::make_unique<compression::u8[]>(vec.size() * 16);
  Truncation<typename TypeParam::DataType, kSize> trunc;
  //---------------------------------------------------------------------------
  // Compression.
  trunc.compress(vec.data(), vec.size(), compression_out.get(), &stats,
                 &TypeParam::cbytes);
  //---------------------------------------------------------------------------
  // Decompression.
  trunc.filter(compression_out.get(), vec.size(), match_bitmap.get(), pred);
  //---------------------------------------------------------------------------
  // Check results.
  alignas(64) u32 ints[16];
  _mm512_store_si512((__m512i *)ints,
                     *reinterpret_cast<__m512i *>(match_bitmap.get()));
  for (u32 i = 0; i < 16; ++i) {
    if (i == comp / 32) {
      ASSERT_EQ(ints[i], 0xFFFFFBFF);
    } else {
      ASSERT_EQ(ints[i], 0xFFFFFFFF);
    }
  }
}