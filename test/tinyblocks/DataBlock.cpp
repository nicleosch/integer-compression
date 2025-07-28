#include <gtest/gtest.h>
//---------------------------------------------------------------------------
#include "storage/Column.hpp"
#include "tinyblocks/DataBlock.hpp"
//---------------------------------------------------------------------------
using namespace compression;
using namespace compression::tinyblocks;
//---------------------------------------------------------------------------
static constexpr u32 kSize = datablock::kDefaultSize * 2;
template <typename T> static constexpr vector<T> make_monotonic_data() {
  vector<T> data(kSize);
  for (u32 i = 0; i < kSize; ++i) {
    data[i] = i;
  }
  return data;
}
template <typename T> static constexpr vector<T> make_data() {
  vector<T> data(kSize);
  for (u32 i = 0; i < kSize; ++i) {
    data[i] = i % 8;
  }
  return data;
}
//---------------------------------------------------------------------------
/// Compile-Time configuration parameters.
template <typename T, algebra::PredicateType pred = algebra::PredicateType::EQ>
struct Config {
  using DataType = T;
  static constexpr algebra::PredicateType kPredicate = pred;
};
//---------------------------------------------------------------------------
/// Test-Suites
template <typename Config>
class DataBlockCompression : public ::testing::Test {};
template <typename Config> class DataBlockFiltering : public ::testing::Test {};
//---------------------------------------------------------------------------
using Config1 = Config<INTEGER>;
using Config2 = Config<INTEGER, algebra::PredicateType::GT>;
using Config3 = Config<INTEGER, algebra::PredicateType::LT>;
using Config4 = Config<INTEGER, algebra::PredicateType::INEQ>;
using Configs = ::testing::Types<Config1, Config2, Config3, Config4>;
TYPED_TEST_CASE(DataBlockCompression, Config1);
TYPED_TEST_CASE(DataBlockFiltering, Configs);
//---------------------------------------------------------------------------
/// Test that MONOTONIC compression works on datablock level.
TYPED_TEST(DataBlockCompression, MonoCompDecompInvariant) {
  using DataType = typename TypeParam::DataType;
  using DataBlocks = datablock::DataBlock<DataType, 512>;
  //---------------------------------------------------------------------------
  storage::Column<DataType> column(make_monotonic_data<DataType>());
  //---------------------------------------------------------------------------
  // compress
  auto compression_out =
      std::make_unique<u8[]>(column.size() * sizeof(DataType) * 2);
  DataBlocks tb;
  tb.compress(column.data(), column.size(), compression_out.get());
  const auto header =
      *reinterpret_cast<const DataBlocks::Header *>(compression_out.get());
  ASSERT_EQ(header.tag.scheme, datablock::Scheme::MONOTONIC);
  //---------------------------------------------------------------------------
  // decompress
  vector<INTEGER> decompression_out(column.size());
  tb.decompress(decompression_out.data(), column.size(), compression_out.get());
  //---------------------------------------------------------------------------
  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
/// Test that TRUNCATION compression works on datablock level.
TYPED_TEST(DataBlockCompression, TruncCompDecompInvariant) {
  using DataType = typename TypeParam::DataType;
  using DataBlocks =
      datablock::DataBlock<DataType, 512, datablock::kDefaultSize, false>;
  //---------------------------------------------------------------------------
  storage::Column<DataType> column(make_data<DataType>());
  //---------------------------------------------------------------------------
  // compress
  auto compression_out =
      std::make_unique<u8[]>(column.size() * sizeof(DataType) * 2);
  DataBlocks tb;
  tb.compress(column.data(), column.size(), compression_out.get());
  const auto header =
      *reinterpret_cast<const DataBlocks::Header *>(compression_out.get());
  ASSERT_EQ(header.tag.scheme, datablock::Scheme::TRUNCATION);
  //---------------------------------------------------------------------------
  // decompress
  vector<INTEGER> decompression_out(column.size());
  tb.decompress(decompression_out.data(), column.size(), compression_out.get());
  //---------------------------------------------------------------------------
  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
/// Test that filtering works on datablock level.
TYPED_TEST(DataBlockFiltering, Filtering) {
  using DataType = typename TypeParam::DataType;
  using DataBlocks = datablock::DataBlock<DataType, 512>;
  //---------------------------------------------------------------------------
  auto vec = make_monotonic_data<DataType>();
  algebra::Predicate<INTEGER> pred(TypeParam::kPredicate, 42);
  storage::Column<DataType> column(vec);
  //---------------------------------------------------------------------------
  vector<u32> matches(kSize);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  // compress
  auto compression_out =
      std::make_unique<u8[]>(column.size() * sizeof(DataType) * 2);
  DataBlocks tb;
  tb.compress(column.data(), column.size(), compression_out.get());
  //---------------------------------------------------------------------------
  // filter
  tb.filter(compression_out.get(), column.size(), pred, matches);
  //---------------------------------------------------------------------------
  for (u32 i = 0; i < matches.size(); ++i) {
    if constexpr (TypeParam::kPredicate == algebra::PredicateType::EQ) {
      if (vec[i] == 42)
        ASSERT_GT(matches[i], 0);
      else
        ASSERT_EQ(matches[i], 0);
    } else if constexpr (TypeParam::kPredicate == algebra::PredicateType::GT) {
      if (vec[i] > 42)
        ASSERT_GT(matches[i], 0);
      else
        ASSERT_EQ(matches[i], 0);
    } else if constexpr (TypeParam::kPredicate == algebra::PredicateType::LT) {
      if (vec[i] < 42)
        ASSERT_GT(matches[i], 0);
      else
        ASSERT_EQ(matches[i], 0);
    } else {
      if (vec[i] != 42)
        ASSERT_GT(matches[i], 0);
      else
        ASSERT_EQ(matches[i], 0);
    }
  }
}
