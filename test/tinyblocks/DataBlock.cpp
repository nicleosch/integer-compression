#include <gtest/gtest.h>
//---------------------------------------------------------------------------
#include "storage/Column.hpp"
#include "tinyblocks/DataBlock.hpp"
//---------------------------------------------------------------------------
using namespace compression;
using namespace compression::tinyblocks;
//---------------------------------------------------------------------------
static constexpr u32 kSize = datablock::kDefaultSize * 16;
template <typename T> static constexpr vector<T> make_data() {
  vector<T> data(kSize);
  for (u32 i = 0; i < kSize; ++i) {
    data[i] = i;
  }
  return data;
}
//---------------------------------------------------------------------------
/// Compile-Time configuration parameters.
template <typename T> struct Config {
  using DataType = T;
};
//---------------------------------------------------------------------------
/// Test-Suites
template <typename Config>
class DataBlockCompression : public ::testing::Test {};
template <typename Config> class DataBlockFiltering : public ::testing::Test {};
//---------------------------------------------------------------------------
using Configs = ::testing::Types<Config<INTEGER>>;
TYPED_TEST_CASE(DataBlockCompression, Configs);
TYPED_TEST_CASE(DataBlockFiltering, Configs);
//---------------------------------------------------------------------------
/// Test that monotonic compression works on datablock level.
TYPED_TEST(DataBlockCompression, CompDecompInvariant) {
  using DataType = typename TypeParam::DataType;
  using DataBlocks = datablock::DataBlock<DataType, 512>;
  //---------------------------------------------------------------------------
  storage::Column<DataType> column(make_data<DataType>());
  //---------------------------------------------------------------------------
  // compress
  auto compression_out =
      std::make_unique<u8[]>(column.size() * sizeof(DataType) * 2);
  DataBlocks tb;
  tb.compress(column.data(), column.size(), compression_out.get());
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
  auto vec = make_data<DataType>();
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, 42);
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
    if (vec[i] == 42)
      ASSERT_GT(matches[i], 0);
    else
      ASSERT_EQ(matches[i], 0);
  }
}
