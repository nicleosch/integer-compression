#include <gtest/gtest.h>
#include <random>
//---------------------------------------------------------------------------
#include "storage/Column.hpp"
#include "tinyblocks/DataBlock.hpp"
//---------------------------------------------------------------------------
using namespace compression;
using namespace compression::tinyblocks;
//---------------------------------------------------------------------------
enum class ColumnName { kPartSupp, kNegativePositive, kMonotonic };
/// A wrapper for the ps_partkey column in TPCH.
template <typename T> struct Partsupp {
  static constexpr const char *path = "../data/tpch/sf1/partsupp.tbl";
  static constexpr u32 index = 0;
  static constexpr ColumnName name = ColumnName::kPartSupp;
  static vector<T> make_data() { return {}; }
};
/// A wrapper for a column including negative and positive values.
template <typename T> struct NegativePositive {
  static constexpr const char *path = nullptr;
  static constexpr u32 index = 0;
  static constexpr ColumnName name = ColumnName::kNegativePositive;
  static vector<T> make_data() {
    const u32 kSize = datablock::kDefaultSize * 16;
    vector<T> data(kSize);
    // Generate random values
    std::mt19937 rng(42);
    std::uniform_int_distribution<T> dist(
        std::numeric_limits<int8_t>::lowest() / 2,
        std::numeric_limits<int8_t>::max() / 2);
    for (u32 i = 0; i < kSize; ++i) {
      data[i] = dist(rng);
    }
    return data;
  }
};
/// A wrapper for a monotonically increasing column.
template <typename T> struct Monotonic {
  static constexpr const char *path = nullptr;
  static constexpr u32 index = 0;
  static constexpr ColumnName name = ColumnName::kMonotonic;
  static vector<T> make_data() {
    const u32 kSize = datablock::kDefaultSize * 16;
    vector<T> data(kSize);
    for (u32 i = 0; i < kSize; ++i) {
      // prevent monotonic compression on datablock level
      data[i] = i % (datablock::kDefaultSize / 2);
    }
    return data;
  }
};
constexpr char kColumnSeperator = '|';
//---------------------------------------------------------------------------
enum class TinyBlockSize : u8 { k64, k128, k256, k512 };
//---------------------------------------------------------------------------
enum class DataType : u8 { kINT, kBIGINT };
//---------------------------------------------------------------------------
/// Compile-Time configuration parameters.
template <typename T, typename Column, u16 size, Scheme scheme,
          algebra::PredicateType pred = algebra::PredicateType::EQ>
struct TinyBlocksConfig {
  using DataType = T;
  using ColumnMeta = Column;
  //---------------------------------------------------------------------------
  static constexpr u16 kTinyBlockSize = size;
  static constexpr Scheme kScheme = scheme;
  static constexpr algebra::PredicateType kPredicate = pred;
};
//---------------------------------------------------------------------------
/// Test-Suites for TinyBlocks.
template <typename Config>
class TinyBlocksTestCompression : public ::testing::Test {
protected:
  void TestDecompressionInvariant() {
    constexpr u16 kTinyBlockSize = Config::kTinyBlockSize;
    //---------------------------------------------------------------------------
    using DataType = typename Config::DataType;
    using DataBlocks = datablock::DataBlock<DataType, kTinyBlockSize>;
    //---------------------------------------------------------------------------
    storage::Column<DataType> column(vector<DataType>{});
    if constexpr (Config::ColumnMeta::name == ColumnName::kPartSupp) {
      // NOT Monotonically Increasing
      column = storage::Column<DataType>::fromFile(Config::ColumnMeta::path,
                                                   Config::ColumnMeta::index,
                                                   kColumnSeperator);
      column.padToMultipleOf(kTinyBlockSize);
    } else {
      column = storage::Column<DataType>(Config::ColumnMeta::make_data());
    }
    //---------------------------------------------------------------------------
    // compress
    auto compression_out =
        std::make_unique<u8[]>(column.size() * sizeof(DataType) * 2);
    DataBlocks tb;
    tb.compress(column.data(), column.size(), compression_out.get(),
                &Config::kScheme);
    //---------------------------------------------------------------------------
    // decompress
    vector<INTEGER> decompression_out(column.size());
    tb.decompress(decompression_out.data(), column.size(),
                  compression_out.get());
    //---------------------------------------------------------------------------
    // verify
    for (size_t i = 0; i < column.size(); ++i) {
      EXPECT_EQ(column.data()[i], decompression_out[i]);
    }
  }
};
template <typename Config>
class TinyBlocksTestCompression1 : public TinyBlocksTestCompression<Config> {};
template <typename Config>
class TinyBlocksTestCompression2 : public TinyBlocksTestCompression<Config> {};
template <typename Config>
class TinyBlocksTestFiltering : public ::testing::Test {};
//---------------------------------------------------------------------------
// Combinations to be tested.
using Config1 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::FOR>;
using Config100 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::FOR,
                                   algebra::PredicateType::GT>;
using Config101 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::FOR,
                                   algebra::PredicateType::LT>;
using Config102 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::FOR,
                                   algebra::PredicateType::INEQ>;
using Config2 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::RLE4>;
using Config3 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::RLE8>;
using Config4 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::DELTA>;
using Config5 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::PFOR>;
using Config6 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::PFOR_EBP>;
using Config7 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::PFOR_EP>;
using Config8 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::PFOR_DELTA>;
using Config9 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 128, Scheme::PFOR_LEMIRE>;
using Config11 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::FOR>;
using Config110 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::FOR,
                                   algebra::PredicateType::GT>;
using Config111 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::FOR,
                                   algebra::PredicateType::LT>;
using Config112 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::FOR,
                                   algebra::PredicateType::INEQ>;
using Config12 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::RLE4>;
using Config13 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::RLE8>;
using Config14 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::DELTA>;
using Config15 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::PFOR>;
using Config16 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::PFOR_EBP>;
using Config17 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::PFOR_EP>;
using Config18 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::PFOR_DELTA>;
using Config19 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 256, Scheme::PFOR_LEMIRE>;
using Config21 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::FOR>;
using Config120 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::FOR,
                                   algebra::PredicateType::GT>;
using Config121 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::FOR,
                                   algebra::PredicateType::LT>;
using Config122 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::FOR,
                                   algebra::PredicateType::INEQ>;
using Config22 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::RLE4>;
using Config23 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::RLE8>;
using Config24 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::DELTA>;
using Config25 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::PFOR>;
using Config26 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::PFOR_EBP>;
using Config27 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::PFOR_EP>;
using Config28 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::PFOR_DELTA>;
using Config29 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 512, Scheme::PFOR_LEMIRE>;
using Config31 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::FOR>;
using Config130 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::FOR,
                                   algebra::PredicateType::GT>;
using Config131 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::FOR,
                                   algebra::PredicateType::LT>;
using Config132 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::FOR,
                                   algebra::PredicateType::INEQ>;
using Config32 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::RLE4>;
using Config33 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::RLE8>;
using Config34 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::DELTA>;
using Config35 = TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::PFOR>;
using Config36 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::PFOR_EBP>;
using Config37 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::PFOR_EP>;
using Config38 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::PFOR_DELTA>;
using Config39 =
    TinyBlocksConfig<INTEGER, Partsupp<INTEGER>, 64, Scheme::PFOR_LEMIRE>;
using Config1220 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 128, Scheme::FOR>;
using Config222 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 128, Scheme::RLE4>;
using Config322 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 128, Scheme::RLE8>;
using Config522 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 128, Scheme::PFOR>;
using Config622 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 128, Scheme::PFOR_EBP>;
using Config722 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 128, Scheme::PFOR_EP>;
using Config922 = TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 128,
                                   Scheme::PFOR_LEMIRE>;
using Config1122 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 256, Scheme::FOR>;
using Config1222 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 256, Scheme::RLE4>;
using Config1322 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 256, Scheme::RLE8>;
using Config1522 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 256, Scheme::PFOR>;
using Config1622 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 256, Scheme::PFOR_EBP>;
using Config1722 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 256, Scheme::PFOR_EP>;
using Config1922 = TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 256,
                                    Scheme::PFOR_LEMIRE>;
using Config2122 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 512, Scheme::FOR>;
using Config2222 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 512, Scheme::RLE4>;
using Config2322 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 512, Scheme::RLE8>;
using Config2522 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 512, Scheme::PFOR>;
using Config2622 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 512, Scheme::PFOR_EBP>;
using Config2722 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 512, Scheme::PFOR_EP>;
using Config2922 = TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 512,
                                    Scheme::PFOR_LEMIRE>;
using Config3122 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 64, Scheme::FOR>;
using Config3222 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 64, Scheme::RLE4>;
using Config3322 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 64, Scheme::RLE8>;
using Config3522 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 64, Scheme::PFOR>;
using Config3622 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 64, Scheme::PFOR_EBP>;
using Config3722 =
    TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 64, Scheme::PFOR_EP>;
using Config3922 = TinyBlocksConfig<INTEGER, NegativePositive<INTEGER>, 64,
                                    Scheme::PFOR_LEMIRE>;
using Config40 =
    TinyBlocksConfig<INTEGER, Monotonic<INTEGER>, 64, Scheme::MONOTONIC>;
using Config41 =
    TinyBlocksConfig<INTEGER, Monotonic<INTEGER>, 128, Scheme::MONOTONIC>;
using Config42 =
    TinyBlocksConfig<INTEGER, Monotonic<INTEGER>, 256, Scheme::MONOTONIC>;
using Config43 =
    TinyBlocksConfig<INTEGER, Monotonic<INTEGER>, 512, Scheme::MONOTONIC>;
//---------------------------------------------------------------------------
/// The test cases declarations.
using Configs =
    ::testing::Types<Config1, Config2, Config3, Config4, Config5, Config6,
                     Config7, Config8, Config9, Config11, Config12, Config13,
                     Config14, Config15, Config16, Config17, Config18, Config19,
                     Config21, Config22, Config23, Config24, Config25, Config26,
                     Config27, Config28, Config29, Config31, Config32, Config33,
                     Config34, Config35, Config36, Config37, Config38, Config39,
                     Config40, Config41, Config42, Config43>;
using ConfigsNegPos =
    ::testing::Types<Config1220, Config222, Config322, Config522, Config622,
                     Config722, Config922, Config1122, Config1222, Config1322,
                     Config1522, Config1622, Config1722, Config1922, Config2122,
                     Config2222, Config2322, Config2522, Config2622, Config2722,
                     Config2922, Config3122, Config3222, Config3322, Config3522,
                     Config3622, Config3722, Config3922>;
using FilterConfigs =
    ::testing::Types<Config1, Config100, Config101, Config102, Config11,
                     Config110, Config111, Config112, Config21, Config120,
                     Config121, Config122, Config31, Config130, Config131,
                     Config132, Config40, Config41, Config42, Config43>;
TYPED_TEST_CASE(TinyBlocksTestCompression1, Configs);
TYPED_TEST_CASE(TinyBlocksTestCompression2, ConfigsNegPos);
TYPED_TEST_CASE(TinyBlocksTestFiltering, FilterConfigs);
//---------------------------------------------------------------------------
/// A Test for TinyBlocks confirming compression and decompression does not
/// alter the data.
TYPED_TEST(TinyBlocksTestCompression1, CompDecompInvariant) {
  this->TestDecompressionInvariant();
}
TYPED_TEST(TinyBlocksTestCompression2, CompDecompInvariant) {
  this->TestDecompressionInvariant();
}
//---------------------------------------------------------------------------
/// Testing TinyBlocks filtering capabilities.
TYPED_TEST(TinyBlocksTestFiltering, Filtering) {
  constexpr u16 kTinyBlockSize = TypeParam::kTinyBlockSize;
  constexpr u32 kSize = datablock::kDefaultSize * 16;
  //---------------------------------------------------------------------------
  using DataType = typename TypeParam::DataType;
  using DataBlocks = datablock::DataBlock<DataType, kTinyBlockSize>;
  //---------------------------------------------------------------------------
  algebra::Predicate<INTEGER> pred(TypeParam::kPredicate, 42);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(kSize);
  if constexpr (TypeParam::ColumnMeta::name == ColumnName::kPartSupp) {
    // NOT Monotonically Increasing
    for (u32 i = 0; i < vec.size(); ++i) {
      if (i % 42 == 0)
        vec[i] = 42;
      else
        vec[i] = i;
    }
  } else {
    // Monotonically Increasing
    vec = TypeParam::ColumnMeta::make_data();
  }
  //---------------------------------------------------------------------------
  vector<u32> matches(kSize);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  storage::Column<DataType> column(vec);
  auto compression_out =
      std::make_unique<u8[]>(column.size() * sizeof(DataType) * 2);
  DataBlocks tb;
  //---------------------------------------------------------------------------
  tb.compress(column.data(), column.size(), compression_out.get(),
              &TypeParam::kScheme);
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