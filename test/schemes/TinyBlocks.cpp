#include <gtest/gtest.h>
//---------------------------------------------------------------------------
#include "storage/Column.hpp"
#include "tinyblocks/DataBlock.hpp"
//---------------------------------------------------------------------------
using namespace compression;
using namespace compression::tinyblocks;
//---------------------------------------------------------------------------
struct Partsupp {
  static constexpr const char *path = "../data/tpch/sf1/partsupp.tbl";
  static constexpr u32 index = 0;
};
constexpr char kColumnSeperator = '|';
//---------------------------------------------------------------------------
enum class TinyBlockSize : u8 { k64, k128, k256, k512 };
//---------------------------------------------------------------------------
enum class DataType : u8 { kINT, kBIGINT };
//---------------------------------------------------------------------------
/// Compile-Time configuration parameters.
template <typename T, typename Column, u16 size, Scheme scheme>
struct TinyBlocksConfig {
  using DataType = T;
  using ColumnMeta = Column;
  //---------------------------------------------------------------------------
  static constexpr u16 kTinyBlockSize = size;
  static constexpr Scheme kScheme = scheme;
};
//---------------------------------------------------------------------------
/// A Test-Suite for TinyBlocks.
template <typename Config> class TinyBlocksTest : public ::testing::Test {};
//---------------------------------------------------------------------------
// Combinations to be tested.
using Config1 = TinyBlocksConfig<INTEGER, Partsupp, 128, Scheme::FOR>;
using Config2 = TinyBlocksConfig<INTEGER, Partsupp, 128, Scheme::RLE4>;
using Config3 = TinyBlocksConfig<INTEGER, Partsupp, 128, Scheme::RLE8>;
using Config4 = TinyBlocksConfig<INTEGER, Partsupp, 128, Scheme::DELTA>;
using Config5 = TinyBlocksConfig<INTEGER, Partsupp, 128, Scheme::PFOR>;
using Config6 = TinyBlocksConfig<INTEGER, Partsupp, 128, Scheme::PFOR_EBP>;
using Config7 = TinyBlocksConfig<INTEGER, Partsupp, 128, Scheme::PFOR_EP>;
using Config8 = TinyBlocksConfig<INTEGER, Partsupp, 128, Scheme::PFOR_DELTA>;
using Config9 = TinyBlocksConfig<INTEGER, Partsupp, 128, Scheme::PFOR_LEMIRE>;
using Config11 = TinyBlocksConfig<INTEGER, Partsupp, 256, Scheme::FOR>;
using Config12 = TinyBlocksConfig<INTEGER, Partsupp, 256, Scheme::RLE4>;
using Config13 = TinyBlocksConfig<INTEGER, Partsupp, 256, Scheme::RLE8>;
using Config14 = TinyBlocksConfig<INTEGER, Partsupp, 256, Scheme::DELTA>;
using Config15 = TinyBlocksConfig<INTEGER, Partsupp, 256, Scheme::PFOR>;
using Config16 = TinyBlocksConfig<INTEGER, Partsupp, 256, Scheme::PFOR_EBP>;
using Config17 = TinyBlocksConfig<INTEGER, Partsupp, 256, Scheme::PFOR_EP>;
using Config18 = TinyBlocksConfig<INTEGER, Partsupp, 256, Scheme::PFOR_DELTA>;
using Config19 = TinyBlocksConfig<INTEGER, Partsupp, 256, Scheme::PFOR_LEMIRE>;
using Config21 = TinyBlocksConfig<INTEGER, Partsupp, 512, Scheme::FOR>;
using Config22 = TinyBlocksConfig<INTEGER, Partsupp, 512, Scheme::RLE4>;
using Config23 = TinyBlocksConfig<INTEGER, Partsupp, 512, Scheme::RLE8>;
using Config24 = TinyBlocksConfig<INTEGER, Partsupp, 512, Scheme::DELTA>;
using Config25 = TinyBlocksConfig<INTEGER, Partsupp, 512, Scheme::PFOR>;
using Config26 = TinyBlocksConfig<INTEGER, Partsupp, 512, Scheme::PFOR_EBP>;
using Config27 = TinyBlocksConfig<INTEGER, Partsupp, 512, Scheme::PFOR_EP>;
using Config28 = TinyBlocksConfig<INTEGER, Partsupp, 512, Scheme::PFOR_DELTA>;
using Config29 = TinyBlocksConfig<INTEGER, Partsupp, 512, Scheme::PFOR_LEMIRE>;
using Config31 = TinyBlocksConfig<INTEGER, Partsupp, 64, Scheme::FOR>;
using Config32 = TinyBlocksConfig<INTEGER, Partsupp, 64, Scheme::RLE4>;
using Config33 = TinyBlocksConfig<INTEGER, Partsupp, 64, Scheme::RLE8>;
using Config34 = TinyBlocksConfig<INTEGER, Partsupp, 64, Scheme::DELTA>;
using Config35 = TinyBlocksConfig<INTEGER, Partsupp, 64, Scheme::PFOR>;
using Config36 = TinyBlocksConfig<INTEGER, Partsupp, 64, Scheme::PFOR_EBP>;
using Config37 = TinyBlocksConfig<INTEGER, Partsupp, 64, Scheme::PFOR_EP>;
using Config38 = TinyBlocksConfig<INTEGER, Partsupp, 64, Scheme::PFOR_DELTA>;
using Config39 = TinyBlocksConfig<INTEGER, Partsupp, 64, Scheme::PFOR_LEMIRE>;
using Configs = ::testing::Types<
    Config1, Config2, Config3, Config4, Config5, Config6, Config7, Config8,
    Config9, Config11, Config12, Config13, Config14, Config15, Config16,
    Config17, Config18, Config19, Config21, Config22, Config23, Config24,
    Config25, Config26, Config27, Config28, Config29, Config31, Config32,
    Config33, Config34, Config35, Config36, Config37, Config38, Config39>;
TYPED_TEST_CASE(TinyBlocksTest, Configs);
//---------------------------------------------------------------------------
/// A Test for TinyBlocks confirming compression and decompression does not
/// alter the data.
TYPED_TEST(TinyBlocksTest, CompDecompInvariant) {
  constexpr u16 kTinyBlockSize = TypeParam::kTinyBlockSize;
  //---------------------------------------------------------------------------
  using DataType = typename TypeParam::DataType;
  using DataBlocks = datablock::DataBlock<DataType, kTinyBlockSize>;
  //---------------------------------------------------------------------------
  auto column = storage::Column<DataType>::fromFile(
      TypeParam::ColumnMeta::path, TypeParam::ColumnMeta::index,
      kColumnSeperator);
  column.padToMultipleOf(kTinyBlockSize);
  //---------------------------------------------------------------------------
  // compress
  auto compression_out =
      std::make_unique<u8[]>(column.size() * sizeof(DataType) * 2);
  DataBlocks tb;
  tb.compress(column.data(), column.size(), compression_out.get(),
              &TypeParam::kScheme);
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