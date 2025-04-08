#include <gtest/gtest.h>
//---------------------------------------------------------------------------
#include "core/Compressor.hpp"
#include "core/Decompressor.hpp"
#include "schemes/TinyBlocks.hpp"
#include "statistics/Statistics.hpp"
#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace c = compression;
namespace cs = c::storage;
namespace cc = c::compressor;
namespace cd = c::decompressor;
//---------------------------------------------------------------------------
// Verifies that the data remains unchanged after compression and decompression.
TEST(TinyBlocksTest, DecompressionInvariant) {
  auto path = "../data/tpch/sf1/partsupp.tbl";
  auto column = cs::Column::fromFile(path, 0, '|');
  constexpr c::u16 kBlockSize = 256;
  column.padToMultipleOf(kBlockSize);

  // compress
  std::unique_ptr<c::u8[]> compression_out;
  cc::compressTinyBlocks<kBlockSize>(column, compression_out);

  // decompress
  c::vector<c::INTEGER> decompression_out;
  cd::decompressTinyBlocks<kBlockSize>(decompression_out, column.size(),
                                       compression_out.get());

  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies correct handling of the case where all input values are identical.
TEST(TinyBlocksTest, OneValue) {
  constexpr c::u16 kBlockSize = 256;
  const c::INTEGER value = 42;

  c::vector<c::INTEGER> data(kBlockSize);
  for (c::u16 i = 0; i < kBlockSize; ++i) {
    data[i] = 42;
  }

  cs::Column column(data);

  // compress
  std::unique_ptr<c::u8[]> compression_out;
  auto size = cc::compressTinyBlocks<kBlockSize>(column, compression_out);

  // Nothing but the slot is stored in the compressed data
  EXPECT_EQ(size, c::TinyBlocksSlot::size());

  // The slot stores the correct information
  const auto &slot =
      *reinterpret_cast<c::TinyBlocksSlot *>(compression_out.get());
  EXPECT_EQ(slot.reference, 42);
  EXPECT_EQ(slot.offset, size);
  EXPECT_EQ(slot.pack_size, 0);

  // decompress
  c::vector<c::INTEGER> decompression_out;
  cd::decompressTinyBlocks<kBlockSize>(decompression_out, column.size(),
                                       compression_out.get());

  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}
//---------------------------------------------------------------------------
// Verifies correct handling of the case where input values increase
// monotincally.
TEST(TinyBlocksTest, IncreaseMonotonically) {
  constexpr c::u16 kBlockSize = 256;
  const c::INTEGER value = 42;

  c::u8 step_size = 2;

  c::vector<c::INTEGER> data(kBlockSize);
  for (c::u16 i = 0; i < kBlockSize; ++i) {
    data[i] = i * step_size;
  }

  cs::Column column(data);

  // compress
  std::unique_ptr<c::u8[]> compression_out;
  auto size = cc::compressTinyBlocks<kBlockSize>(column, compression_out);

  // Nothing but the slot is stored in the compressed data
  EXPECT_EQ(size, c::TinyBlocksSlot::size());

  // The slot stores the correct information
  const auto &slot =
      *reinterpret_cast<c::TinyBlocksSlot *>(compression_out.get());
  EXPECT_EQ(slot.reference, 0);
  EXPECT_EQ(slot.offset, size);
  EXPECT_EQ(slot.pack_size, 64 + step_size);

  // decompress
  c::vector<c::INTEGER> decompression_out;
  cd::decompressTinyBlocks<kBlockSize>(decompression_out, column.size(),
                                       compression_out.get());

  // verify
  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(column.data()[i], decompression_out[i]);
  }
}