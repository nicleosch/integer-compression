#include <gtest/gtest.h>
#include <memory>
//---------------------------------------------------------------------------
#include "bitpacking/simd32bit/Block64BitPacking.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
// The answer to the ultimate question of life, the universe, and everything.
static constexpr INTEGER comp = 42;
static constexpr u32 size = 64;
static constexpr u8 bit = 7;
//---------------------------------------------------------------------------
// Testing the Equality Filter where there is supposed to be a single match.
TEST(Block64BitPackingTest, EQFilterSingle) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(size);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = i;
  }
  vector<u32> matches(size);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m128i[]>(64);
  bitpacking::simd32::block64::pack(reinterpret_cast<u32 *>(vec.data()),
                                    compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::block64::filter(compressed.get(), matches.data(), bit,
                                      pred);
  //---------------------------------------------------------------------------
  // All values should be 0 except the value at the offset of value 42.
  ASSERT_GT(matches[comp], 0);
  for (u32 i = 0; i < matches.size(); ++i) {
    if (i != comp)
      ASSERT_EQ(matches[i], 0);
  }
}
//---------------------------------------------------------------------------
// Testing the Equality Filter where there is supposed to be multiple matches.
TEST(Block64BitPackingTest, EQFilterMultiple) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(size);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = comp;
  }
  vec[comp] = 14;
  vector<u32> matches(size);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m128i[]>(64);
  bitpacking::simd32::block64::pack(reinterpret_cast<u32 *>(vec.data()),
                                    compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::block64::filter(compressed.get(), matches.data(), bit,
                                      pred);
  //---------------------------------------------------------------------------
  // All values should be 1 except the value at the offset of value 42.
  ASSERT_EQ(matches[comp], 0);
  for (u32 i = 0; i < matches.size(); ++i) {
    if (i != comp)
      ASSERT_GT(matches[i], 0);
  }
}
//---------------------------------------------------------------------------
// Testing the Inequality Filter where there is supposed to be a single match.
TEST(Block64BitPackingTest, NEQFilterSingle) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::INEQ, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(size);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = comp;
  }
  vec[comp] = 14;
  vector<u32> matches(size);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m128i[]>(64);
  bitpacking::simd32::block64::pack(reinterpret_cast<u32 *>(vec.data()),
                                    compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::block64::filter(compressed.get(), matches.data(), bit,
                                      pred);
  //---------------------------------------------------------------------------
  // All values should be 0 except the value at the offset of value 42.
  ASSERT_GT(matches[comp], 0);
  for (u32 i = 0; i < matches.size(); ++i) {
    if (i != comp)
      ASSERT_EQ(matches[i], 0);
  }
}
//---------------------------------------------------------------------------
// Testing the Inequality Filter where there is supposed to be multiple matches.
TEST(Block64BitPackingTest, NEQFilterMultiple) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::INEQ, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(size);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = i;
  }
  vector<u32> matches(size);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m128i[]>(64);
  bitpacking::simd32::block64::pack(reinterpret_cast<u32 *>(vec.data()),
                                    compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::block64::filter(compressed.get(), matches.data(), bit,
                                      pred);
  //---------------------------------------------------------------------------
  // All values should be 1 except the value at the offset of value 42.
  ASSERT_EQ(matches[comp], 0);
  for (u32 i = 0; i < matches.size(); ++i) {
    if (i != comp)
      ASSERT_GT(matches[i], 0);
  }
}
//---------------------------------------------------------------------------
// Testing the GreaterThan Filter where there is supposed to be a single match.
TEST(Block64BitPackingTest, GTFilterSingle) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::GT, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(size);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = comp;
  }
  vec[comp] = 56;
  vector<u32> matches(size);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m128i[]>(64);
  bitpacking::simd32::block64::pack(reinterpret_cast<u32 *>(vec.data()),
                                    compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::block64::filter(compressed.get(), matches.data(), bit,
                                      pred);
  //---------------------------------------------------------------------------
  // All values should be 0 except the value at the offset of value 42.
  ASSERT_GT(matches[comp], 0);
  for (u32 i = 0; i < matches.size(); ++i) {
    if (i != comp)
      ASSERT_EQ(matches[i], 0);
  }
}
//---------------------------------------------------------------------------
// Testing the GreaterThan Filter where there is supposed to be multiple
// matches.
TEST(Block64BitPackingTest, GTFilterMultiple) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::GT, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(size);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = i;
  }
  vector<u32> matches(size);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m128i[]>(64);
  bitpacking::simd32::block64::pack(reinterpret_cast<u32 *>(vec.data()),
                                    compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::block64::filter(compressed.get(), matches.data(), bit,
                                      pred);
  //---------------------------------------------------------------------------
  // All values at index larger than 42 should be larger than 0, else 0.
  for (u32 i = 0; i < matches.size(); ++i) {
    if (i > comp)
      ASSERT_GT(matches[i], 0);
    else
      ASSERT_EQ(matches[i], 0);
  }
}
//---------------------------------------------------------------------------
// Testing the LessThan Filter where there is supposed to be a single match.
TEST(Block64BitPackingTest, LTFilterSingle) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::LT, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(size);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = comp;
  }
  vec[comp] = 13;
  vector<u32> matches(size);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m128i[]>(64);
  bitpacking::simd32::block64::pack(reinterpret_cast<u32 *>(vec.data()),
                                    compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::block64::filter(compressed.get(), matches.data(), bit,
                                      pred);
  //---------------------------------------------------------------------------
  // All values should be 0 except the value at the offset of value 42.
  ASSERT_GT(matches[comp], 0);
  for (u32 i = 0; i < matches.size(); ++i) {
    if (i != comp)
      ASSERT_EQ(matches[i], 0);
  }
}
//---------------------------------------------------------------------------
// Testing the LessThan Filter where there is supposed to be multiple matches.
TEST(Block64BitPackingTest, LTFilterMultiple) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::LT, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(size);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = i;
  }
  vector<u32> matches(size);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m128i[]>(64);
  bitpacking::simd32::block64::pack(reinterpret_cast<u32 *>(vec.data()),
                                    compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::block64::filter(compressed.get(), matches.data(), bit,
                                      pred);
  //---------------------------------------------------------------------------
  // All values at index larger than 42 should be larger than 0, else 0.
  for (u32 i = 0; i < matches.size(); ++i) {
    if (i >= comp)
      ASSERT_EQ(matches[i], 0);
    else
      ASSERT_GT(matches[i], 0);
  }
}