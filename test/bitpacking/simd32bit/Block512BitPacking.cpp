#include <gtest/gtest.h>
#include <memory>
#include <random>
//---------------------------------------------------------------------------
#include "bitpacking/simd32bit/Block512BitPacking.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
// The answer to the ultimate question of life, the universe, and everything.
static constexpr INTEGER comp = 42;
static constexpr u8 bit = 9;
//---------------------------------------------------------------------------
// Testing the Equality Filter where there is supposed to be a single match.
TEST(AVX512BitPackingTest, EQFilterSingle) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(512);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = i;
  }
  vector<u32> matches(512);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::pack(reinterpret_cast<u32 *>(vec.data()),
                                   compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filter(compressed.get(), matches.data(), bit,
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
TEST(AVX512BitPackingTest, EQFilterMultiple) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(512);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = comp;
  }
  vec[comp] = 14;
  vector<u32> matches(512);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::pack(reinterpret_cast<u32 *>(vec.data()),
                                   compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filter(compressed.get(), matches.data(), bit,
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
TEST(AVX512BitPackingTest, NEQFilterSingle) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::INEQ, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(512);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = comp;
  }
  vec[comp] = 14;
  vector<u32> matches(512);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::pack(reinterpret_cast<u32 *>(vec.data()),
                                   compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filter(compressed.get(), matches.data(), bit,
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
TEST(AVX512BitPackingTest, NEQFilterMultiple) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::INEQ, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(512);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = i;
  }
  vector<u32> matches(512);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::pack(reinterpret_cast<u32 *>(vec.data()),
                                   compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filter(compressed.get(), matches.data(), bit,
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
TEST(AVX512BitPackingTest, GTFilterSingle) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::GT, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(512);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = comp;
  }
  vec[comp] = 128;
  vector<u32> matches(512);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::pack(reinterpret_cast<u32 *>(vec.data()),
                                   compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filter(compressed.get(), matches.data(), bit,
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
TEST(AVX512BitPackingTest, GTFilterMultiple) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::GT, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(512);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = i;
  }
  vector<u32> matches(512);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::pack(reinterpret_cast<u32 *>(vec.data()),
                                   compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filter(compressed.get(), matches.data(), bit,
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
TEST(AVX512BitPackingTest, LTFilterSingle) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::LT, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(512);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = comp;
  }
  vec[comp] = 13;
  vector<u32> matches(512);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::pack(reinterpret_cast<u32 *>(vec.data()),
                                   compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filter(compressed.get(), matches.data(), bit,
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
TEST(AVX512BitPackingTest, LTFilterMultiple) {
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::LT, comp);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(512);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = i;
  }
  vector<u32> matches(512);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::pack(reinterpret_cast<u32 *>(vec.data()),
                                   compressed.get(), bit);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filter(compressed.get(), matches.data(), bit,
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
//---------------------------------------------------------------------------
/// Compile-Time configuration parameters.
template <typename T, u8 kBit> struct Config {
  using DataType = T;
  //---------------------------------------------------------------------------
  static constexpr u8 kPackSize = kBit;
};
//---------------------------------------------------------------------------
template <typename Config>
class BP512CompDecompInvariant : public ::testing::Test {};
template <typename Config>
class BP512FastCompDecompInvariant : public ::testing::Test {};
template <typename Config> class BP512FastFilterEQ : public ::testing::Test {};
template <typename Config>
class BP512FastFilterMaskEQ : public ::testing::Test {};
//---------------------------------------------------------------------------
using C0 = Config<INTEGER, 0>;
using C1 = Config<INTEGER, 1>;
using C2 = Config<INTEGER, 2>;
using C3 = Config<INTEGER, 3>;
using C4 = Config<INTEGER, 4>;
using C5 = Config<INTEGER, 5>;
using C6 = Config<INTEGER, 6>;
using C7 = Config<INTEGER, 7>;
using C8 = Config<INTEGER, 8>;
using C9 = Config<INTEGER, 9>;
using C10 = Config<INTEGER, 10>;
using C11 = Config<INTEGER, 11>;
using C12 = Config<INTEGER, 12>;
using C13 = Config<INTEGER, 13>;
using C14 = Config<INTEGER, 14>;
using C15 = Config<INTEGER, 15>;
using C16 = Config<INTEGER, 16>;
using C17 = Config<INTEGER, 17>;
using C18 = Config<INTEGER, 18>;
using C19 = Config<INTEGER, 19>;
using C20 = Config<INTEGER, 20>;
using C21 = Config<INTEGER, 21>;
using C22 = Config<INTEGER, 22>;
using C23 = Config<INTEGER, 23>;
using C24 = Config<INTEGER, 24>;
using C25 = Config<INTEGER, 25>;
using C26 = Config<INTEGER, 26>;
using C27 = Config<INTEGER, 27>;
using C28 = Config<INTEGER, 28>;
using C29 = Config<INTEGER, 29>;
using C30 = Config<INTEGER, 30>;
using C31 = Config<INTEGER, 31>;
using C32 = Config<INTEGER, 32>;
using Configs =
    ::testing::Types<C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13,
                     C14, C15, C16, C17, C18, C19, C20, C21, C22, C23, C24, C25,
                     C26, C27, C28, C29, C30, C31, C32>;
using FastPackConfigs = ::testing::Types<C1, C2, C3, C4, C5, C6, C7, C8>;
//---------------------------------------------------------------------------
TYPED_TEST_CASE(BP512CompDecompInvariant, Configs);
TYPED_TEST(BP512CompDecompInvariant, CompDecompInvariant) {
  const u32 kSize = 512;
  //---------------------------------------------------------------------------
  // Generate random data.
  vector<INTEGER> vec(kSize);
  std::mt19937 rng(42);
  std::uniform_int_distribution<typename TypeParam::DataType> dist(0, 1);
  for (u32 i = 0; i < kSize; ++i) {
    vec[i] = dist(rng);
  }
  //---------------------------------------------------------------------------
  // Compression.
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::pack(reinterpret_cast<u32 *>(vec.data()),
                                   compressed.get(), TypeParam::kPackSize);
  //---------------------------------------------------------------------------
  // Decompression.
  vector<INTEGER> out(512);
  bitpacking::simd32::avx512::unpack(compressed.get(),
                                     reinterpret_cast<u32 *>(out.data()),
                                     TypeParam::kPackSize);
  //---------------------------------------------------------------------------
  for (u32 i = 0; i < out.size(); ++i) {
    ASSERT_EQ(out[i], vec[i]);
  }
}
//---------------------------------------------------------------------------
TYPED_TEST_CASE(BP512FastCompDecompInvariant, FastPackConfigs);
TYPED_TEST(BP512FastCompDecompInvariant, FastCompDecompInvariant) {
  const u32 kSize = 512;
  //---------------------------------------------------------------------------
  // Generate random data.
  vector<INTEGER> vec(kSize);
  std::mt19937 rng(42);
  std::uniform_int_distribution<typename TypeParam::DataType> dist(0, 1);
  for (u32 i = 0; i < kSize; ++i) {
    vec[i] = dist(rng);
  }
  //---------------------------------------------------------------------------
  // Compression.
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::packfast(reinterpret_cast<u32 *>(vec.data()),
                                       compressed.get(), TypeParam::kPackSize);
  //---------------------------------------------------------------------------
  // Decompression.
  vector<INTEGER> out(512);
  bitpacking::simd32::avx512::unpackfast(compressed.get(),
                                         reinterpret_cast<u32 *>(out.data()),
                                         TypeParam::kPackSize);
  //---------------------------------------------------------------------------
  for (u32 i = 0; i < out.size(); ++i) {
    ASSERT_EQ(out[i], vec[i]);
  }
}
//---------------------------------------------------------------------------
TYPED_TEST_CASE(BP512FastFilterEQ, FastPackConfigs);
TYPED_TEST(BP512FastFilterEQ, FilterEQ) {
  const u32 kSize = 512;
  //---------------------------------------------------------------------------
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, 1);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(kSize);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = 0;
  }
  vec[comp] = 1;
  vector<u8> matches(kSize);
  for (u32 i = 0; i < matches.size(); ++i) {
    matches[i] = 0;
  }
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::packfast(reinterpret_cast<u32 *>(vec.data()),
                                       compressed.get(), TypeParam::kPackSize);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filterfast(compressed.get(), matches.data(),
                                         TypeParam::kPackSize, pred);
  //---------------------------------------------------------------------------
  // All values should be 0 except the value at the offset of value 42.
  ASSERT_GT(matches[comp], 0);
  for (u32 i = 0; i < matches.size(); ++i) {
    if (i != comp)
      ASSERT_EQ(matches[i], 0);
  }
}
//---------------------------------------------------------------------------
TYPED_TEST_CASE(BP512FastFilterMaskEQ, FastPackConfigs);
TYPED_TEST(BP512FastFilterMaskEQ, FilterEQSingle) {
  const u32 kSize = 512;
  //---------------------------------------------------------------------------
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, 1);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(kSize);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = 0;
  }
  vec[comp] = 1;
  __m512i match_bitmap = _mm512_setzero_si512();
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::packfast(reinterpret_cast<u32 *>(vec.data()),
                                       compressed.get(), TypeParam::kPackSize);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filterfastmask(compressed.get(), &match_bitmap,
                                             TypeParam::kPackSize, pred);
  //---------------------------------------------------------------------------
  alignas(64) u32 ints[16];
  _mm512_store_si512((__m512i *)ints, match_bitmap);
  for (u32 i = 0; i < 16; ++i) {
    if (i == comp / 32) {
      ASSERT_EQ(ints[i], 1 << (comp % 32));
    } else {
      ASSERT_EQ(ints[i], 0);
    }
  }
}
//---------------------------------------------------------------------------
TYPED_TEST(BP512FastFilterMaskEQ, FilterEQMultiple) {
  const u32 kSize = 512;
  //---------------------------------------------------------------------------
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, 1);
  //---------------------------------------------------------------------------
  vector<INTEGER> vec(kSize);
  for (u32 i = 0; i < vec.size(); ++i) {
    vec[i] = 1;
  }
  vec[comp] = 0;
  __m512i match_bitmap = _mm512_setzero_si512();
  //---------------------------------------------------------------------------
  auto compressed = std::make_unique<__m512i[]>(64);
  bitpacking::simd32::avx512::packfast(reinterpret_cast<u32 *>(vec.data()),
                                       compressed.get(), TypeParam::kPackSize);
  //---------------------------------------------------------------------------
  bitpacking::simd32::avx512::filterfastmask(compressed.get(), &match_bitmap,
                                             TypeParam::kPackSize, pred);
  //---------------------------------------------------------------------------
  alignas(64) u32 ints[16];
  _mm512_store_si512((__m512i *)ints, match_bitmap);
  for (u32 i = 0; i < 16; ++i) {
    if (i == comp / 32) {
      ASSERT_EQ(ints[i], 0xFFFFFBFF);
    } else {
      ASSERT_EQ(ints[i], 0xFFFFFFFF);
    }
  }
}