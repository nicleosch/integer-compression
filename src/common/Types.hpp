//---------------------------------------------------------------------------
// Inspired by https://github.com/maxi-k/btrblocks
//---------------------------------------------------------------------------
#pragma once
//---------------------------------------------------------------------------
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
//---------------------------------------------------------------------------
using s8 = int8_t;
using s16 = int16_t;
using s32 = int32_t;
using s64 = int64_t;
//---------------------------------------------------------------------------
using INTEGER = s32;
using BIGINT = s64;
using MatchVector = std::vector<u32>;
using Match = MatchVector::value_type;
//---------------------------------------------------------------------------
using std::string;
using std::unordered_map;
using std::vector;
//---------------------------------------------------------------------------
enum class ColumnType : u8 {
  kInteger = 0,
  kBigInt = 0,
};
} // namespace compression