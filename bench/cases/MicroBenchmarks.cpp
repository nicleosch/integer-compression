#include <filesystem>
#include <fstream>
#include <random>
//---------------------------------------------------------------------------
#include "common/Encodings.hpp"
#include "common/PerfEvent.hpp"
#include "common/Types.hpp"
#include "extern/BtrBlocks.hpp"
#include "schemes/Truncation.hpp"
#include "schemes/Uncompressed.hpp"
#include "tinyblocks/TinyBlocks.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
namespace benchmarks {
//---------------------------------------------------------------------------
/// @brief Constants required for the benchmarks.
static const u32 kTupleCount = 512;
static const u32 kIterations = 1024 * 1024;
static std::vector<INTEGER> vec = [] {
  std::vector<INTEGER> v(kTupleCount);
  for (u32 i = 0; i < v.size(); ++i) {
    if (i / 8 % 2 == 0)
      v[i] = i % 2;
    else
      v[i] = (i + 1) % 2;
  }
  return v;
}();
static const u64 kBigTupleCount = 1ull << 30;
static const u32 kBigIterations = 16;
static const std::vector<INTEGER> bigVec = [] {
  std::vector<INTEGER> v(kBigTupleCount);
  for (u32 i = 0; i < v.size(); ++i) {
    if (i / 8 % 2 == 0)
      v[i] = i % 2;
    else
      v[i] = (i + 1) % 2;
  }
  return v;
}();
static const algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, 1);
static std::ofstream null_stream("/dev/null");
static const char *kResultFolder = "results/";
static const char *kFilterFolder = "results/filter/";
/// @brief TinyBlocks compression schemes
static const tinyblocks::Scheme schemes[] = {
    tinyblocks::Scheme::FOR,         tinyblocks::Scheme::RLE4,
    tinyblocks::Scheme::RLE8,        tinyblocks::Scheme::PFOR,
    tinyblocks::Scheme::PFOR_EP,     tinyblocks::Scheme::PFOR_EBP,
    tinyblocks::Scheme::PFOR_LEMIRE,
};
static const tinyblocks::Scheme deltaSchemes[] = {
    tinyblocks::Scheme::DELTA,
    tinyblocks::Scheme::PFOR_DELTA,
};
/// @brief BtrBlocks compression schemes
static const btrblocks::IntegerSchemeType btrblocksSchemes[] = {
    btrblocks::IntegerSchemeType::DICT,
    btrblocks::IntegerSchemeType::RLE,
    btrblocks::IntegerSchemeType::PFOR,
    btrblocks::IntegerSchemeType::BP,
};
static const btrblocks::IntegerSchemeType btrblocksDeltaSchemes[] = {
    btrblocks::IntegerSchemeType::PFOR_DELTA,
};
/// @brief The different types of datasets we're benchmarking on.
enum class Data { kRandom, kOneValue, kMonotonic };
static const char *toString(Data d) {
  switch (d) {
  case Data::kRandom:
    return "Random";
  case Data::kOneValue:
    return "OneValue";
  case Data::kMonotonic:
    return "Monotonic";
  default:
    return "Unknown";
  }
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
namespace utils {
/// @brief Iterates over a large block of data trying to trash the CPU caches.
static void thrashCPUCaches() {
  const u64 size = 100 * 1024 * 1024; // 100MB
  std::vector<uint8_t> block(size);
  volatile uint64_t sink = 0;
  for (size_t i = 0; i < block.size(); ++i) {
    sink += block[i];
  }
}
//---------------------------------------------------------------------------
/// @brief Prints the report header to given file.
static void printReport(PerfEvent &event, std::ostream &file) {
  event.printReport(file, null_stream, 1);
  file << ", Cycles/Tuple, Instructions/Tuple, Seconds, "
          "UC-Bandwidth(GB/s), C-Bandwidth(GB/s), Bit\n";
}
//---------------------------------------------------------------------------
/// @brief Starts cpu counters and writes the report header to the file.
static void startCounters(PerfEvent &event, std::ostream &file) {
  printReport(event, file);
  event.startCounters();
}
//---------------------------------------------------------------------------
/// @brief Stops CPU counters and writes the report payload to the file.
static void stopCounters(PerfEvent &event, std::ostream &file,
                         u64 tuple_count = kTupleCount,
                         u64 iterations = kIterations, u32 pack_size = 0) {
  event.stopCounters();
  event.printReport(null_stream, file, 1);
  file << ", "
       << static_cast<double>(event.getCounter("cycles")) /
              (tuple_count * iterations)
       << ", "
       << static_cast<double>(event.getCounter("instructions")) /
              (tuple_count * iterations)
       << ", " << event.getDuration() << ", "
       << static_cast<double>((tuple_count * 4) // Uncompressed Integers
                              * iterations      // Iterations
                              / 1'000'000'000)  // Byte -> GB
              / event.getDuration()
       << ", "
       << static_cast<double>(
              (tuple_count * pack_size / 8) // Compressed Integers
              * iterations                  // Iterations
              / 1'000'000'000)              // Byte -> GB
              / event.getDuration()
       << ", " << pack_size << "\n";
}
//---------------------------------------------------------------------------
/// @brief Generates a vector of random values that require "pack_size" bits
/// to represent.
/// @param data The vector to be filled with data.
/// @param count The size of the vector.
/// @param pack_size The number of bits required to store the highest value
/// within a TinyBlock.
template <typename T, u16 kTinyBlocksSize>
static void generateRandomValues(vector<T> &data, const u32 count,
                                 const u8 pack_size = sizeof(T) * 8) {
  data.resize(count);
  //---------------------------------------------------------------------------
  std::mt19937 gen(42);
  std::bernoulli_distribution dist(0.5);
  //---------------------------------------------------------------------------
  if (pack_size == 0) {
    std ::fill(data.begin(), data.end(), 0);
    return;
  }
  //---------------------------------------------------------------------------
  u32 max = 1ULL << pack_size;
  for (u32 i = 0; i < count; ++i) {
    if (i % kTinyBlocksSize == 0) {
      if (pack_size == 32) {
        // data[i] = std::numeric_limits<T>::max();
        // data[i == 0 ? i + 1 : i - 1] = std::numeric_limits<T>::min();

        // TODO: Fix: Handle pack size 32 through all tinyblocks-schemes.
        // Thus, for now we use pack size 31 instead.
        data[i] = static_cast<T>((1ULL << 31) - 1);
      } else {
        data[i] = static_cast<T>(max - 1);
      }
    } else {
      data[i] = dist(gen);
    }
  }
};
//---------------------------------------------------------------------------
/// @brief Generates a vector of monotonically increasing values that require
/// "pack_size" bits to represent.
/// @param data The vector to be filled with data.
/// @param count The size of the vector.
/// @param pack_size The number of bits required to store the highest value
/// within a TinyBlock.
template <typename T>
static void generateMonotonicValues(vector<T> &data, const u32 count,
                                    const u8 pack_size = sizeof(T) * 8) {
  data.resize(count);
  //---------------------------------------------------------------------------
  u32 max = 1ULL << pack_size;
  for (u32 i = 0; i < count; ++i) {
    if (i < max)
      data[i] = static_cast<T>(i);
    else
      data[i] = static_cast<T>(max - 1);
  }
};
//---------------------------------------------------------------------------
/// @brief Generates a vector of a single value that require
/// "pack_size" bits to represent.
/// @param data The vector to be filled with data.
/// @param count The size of the vector.
/// @param pack_size The number of bits required to store the highest value
/// within a TinyBlock.
template <typename T>
static void generateOneValue(vector<T> &data, const u32 count,
                             const u8 pack_size = sizeof(T) * 8) {
  data.resize(count);
  //---------------------------------------------------------------------------
  u32 max = 1ULL << pack_size;
  for (u32 i = 0; i < count; ++i) {
    data[i] = static_cast<T>(max - 1);
  };
}
//---------------------------------------------------------------------------
template <typename T, Data kData>
static void generateData(vector<T> &data, const u32 count,
                         const u8 pack_size = sizeof(T) * 8) {
  if constexpr (kData == Data::kRandom)
    utils::generateRandomValues<T, 64>(data, kTupleCount, pack_size);
  else if constexpr (kData == Data::kOneValue)
    utils::generateOneValue<T>(data, kTupleCount, pack_size);
  else
    utils::generateMonotonicValues<T>(data, kTupleCount, pack_size);
}
//---------------------------------------------------------------------------
} // namespace utils
//---------------------------------------------------------------------------

/// @brief Measures decompression speed for provided encodings and data.
template <typename T, EncodingType kType, Data kData>
static void decompressionMicroBenchmark() {
  const u8 kBits = sizeof(T) * 8;
  //---------------------------------------------------------------------------
  // Setup
  std::unique_ptr<Encoding<T>> encoding;
  if constexpr (kType == EncodingType::kUncompressed) {
    encoding = std::make_unique<UncompressedEncoding<T>>();
  } else if constexpr (kType == EncodingType::kTinyBlocks64) {
    encoding = std::make_unique<TinyBlocksEncoding<T, 64>>();
  } else if constexpr (kType == EncodingType::kTinyBlocks128) {
    encoding = std::make_unique<TinyBlocksEncoding<T, 128>>();
  } else if constexpr (kType == EncodingType::kTinyBlocks256) {
    encoding = std::make_unique<TinyBlocksEncoding<T, 256>>();
  } else if constexpr (kType == EncodingType::kTinyBlocks512) {
    encoding = std::make_unique<TinyBlocksEncoding<T, 512>>();
  } else if constexpr (kType == EncodingType::kBtrBlocks1) {
    encoding = std::make_unique<BtrBlocksEncoding<T, 1>>();
  } else if constexpr (kType == EncodingType::kBtrBlocks3) {
    encoding = std::make_unique<BtrBlocksEncoding<T, 3>>();
  } else if constexpr (kType == EncodingType::kBtrBlocks1_256) {
    encoding = std::make_unique<BtrBlocksEncoding<T, 1, 256>>();
  } else if constexpr (kType == EncodingType::kBtrBlocks3_256) {
    encoding = std::make_unique<BtrBlocksEncoding<T, 3, 256>>();
  } else {
    static_assert(false, "Unsupported encoding type!");
  }
  //---------------------------------------------------------------------------
  // Compression & Decompression
  auto cbuffer = std::make_unique<u8[]>(kTupleCount * sizeof(T) * 2);
  if constexpr (kType == EncodingType::kTinyBlocks64 ||
                kType == EncodingType::kTinyBlocks128 ||
                kType == EncodingType::kTinyBlocks256 ||
                kType == EncodingType::kTinyBlocks512) {
    for (const auto &scheme : schemes) {
      // The file to write to.
      std::ofstream bp_file(
          std::string(kResultFolder) + std::string(toString(kType)) + "_" +
          std::to_string(static_cast<u32>(scheme)) + "_" +
          std::to_string(sizeof(T) * 8) + "bit" + toString(kData) + ".csv");
      //---------------------------------------------------------------------------
      for (u8 pack_size = kBits - 31; pack_size <= kBits; ++pack_size) {
        //---------------------------------------------------------------------------
        std::cout << "Starting " << std::to_string(static_cast<u32>(scheme))
                  << " with pack size: "
                  << std::to_string(static_cast<u32>(pack_size)) << std::endl;
        // The buffer to compress into.
        cbuffer = std::make_unique<u8[]>(kTupleCount * sizeof(T) * 2);
        vector<T> column;
        utils::generateData<T, kData>(column, kTupleCount, pack_size);
        //---------------------------------------------------------------------------
        // Compress
        encoding->compress(column, cbuffer.get(), &scheme);
        //---------------------------------------------------------------------------
        // Register cpu counters
        PerfEvent perf_event;
        if (pack_size == kBits - 31) {
          perf_event.printReport(bp_file, null_stream, 1);
          bp_file << ", Cycles/Tuple, Instructions/Tuple\n";
        }
        perf_event.startCounters();
        //---------------------------------------------------------------------------
        // Decompress
        for (u32 i = 0; i < kIterations; ++i) {
          encoding->decompress(cbuffer.get(), column);
        }
        //---------------------------------------------------------------------------
        utils::stopCounters(perf_event, bp_file);
      }
    }
  } else if constexpr (kType == EncodingType::kUncompressed ||
                       kType == EncodingType::kBtrBlocks3 ||
                       kType == EncodingType::kBtrBlocks3_256) {
    vector<T> column;
    utils::generateData<T, kData>(column, kTupleCount);
    //---------------------------------------------------------------------------
    encoding->compress(column, cbuffer.get(), nullptr);
    //---------------------------------------------------------------------------
    PerfEvent perf_event;
    std::ofstream bp_file(std::string(kResultFolder) + toString(kType) +
                          std::to_string(sizeof(T) * 8) + "bit" +
                          toString(kData) + ".csv");
    utils::startCounters(perf_event, bp_file);
    //---------------------------------------------------------------------------
    for (u32 i = 0; i < kIterations; ++i) {
      encoding->decompress(cbuffer.get(), column);
    }
    //---------------------------------------------------------------------------
    utils::stopCounters(perf_event, bp_file);
  } else {
    for (const auto &scheme : btrblocksSchemes) {
      // The file to write to.
      std::ofstream bp_file(
          std::string(kResultFolder) + std::string(toString(kType)) + "_" +
          std::to_string(static_cast<u32>(scheme)) + "_" +
          std::to_string(sizeof(T) * 8) + "bit" + toString(kData) + ".csv");
      //---------------------------------------------------------------------------
      for (u8 pack_size = kBits - 31; pack_size <= kBits; ++pack_size) {
        // The buffer to compress into.
        cbuffer = std::make_unique<u8[]>(kTupleCount * sizeof(T) * 2);
        vector<T> column;
        utils::generateData<T, kData>(column, kTupleCount, pack_size);
        //---------------------------------------------------------------------------
        std::cout << "Starting " << std::to_string(static_cast<u32>(scheme))
                  << " with pack size: "
                  << std::to_string(static_cast<u32>(pack_size)) << std::endl;
        //---------------------------------------------------------------------------
        // Compress
        encoding->compress(column, cbuffer.get(), &scheme);
        //---------------------------------------------------------------------------
        // Register cpu counters
        PerfEvent perf_event;
        if (pack_size == kBits - 31) {
          perf_event.printReport(bp_file, null_stream, 1);
          bp_file << ", Cycles/Tuple, Instructions/Tuple\n";
        }
        perf_event.startCounters();
        //---------------------------------------------------------------------------
        // Decompress
        for (u32 i = 0; i < kIterations; ++i) {
          encoding->decompress(cbuffer.get(), column);
        }
        //---------------------------------------------------------------------------
        utils::stopCounters(perf_event, bp_file);
      }
    }
  }
}

/// @brief Measures filter speed for provided encodings and data.
template <typename T, EncodingType kType, Data kData>
static void filterMicroBenchmark() {
  const u8 kBits = sizeof(T) * 8;
  algebra::Predicate<INTEGER> pred(algebra::PredicateType::EQ, 1);
  //---------------------------------------------------------------------------
  // Setup
  std::unique_ptr<Encoding<T>> encoding;
  if constexpr (kType == EncodingType::kUncompressed) {
    encoding = std::make_unique<UncompressedEncoding<T>>();
  } else if constexpr (kType == EncodingType::kTinyBlocks64) {
    encoding = std::make_unique<TinyBlocksEncoding<T, 64>>();
  } else if constexpr (kType == EncodingType::kTinyBlocks128) {
    encoding = std::make_unique<TinyBlocksEncoding<T, 128>>();
  } else if constexpr (kType == EncodingType::kTinyBlocks256) {
    encoding = std::make_unique<TinyBlocksEncoding<T, 256>>();
  } else if constexpr (kType == EncodingType::kTinyBlocks512) {
    encoding = std::make_unique<TinyBlocksEncoding<T, 512>>();
  } else if constexpr (kType == EncodingType::kBtrBlocks1) {
    encoding = std::make_unique<BtrBlocksEncoding<T, 1>>();
  } else if constexpr (kType == EncodingType::kBtrBlocks3) {
    encoding = std::make_unique<BtrBlocksEncoding<T, 3>>();
  } else if constexpr (kType == EncodingType::kBtrBlocks1_256) {
    encoding = std::make_unique<BtrBlocksEncoding<T, 1, 256>>();
  } else if constexpr (kType == EncodingType::kBtrBlocks3_256) {
    encoding = std::make_unique<BtrBlocksEncoding<T, 3, 256>>();
  } else {
    static_assert(false, "Unsupported encoding type!");
  }
  //---------------------------------------------------------------------------
  // Compression & Decompression
  auto cbuffer = std::make_unique<u8[]>(kTupleCount * sizeof(T) * 2);
  if constexpr (kType == EncodingType::kTinyBlocks512) {
    for (const auto &scheme : schemes) {
      // The file to write to.
      std::ofstream bp_file(
          std::string(kFilterFolder) + std::string(toString(kType)) + "_" +
          std::to_string(static_cast<u32>(scheme)) + "_" +
          std::to_string(sizeof(T) * 8) + "bit" + toString(kData) + ".csv");
      //---------------------------------------------------------------------------
      for (u8 pack_size = 1; pack_size <= kBits; ++pack_size) {
        //---------------------------------------------------------------------------
        std::cout << "Starting " << std::to_string(static_cast<u32>(scheme))
                  << " with pack size: "
                  << std::to_string(static_cast<u32>(pack_size)) << std::endl;
        // The buffer to compress into.
        cbuffer = std::make_unique<u8[]>(kTupleCount * sizeof(T) * 2);
        vector<T> column;
        utils::generateData<T, kData>(column, kTupleCount, pack_size);
        MatchVector mv(column.size());
        //---------------------------------------------------------------------------
        // Compress
        encoding->compress(column, cbuffer.get(), &scheme);
        //---------------------------------------------------------------------------
        // Register cpu counters
        PerfEvent perf_event;
        if (pack_size == kBits - 31) {
          perf_event.printReport(bp_file, null_stream, 1);
          bp_file << ", Cycles/Tuple, Instructions/Tuple\n";
        }
        perf_event.startCounters();
        //---------------------------------------------------------------------------
        // Decompress
        for (u32 i = 0; i < kIterations; ++i) {
          encoding->filter(cbuffer.get(), column.size(), pred, mv);
        }
        //---------------------------------------------------------------------------
        utils::stopCounters(perf_event, bp_file);
      }
    }
  } else {
    static_assert(false);
  }
}

/// @brief Measures decompression & filtering speed for SIMD-Bitpacking
/// techniques.
template <typename T> static void bpMicroBenchmark() {
  // Filtering using 32 bit lanes.
  std::ofstream bp1_file(std::string(kFilterFolder) + "32bitLaneFilter.csv");
  for (u32 ps = 1; ps < 33; ++ps) {
    auto compressed = std::make_unique<__m512i[]>(64);
    bitpacking::simd32::avx512::pack(reinterpret_cast<const u32 *>(vec.data()),
                                     compressed.get(), ps);
    //---------------------------------------------------------------------------
    vector<u32> mv(kTupleCount);
    // Register cpu counters
    PerfEvent perf_event;
    if (ps == 1)
      utils::printReport(perf_event, bp1_file);
    perf_event.startCounters();
    //---------------------------------------------------------------------------
    // Decompress
    for (u32 i = 0; i < kIterations; ++i) {
      bitpacking::simd32::avx512::filter(compressed.get(), mv.data(), ps, pred);
    }
    //---------------------------------------------------------------------------
    utils::stopCounters(perf_event, bp1_file, kTupleCount, kIterations, ps);
  }
  // Filtering using 8 bit lanes.
  std::ofstream bp2_file(std::string(kFilterFolder) + "8bitLaneFilter.csv");
  for (u32 ps = 1; ps < 9; ++ps) {
    //---------------------------------------------------------------------------
    auto compressed = std::make_unique<__m512i[]>(64);
    bitpacking::simd32::avx512::packfast(
        reinterpret_cast<const u32 *>(vec.data()), compressed.get(), ps);
    //---------------------------------------------------------------------------
    vector<u8> mv(kTupleCount);
    // Register cpu counters
    PerfEvent perf_event;
    if (ps == 1)
      utils::printReport(perf_event, bp2_file);
    perf_event.startCounters();
    //---------------------------------------------------------------------------
    // Decompress
    for (u32 i = 0; i < kIterations; ++i) {
      bitpacking::simd32::avx512::filterfast(compressed.get(), mv.data(), ps,
                                             pred);
    }
    //---------------------------------------------------------------------------
    utils::stopCounters(perf_event, bp2_file, kTupleCount, kIterations, ps);
  }
  //---------------------------------------------------------------------------
  // Unpacking using 8 bit lanes.
  std::ofstream bp3_file(std::string(kFilterFolder) + "8bitLaneUnpacking.csv");
  for (u32 ps = 1; ps < 9; ++ps) {
    auto compressed = std::make_unique<__m512i[]>(64);
    bitpacking::simd32::avx512::packfast(
        reinterpret_cast<const u32 *>(vec.data()), compressed.get(), ps);
    //---------------------------------------------------------------------------
    // Register cpu counters
    PerfEvent perf_event;
    if (ps == 1)
      utils::printReport(perf_event, bp3_file);
    perf_event.startCounters();
    //---------------------------------------------------------------------------
    // Decompress
    for (u32 i = 0; i < kIterations; ++i) {
      bitpacking::simd32::avx512::unpackfast(
          compressed.get(), reinterpret_cast<u32 *>(vec.data()), ps);
    }
    //---------------------------------------------------------------------------
    utils::stopCounters(perf_event, bp3_file, kTupleCount, kIterations, ps);
    //---------------------------------------------------------------------------
    // Unpacking using 32 bit lanes.
    std::ofstream bp4_file(std::string(kFilterFolder) +
                           "32bitLaneUnpacking.csv");
    for (u32 ps = 1; ps < 9; ++ps) {
      auto compressed = std::make_unique<__m512i[]>(64);
      bitpacking::simd32::avx512::pack(
          reinterpret_cast<const u32 *>(vec.data()), compressed.get(), ps);
      //---------------------------------------------------------------------------
      // Register cpu counters
      PerfEvent perf_event;
      if (ps == 1)
        utils::printReport(perf_event, bp4_file);
      perf_event.startCounters();
      //---------------------------------------------------------------------------
      // Decompress
      for (u32 i = 0; i < kIterations; ++i) {
        bitpacking::simd32::avx512::unpack(
            compressed.get(), reinterpret_cast<u32 *>(vec.data()), ps);
      }
      //---------------------------------------------------------------------------
      utils::stopCounters(perf_event, bp4_file, kTupleCount, kIterations, ps);
    }
  }
  // Filtering using Prof. Neumanns bithacks.
  std::ofstream bp5_file(std::string(kFilterFolder) + "BitHackFilter.csv");
  for (u32 ps = 1; ps < 9; ++ps) {
    if (ps == 3 || (ps > 4 && ps < 8))
      continue;
    //---------------------------------------------------------------------------
    auto compressed = std::make_unique<__m512i[]>(64);
    bitpacking::simd32::avx512::packfast(
        reinterpret_cast<const u32 *>(vec.data()), compressed.get(), ps);
    //---------------------------------------------------------------------------
    __m512i match_bitmap;
    // Register cpu counters
    PerfEvent perf_event;
    if (ps == 1)
      utils::printReport(perf_event, bp5_file);
    perf_event.startCounters();
    //---------------------------------------------------------------------------
    // Decompress
    for (u32 i = 0; i < kIterations; ++i) {
      bitpacking::simd32::avx512::filterbithack(compressed.get(), &match_bitmap,
                                                ps, pred);
    }
    //---------------------------------------------------------------------------
    utils::stopCounters(perf_event, bp5_file, kTupleCount, kIterations, ps);
  }
  // Filtering using 8 bit Lanes with bitmask as output.
  std::ofstream bp6_file(std::string(kFilterFolder) + "8bitLaneFilterMask.csv");
  for (u32 ps = 1; ps < 9; ++ps) {
    //---------------------------------------------------------------------------
    auto compressed = std::make_unique<__m512i[]>(64);
    bitpacking::simd32::avx512::packfast(
        reinterpret_cast<const u32 *>(vec.data()), compressed.get(), ps);
    //---------------------------------------------------------------------------
    __m512i match_bitmap;
    // Register cpu counters
    PerfEvent perf_event;
    if (ps == 1)
      utils::printReport(perf_event, bp6_file);
    perf_event.startCounters();
    //---------------------------------------------------------------------------
    // Decompress
    for (u32 i = 0; i < kIterations; ++i) {
      bitpacking::simd32::avx512::filterfastmask(compressed.get(),
                                                 &match_bitmap, ps, pred);
    }
    //---------------------------------------------------------------------------
    utils::stopCounters(perf_event, bp6_file, kTupleCount, kIterations, ps);
  }
  // Filtering using 32 bit Lanes with bitmask as output.
  std::ofstream bp7_file(std::string(kFilterFolder) +
                         "32bitLaneFilterMask.csv");
  for (u32 ps = 1; ps < 9; ++ps) {
    //---------------------------------------------------------------------------
    auto compressed = std::make_unique<__m512i[]>(64);
    bitpacking::simd32::avx512::pack(reinterpret_cast<const u32 *>(vec.data()),
                                     compressed.get(), ps);
    //---------------------------------------------------------------------------
    __m512i match_bitmap;
    // Register cpu counters
    PerfEvent perf_event;
    if (ps == 1)
      utils::printReport(perf_event, bp7_file);
    perf_event.startCounters();
    //---------------------------------------------------------------------------
    // Decompress
    for (u32 i = 0; i < kIterations; ++i) {
      bitpacking::simd32::avx512::filtermask(compressed.get(), &match_bitmap,
                                             ps, pred);
    }
    //---------------------------------------------------------------------------
    utils::stopCounters(perf_event, bp7_file, kTupleCount, kIterations, ps);
  }
}

/// @brief Measures filtering speed for different bitpacking techniques.
/// Note: This benchmark uses the large vector, thus data is read from memory.
template <typename T> static void bpFilterBigBenchmark() {
  std::ofstream bp_file(std::string(kFilterFolder) + "FilterBig" +
                        std::to_string(sizeof(T) * 8) + "bit" + ".csv");
  //---------------------------------------------------------------------------
  for (u32 ps = 1; ps < 9; ++ps) {
    //---------------------------------------------------------------------------
    auto compressed = std::make_unique<__m512i[]>(kBigTupleCount / 16);
    u64 cvecs = kBigTupleCount / 512;
    for (u64 i = 0; i < cvecs; ++i) {
      bitpacking::simd32::avx512::packfast(
          reinterpret_cast<const u32 *>(bigVec.data() + i * 512),
          compressed.get() + i * ps, ps);
    }
    u64 pvecs = kBigTupleCount * ps / 8 / sizeof(__m512i);
    //---------------------------------------------------------------------------
    __m512i match_bitmap;
    // Register cpu counters
    PerfEvent perf_event;
    if (ps == 1)
      utils::printReport(perf_event, bp_file);
    perf_event.startCounters();
    //---------------------------------------------------------------------------
    // Decompress
    for (u32 i = 0; i < kBigIterations; ++i) {
      for (u32 j = 0; j < pvecs; j += ps) {
        bitpacking::simd32::avx512::filterfastmask(compressed.get() + j,
                                                   &match_bitmap, ps, pred);
      }
    }
    //---------------------------------------------------------------------------
    utils::stopCounters(perf_event, bp_file, kBigTupleCount, kBigIterations,
                        ps);
  }
}

/// @brief This function measures the single threaded memory bandwidth.
static void memoryBandwidth() {
  std::ofstream bp_file(std::string(kFilterFolder) + "MemoryBandwidth.csv");
  //---------------------------------------------------------------------------
  u64 intPerReg = sizeof(__m512i) / sizeof(INTEGER);
  u64 cloads = kBigTupleCount / intPerReg;
  volatile __m512i sink;
  //---------------------------------------------------------------------------
  // Measurement
  PerfEvent perf_event;
  utils::startCounters(perf_event, bp_file);
  //---------------------------------------------------------------------------
  for (u32 i = 0; i < kBigIterations; ++i) {
    for (u32 j = 0; j < cloads; ++j) {
      __m512i val = _mm512_loadu_si512(bigVec.data() + intPerReg * j);
      asm volatile("" : : "x"(val));
    }
  }
  //---------------------------------------------------------------------------
  utils::stopCounters(perf_event, bp_file, kBigTupleCount, kBigIterations);
}

static void decompressionBenchmarks() {
  std::filesystem::create_directories(kResultFolder);
  // decompressionMicroBenchmark<INTEGER, EncodingType::kUncompressed,
  //                             Data::kRandom>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks512,
  //                             Data::kRandom>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks128,
  //                             Data::kRandom>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks256,
  //                             Data::kRandom>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks512,
  //                             Data::kRandom>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks1,
  //                             Data::kRandom>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks3,
  //                             Data::kRandom>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks1_256,
  //                             Data::kRandom>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks3_256,
  //                             Data::kRandom>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kUncompressed,
  //                             Data::kOneValue>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks64,
  //                             Data::kOneValue>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks128,
  //                             Data::kOneValue>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks256,
  //                             Data::kOneValue>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks512,
  //                             Data::kOneValue>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks1,
  //                             Data::kOneValue>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks3,
  //                             Data::kOneValue>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks1_256,
  //                             Data::kOneValue>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks3_256,
  //                             Data::kOneValue>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kUncompressed,
  //                             Data::kMonotonic>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks64,
  //                             Data::kMonotonic>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks128,
  //                             Data::kMonotonic>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks256,
  //                             Data::kMonotonic>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kTinyBlocks512,
  //                             Data::kMonotonic>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks1,
  //                             Data::kMonotonic>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks3,
  //                             Data::kMonotonic>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks1_256,
  //                             Data::kMonotonic>();
  // decompressionMicroBenchmark<INTEGER, EncodingType::kBtrBlocks3_256,
  //                             Data::kMonotonic>();
};
static void filterBenchmarks() {
  std::filesystem::create_directories(kResultFolder);
  std::filesystem::create_directories(kFilterFolder);
  bpMicroBenchmark<INTEGER>();
  bpFilterBigBenchmark<INTEGER>();
  memoryBandwidth();
}
//---------------------------------------------------------------------------
} // namespace benchmarks