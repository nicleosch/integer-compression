#include <filesystem>
#include <fstream>
#include <random>
//---------------------------------------------------------------------------
#include "common/Encodings.hpp"
#include "common/PerfEvent.hpp"
#include "common/Types.hpp"
#include "extern/BtrBlocks.hpp"
#include "schemes/Uncompressed.hpp"
#include "tinyblocks/TinyBlocks.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
namespace benchmarks {
//---------------------------------------------------------------------------
/// @brief Constants required for the benchmarks.
static const u32 kTupleCount = 1024;
static const u32 kIterations = 1024 * 1024;
static std::ofstream null_stream("/dev/null");
static const char *kResultFolder = "results/";
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
/// @brief Starts cpu counters and writes the report header to the file.
static void startCounters(PerfEvent &event, std::ostream &file) {
  event.printReport(file, null_stream, 1);
  file << ", Cycles/Tuple, Instructions/Tuple\n";
  event.startCounters();
}
//---------------------------------------------------------------------------
/// @brief Stops CPU counters and writes the report payload to the file.
static void stopCounters(PerfEvent &event, std::ostream &file) {
  event.stopCounters();
  event.printReport(null_stream, file, 1);
  file << ", "
       << static_cast<double>(event.getCounter("cycles")) /
              (kTupleCount * kIterations)
       << ", "
       << static_cast<double>(event.getCounter("instructions")) /
              (kTupleCount * kIterations)
       << "\n";
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
  for (u32 i = 0; i < count; ++i) {
    if (i % kTinyBlocksSize == 0) {
      if (pack_size == 32)
        data[i] = -1;
      else
        data[i] = static_cast<T>(1ULL << pack_size);
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

template <typename T, EncodingType kType, Data kData>
static void decompressionSpeed() {
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

static void decompressionBenchmarks() {
  std::filesystem::create_directories(kResultFolder);
  decompressionSpeed<INTEGER, EncodingType::kUncompressed, Data::kRandom>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks64, Data::kRandom>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks128, Data::kRandom>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks256, Data::kRandom>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks512, Data::kRandom>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks1, Data::kRandom>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks3, Data::kRandom>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks1_256, Data::kRandom>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks3_256, Data::kRandom>();
  decompressionSpeed<INTEGER, EncodingType::kUncompressed, Data::kOneValue>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks64, Data::kOneValue>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks128, Data::kOneValue>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks256, Data::kOneValue>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks512, Data::kOneValue>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks1, Data::kOneValue>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks3, Data::kOneValue>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks1_256, Data::kOneValue>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks3_256, Data::kOneValue>();
  decompressionSpeed<INTEGER, EncodingType::kUncompressed, Data::kMonotonic>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks64, Data::kMonotonic>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks128, Data::kMonotonic>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks256, Data::kMonotonic>();
  decompressionSpeed<INTEGER, EncodingType::kTinyBlocks512, Data::kMonotonic>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks1, Data::kMonotonic>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks3, Data::kMonotonic>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks1_256,
                     Data::kMonotonic>();
  decompressionSpeed<INTEGER, EncodingType::kBtrBlocks3_256,
                     Data::kMonotonic>();
};
//---------------------------------------------------------------------------
} // namespace benchmarks