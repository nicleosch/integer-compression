#include <fstream>
#include <random>
//---------------------------------------------------------------------------
#include "common/PerfEvent.hpp"
#include "common/Types.hpp"
#include "tinyblocks/TinyBlocks.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
namespace benchmarks {
//---------------------------------------------------------------------------
static const u32 tuple_count = 1024;
static const u32 iterations = 1024 * 1024;
static std::ofstream null_stream("/dev/null");
//---------------------------------------------------------------------------
template <typename T, u16 kTinyBlocksSize> static void bpBenchmark() {
  //---------------------------------------------------------------------------
  using TinyBlocks = tinyblocks::TinyBlocks<T, kTinyBlocksSize>;
  //---------------------------------------------------------------------------
  // The file to write to.
  std::ofstream bp_file("bp_" + std::to_string(sizeof(T) * 8) + "bit_" +
                        std::to_string(kTinyBlocksSize) + ".csv");
  //---------------------------------------------------------------------------
  // Create a column of 1024 integers and fill randomly with 0s and 1s
  vector<T> column(1024);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution dist(0.5);
  for (auto &i : column) {
    i = dist(gen);
  }
  //---------------------------------------------------------------------------
  // Buffer to compress into
  auto cbuffer = std::make_unique<u8[]>(column.size() * sizeof(T) * 2);
  //---------------------------------------------------------------------------
  // Calculate statistics on the data used for compression.
  vector<Statistics<T>> stats;
  auto block_count = column.size() / kTinyBlocksSize;
  for (size_t i = 0; i < block_count; ++i) {
    stats.push_back(Statistics<T>::generateFrom(
        column.data() + i * kTinyBlocksSize, kTinyBlocksSize));
  }
  //---------------------------------------------------------------------------
  TinyBlocks tb;
  const u8 bits = sizeof(T) * 8;
  for (u8 pack_size = bits - 31; pack_size <= bits; ++pack_size) {
    // Compress
    tb.compress(column.data(), column.size(), cbuffer.get(), stats.data(),
                tinyblocks::Scheme::FOR);
    //---------------------------------------------------------------------------
    // Register cpu counters
    PerfEvent perf_event;
    if (pack_size == bits - 31) {
      perf_event.printReport(bp_file, null_stream, 1);
      bp_file << "Cycles/Tuple, Instructions/Tuple\n";
    }
    perf_event.startCounters();
    //---------------------------------------------------------------------------
    // Decompress
    for (u32 i = 0; i < iterations; ++i) {
      tb.decompress(column.data(), column.size(), cbuffer.get());
    }
    //---------------------------------------------------------------------------
    // Stop cpu counters
    perf_event.stopCounters();
    perf_event.printReport(null_stream, bp_file, 1);
    bp_file << ", "
            << static_cast<double>(perf_event.getCounter("cycles")) /
                   (tuple_count * iterations)
            << ", "
            << static_cast<double>(perf_event.getCounter("instructions")) /
                   (tuple_count * iterations)
            << "\n";
  }
  //---------------------------------------------------------------------------
}
//---------------------------------------------------------------------------
static void decompressionBenchmarks() {
  //---------------------------------------------------------------------------
  bpBenchmark<INTEGER, 64>();
  bpBenchmark<INTEGER, 128>();
  bpBenchmark<INTEGER, 256>();
  bpBenchmark<INTEGER, 512>();
  bpBenchmark<BIGINT, 64>();
  bpBenchmark<BIGINT, 128>();
  bpBenchmark<BIGINT, 256>();
  bpBenchmark<BIGINT, 512>();
  //---------------------------------------------------------------------------
};
//---------------------------------------------------------------------------
} // namespace benchmarks