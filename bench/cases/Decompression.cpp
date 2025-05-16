#include <random>
//---------------------------------------------------------------------------
#include "common/PerfEvent.hpp"
#include "common/Types.hpp"
#include "schemes/TinyBlocks.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
namespace benchmarks {
//---------------------------------------------------------------------------
static void cycleBenchmark() {
  //---------------------------------------------------------------------------
  u16 iterations = 1024;
  const u16 kTinyBlocksSize = 512;
  using TinyBlocks = TinyBlocks<INTEGER, kTinyBlocksSize>;
  //---------------------------------------------------------------------------
  // Create a column of 1024 integers and fill randomly with 0s and 1s
  vector<INTEGER> column(1024);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution dist(0.5);
  for (auto &i : column) {
    i = dist(gen);
  }
  //---------------------------------------------------------------------------
  // Buffer to compress into
  auto cbuffer = std::make_unique<u8[]>(column.size() * sizeof(INTEGER) * 2);
  //---------------------------------------------------------------------------
  // Calculate statistics on the data used for compression.
  vector<Statistics<INTEGER>> stats;
  auto block_count = column.size() / kTinyBlocksSize;
  for (size_t i = 0; i < block_count; ++i) {
    stats.push_back(Statistics<INTEGER>::generateFrom(
        column.data() + i * kTinyBlocksSize, kTinyBlocksSize));
  }
  //---------------------------------------------------------------------------
  TinyBlocks tb;
  for (u8 pack_size = 1; pack_size <= 32; ++pack_size) {
    // Compress
    const TinyBlocks::Opcode opcode{TinyBlocks::Scheme::BIT_PACKING, pack_size};
    tb.compress(column.data(), column.size(), cbuffer.get(), stats.data(),
                opcode);
    //---------------------------------------------------------------------------
    // Register cpu counters
    PerfEvent perf_event;
    perf_event.startCounters();
    //---------------------------------------------------------------------------
    // Decompress
    for (u16 i = 0; i < iterations; ++i) {
      tb.decompress(column.data(), column.size(), cbuffer.get());
    }
    //---------------------------------------------------------------------------
    // Output cpu counters
    perf_event.stopCounters();
    std::cout << "Pack Size: " << static_cast<u32>(pack_size) << std::endl;
    u32 tuples = column.size() * iterations;
    u32 cycles = perf_event.getCounter("cycles");
    std::cout << "Tuples: " << tuples << std::endl;
    std::cout << "Cycles: " << cycles << std::endl;
    std::cout << "Cycles/Tuple: " << static_cast<double>(cycles) / tuples
              << std::endl;
    perf_event.printReport(std::cout, 1);
    std::cout << std::endl;
  }
  //---------------------------------------------------------------------------
}
//---------------------------------------------------------------------------
} // namespace benchmarks