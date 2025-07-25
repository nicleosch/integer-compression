#include "cases/MicroBenchmarks.cpp"
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  benchmarks::filterBenchmarks();
  benchmarks::decompressionBenchmarks();
}
// ---------------------------------------------------------------------------
