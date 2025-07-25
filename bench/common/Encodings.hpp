#include "common/Types.hpp"
#include "common/Utils.hpp"
//---------------------------------------------------------------------------
#include "extern/BtrBlocks.hpp"
#include "schemes/Uncompressed.hpp"
#include "tinyblocks/TinyBlocks.hpp"
//---------------------------------------------------------------------------
using namespace compression;
//---------------------------------------------------------------------------
namespace benchmarks {
//---------------------------------------------------------------------------
enum class EncodingType {
  kUncompressed,
  kTinyBlocks64,
  kTinyBlocks128,
  kTinyBlocks256,
  kTinyBlocks512,
  kBtrBlocks1,
  kBtrBlocks3,
  kBtrBlocksReduced1,
  kBtrBlocksReduced3,
  kBtrBlocks1_256,
  kBtrBlocks3_256
};
inline const char *toString(EncodingType type) {
  switch (type) {
  case EncodingType::kUncompressed:
    return "Uncompressed";
  case EncodingType::kTinyBlocks64:
    return "TinyBlocks64";
  case EncodingType::kTinyBlocks128:
    return "TinyBlocks128";
  case EncodingType::kTinyBlocks256:
    return "TinyBlocks256";
  case EncodingType::kTinyBlocks512:
    return "TinyBlocks512";
  case EncodingType::kBtrBlocks1:
    return "BtrBlocks1";
  case EncodingType::kBtrBlocks3:
    return "BtrBlocks3";
  case EncodingType::kBtrBlocksReduced1:
    return "BtrBlocksReduced1";
  case EncodingType::kBtrBlocksReduced3:
    return "BtrBlocksReduced3";
  case EncodingType::kBtrBlocks1_256:
    return "BtrBlocks1_256";
  case EncodingType::kBtrBlocks3_256:
    return "BtrBlocks3_256";
  default:
    return "Unknown";
  }
}
//---------------------------------------------------------------------------
template <typename T> class Encoding {
public:
  virtual ~Encoding() = default;
  //---------------------------------------------------------------------------
  virtual u32 compress(const vector<T> &src, u8 *dest,
                       const void *tscheme = nullptr) = 0;
  //---------------------------------------------------------------------------
  virtual void decompress(const u8 *src, vector<T> &dest) = 0;
  //---------------------------------------------------------------------------
  virtual void filter(const u8 *data, const u32 size,
                      algebra::Predicate<T> &predicate, MatchVector &mv) = 0;
  //---------------------------------------------------------------------------
  virtual EncodingType getType() const = 0;
};
//---------------------------------------------------------------------------
template <typename T> class UncompressedEncoding : public Encoding<T> {
public:
  u32 compress(const vector<T> &src, u8 *dest,
               const void *tscheme = nullptr) override {
    return this->scheme.compress(src.data(), src.size(), dest, nullptr)
        .payload_size;
  }
  //---------------------------------------------------------------------------
  void decompress(const u8 *src, vector<T> &dest) override {
    this->scheme.decompress(dest.data(), dest.size(), src);
  }
  //---------------------------------------------------------------------------
  void filter(const u8 *data, const u32 size, algebra::Predicate<T> &predicate,
              MatchVector &mv) {
    assert(false);
  }
  //---------------------------------------------------------------------------
  EncodingType getType() const override { return EncodingType::kUncompressed; }

private:
  Uncompressed<T> scheme;
};
//---------------------------------------------------------------------------
template <typename T, u32 kTinyBlocksSize>
class TinyBlocksEncoding : public Encoding<T> {
public:
  u32 compress(const vector<T> &src, u8 *dest,
               const void *tscheme = nullptr) override {
    vector<Statistics<T>> stats;
    auto block_count = src.size() / kTinyBlocksSize;
    for (size_t i = 0; i < block_count; ++i) {
      stats.push_back(Statistics<T>::generateFrom(
          src.data() + i * kTinyBlocksSize, kTinyBlocksSize));
    }
    return this->scheme
        .compress(src.data(), src.size(), dest, stats.data(),
                  *reinterpret_cast<const tinyblocks::Scheme *>(tscheme))
        .payload_size;
  }
  //---------------------------------------------------------------------------
  void decompress(const u8 *src, vector<T> &dest) override {
    this->scheme.decompress(dest.data(), dest.size(), src);
  }
  //---------------------------------------------------------------------------
  void filter(const u8 *data, const u32 size, algebra::Predicate<T> &predicate,
              MatchVector &mv) {
    this->scheme.filter(data, size, predicate, mv.data());
  }
  //---------------------------------------------------------------------------
  EncodingType getType() const override {
    if constexpr (kTinyBlocksSize == 64)
      return EncodingType::kTinyBlocks64;
    else if constexpr (kTinyBlocksSize == 128)
      return EncodingType::kTinyBlocks128;
    else if constexpr (kTinyBlocksSize == 256)
      return EncodingType::kTinyBlocks256;
    else
      return EncodingType::kTinyBlocks512;
  }

private:
  tinyblocks::TinyBlocks<T, kTinyBlocksSize> scheme;
};
//---------------------------------------------------------------------------
template <typename T, u32 kDepth, u32 kBlockSize = 65536>
class BtrBlocksEncoding : public Encoding<T> {
public:
  u32 compress(const vector<T> &src, u8 *dest,
               const void *tscheme = nullptr) override {
    // Setup BtrBlocks
    btrblocks::BtrBlocksConfig::configure(
        [&](btrblocks::BtrBlocksConfig &config) {
          config.integers.max_cascade_depth = kDepth;
          if (tscheme == nullptr)
            config.integers.schemes.enableAll();
          else
            config.integers.override_scheme =
                *reinterpret_cast<const btrblocks::IntegerSchemeType *>(
                    tscheme);
          config.block_size = kBlockSize;
        });
    // Setup Data
    btrblocks::Vector<INTEGER> vec(src.size());
    for (auto i = 0; i < vec.count; ++i) {
      vec[i] = src.data()[i];
    }
    // Compress in Datablock-Granularity.
    this->relation.addColumn({"ints", std::move(vec)});
    this->datablock = std::make_unique<btrblocks::Datablock>(this->relation);
    vector<btrblocks::Range> ranges =
        this->relation.getRanges(btrblocks::SplitStrategy::SEQUENTIAL, 9999999);
    this->block_count = ranges.size();
    this->compressed_chunks.resize(this->block_count);
    for (u32 chunk_i = 0; chunk_i < this->block_count; chunk_i++) {
      auto chunk = this->relation.getChunk(ranges, chunk_i);
      this->datablock->compress(chunk, this->compressed_chunks[chunk_i]);
    }
    // TODO: Adjust return value to compression size
    return 0;
  }
  //---------------------------------------------------------------------------
  void decompress(const u8 *src, vector<T> &dest) override {
    for (u32 chunk_i = 0; chunk_i < this->block_count; chunk_i++) {
      btrblocks::Chunk decompressed =
          this->datablock->decompress(this->compressed_chunks[chunk_i]);
    }
  }
  //---------------------------------------------------------------------------
  void filter(const u8 *data, const u32 size, algebra::Predicate<T> &predicate,
              MatchVector &mv) {
    assert(false);
  }
  //---------------------------------------------------------------------------
  EncodingType getType() const override {
    if constexpr (kDepth == 1) {
      if constexpr (kBlockSize == 65536)
        return EncodingType::kBtrBlocks1;
      else
        return EncodingType::kBtrBlocks1_256;
    } else {
      if constexpr (kBlockSize == 65536)
        return EncodingType::kBtrBlocks3;
      else
        return EncodingType::kBtrBlocks3_256;
    }
  }

private:
  btrblocks::Relation relation;
  std::unique_ptr<btrblocks::Datablock> datablock;
  vector<btrblocks::BytesArray> compressed_chunks;
  u32 block_count = 0;
};
//---------------------------------------------------------------------------
} // namespace benchmarks