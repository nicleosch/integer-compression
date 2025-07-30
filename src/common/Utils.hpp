#pragma once
//---------------------------------------------------------------------------
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <ostream>
#include <sys/mman.h>
#include <unistd.h>
//---------------------------------------------------------------------------
#include "common/Types.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace utils {
//---------------------------------------------------------------------------
/// @brief Representation of a memory mapped file.
class MemoryMappedFile {
public:
  explicit MemoryMappedFile(const char *path) {
    fd = open(path, O_RDONLY);
    if (fd < 0) {
      std::cerr << "Memory mapping the file failed. The path does not exist."
                << std::endl;
      exit(1);
    }
    file_size = lseek(fd, 0, SEEK_END);
    data = static_cast<char *>(
        mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0));
    if (data == MAP_FAILED) {
      std::cerr << "Memory mapping the file failed." << std::endl;
      close(fd);
      exit(1);
    }
  }
  ~MemoryMappedFile() {
    munmap(data, file_size);
    close(fd);
  }
  MemoryMappedFile(const MemoryMappedFile &other) = delete;
  MemoryMappedFile &operator=(const MemoryMappedFile &other) = delete;

  const char *begin() const { return data; }
  const char *end() const { return data + file_size; }
  u32 size() const { return file_size; }

private:
  int fd;
  u32 file_size;
  char *data;
};
//---------------------------------------------------------------------------
/// @brief Jump, in given array, to the ith delimiter.
/// @param begin The start of the array.
/// @param end The end of the array.
/// @param delimiter The delimiter to jump to.
/// @param i The distance in delimiters to jump.
inline const char *jumpToIthDelimiter(const char *begin, const char *end,
                                      char delimiter, s32 i) {
  while (begin != end) {
    if (*begin == delimiter)
      --i;
    if (i <= 0)
      return begin;
    ++begin;
  }
  return begin;
}
//---------------------------------------------------------------------------
/// @brief Dump a buffer to an ostream.
/// @param data The data that should be dumped.
/// @param length The length of the data.
/// @param out The output stream.
/// @param width The line width.
inline void hex_dump(const std::byte *data, size_t length, std::ostream &out,
                     std::size_t width = 16) {
  const auto *begin = reinterpret_cast<const char *>(data);
  const auto *end = begin + length;
  size_t line_length = 0;
  for (const auto *line = begin; line != end; line += line_length) {
    out.width(4);
    out.fill('0');
    out << std::hex << line - begin << " : ";
    line_length = std::min(width, static_cast<std::size_t>(end - line));
    for (std::size_t pass = 1; pass <= 2; ++pass) {
      for (const char *next = line; next != end && next != line + width;
           ++next) {
        char ch = *next;
        switch (pass) {
        case 1:
          out << (ch < 32 ? '.' : ch);
          break;
        case 2:
          if (next != line) {
            out << " ";
          }
          out.width(2);
          out.fill('0');
          out << std::hex << std::uppercase
              << static_cast<int>(static_cast<unsigned char>(ch));
          break;
        }
      }
      if (pass == 1 && line_length != width) {
        out << std::string(width - line_length, ' ');
      }
      out << " ";
    }
    out << std::endl;
  }
}
//---------------------------------------------------------------------------
/// @brief A utility to time operations.
class Timer {
public:
  /// Constructor.
  Timer() = default;
  /// Start a time measurement.
  void start() { tstart = std::chrono::high_resolution_clock::now(); }
  /// End a time measurement.
  void end() { tend = std::chrono::high_resolution_clock::now(); }
  /// Get the duration in microseconds.
  double getMicroSeconds() {
    return std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart)
        .count();
  }
  /// Get the duration in milliseconds.
  double getMilliSeconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart)
        .count();
  }
  /// Get the duration in seconds.
  double getSeconds() {
    return std::chrono::duration<double>(tend - tstart).count();
  }

private:
  std::chrono::high_resolution_clock::time_point tstart;
  std::chrono::high_resolution_clock::time_point tend;
};
//---------------------------------------------------------------------------
/// @brief Template specialization to determine the number of required bits to
/// represent an integer.
template <typename T> inline u8 requiredBits(T value);
template <> inline u8 requiredBits(u32 value) {
  if (value == 0)
    return 1;
  return static_cast<u8>(sizeof(u32) * 8) - __builtin_clz(value);
}
template <> inline u8 requiredBits(u64 value) {
  if (value == 0)
    return 1;
  return static_cast<u8>(sizeof(u64) * 8) - __builtin_clzll(value);
}
//---------------------------------------------------------------------------
/// @brief A utility to safely load from an unaligned address.
template <typename T> inline T unalignedLoad(const void *ptr) {
  T value;
  std::memcpy(&value, ptr, sizeof(T));
  return value;
}
//---------------------------------------------------------------------------
/// @brief A utility to safely write to an unaligned address.
template <typename T> inline void unalignedStore(void *ptr, const T value) {
  std::memcpy(ptr, &value, sizeof(T));
}
//---------------------------------------------------------------------------
/// @brief Align ptr to specified alignment.
template <typename T> inline T *align(T *ptr, u64 alignment, u64 &padding) {
  uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned = (raw + alignment - 1) & ~(alignment - 1);
  padding = aligned - raw;
  return reinterpret_cast<T *>(aligned);
}
//---------------------------------------------------------------------------
/// @brief Reads a 1 GB block of data attempting to thrash CPU caches.
static void thrashCPUCaches() {
  const u64 size = 1000 * 1024 * 1024; // 1 GB
  vector<uint8_t> block(size);
  volatile u64 sink = 0;
  for (u64 i = 0; i < block.size(); ++i) {
    sink += block[i];
  }
}
//---------------------------------------------------------------------------
} // namespace utils
//---------------------------------------------------------------------------
} // namespace compression