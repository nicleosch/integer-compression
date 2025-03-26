#pragma once
//---------------------------------------------------------------------------
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <ostream>
#include <sys/mman.h>
#include <unistd.h>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace utils {
//---------------------------------------------------------------------------
class MemoryMappedFile {
public:
  explicit MemoryMappedFile(const char *path) {
    fd = open(path, O_RDONLY);
    if (fd < 0)
      exit(1);
    file_size = lseek(fd, 0, SEEK_END);
    data = static_cast<char *>(
        mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0));
    if (data == MAP_FAILED) {
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
class Timer {
public:
  explicit Timer() : start(std::chrono::high_resolution_clock::now()) {}
  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time: " << duration << std::endl;
  }

private:
  std::chrono::high_resolution_clock::time_point start;
};
//---------------------------------------------------------------------------
} // namespace utils
//---------------------------------------------------------------------------
} // namespace compression