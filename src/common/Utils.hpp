#pragma once
//---------------------------------------------------------------------------
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <ostream>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace utils {
//---------------------------------------------------------------------------
class MemoryMappedFile {
  public:
    explicit MemoryMappedFile(const char* path) {
      fd = open(path, O_RDONLY);
      if (fd < 0) exit(1);
      file_size = lseek(fd, 0, SEEK_END);
      data = static_cast<char*>(mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0));
      if (data == MAP_FAILED) {
        close(fd);
        exit(1);
      }
    }
    ~MemoryMappedFile() {
      munmap(data, file_size);
      close(fd);
    }
    MemoryMappedFile(const MemoryMappedFile& other) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile& other) = delete;
  
    const char* begin() const { return data; }
    const char* end() const { return data + file_size; }
    u32 size() const { return file_size; }
  
    private:
    int fd;
    u32 file_size;
    char* data;
};
//---------------------------------------------------------------------------
/// @brief Jump, in given array, to the ith delimiter.
/// @param begin The start of the array.
/// @param end The end of the array.
/// @param delimiter The delimiter to jump to.
/// @param i The distance in delimiters to jump.
inline const char* jumpToIthDelimiter(
  const char* begin, const char* end, char delimiter, s32 i)
{
  while(begin != end) {
    if(*begin == delimiter) --i;
    if(i <= 0) return begin;
    ++begin;
  }
  return begin;
}
//---------------------------------------------------------------------------
}  // utils
//---------------------------------------------------------------------------
}  // compression