#pragma once
//---------------------------------------------------------------------------
#include <charconv>
//---------------------------------------------------------------------------
#include "common/Types.hpp"
#include "common/Utils.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace storage {
//---------------------------------------------------------------------------
template <typename T> class Column {
public:
  //---------------------------------------------------------------------------
  Column() = delete;
  //---------------------------------------------------------------------------
  explicit Column(vector<T> data) : raw_data(std::move(data)) {}
  //---------------------------------------------------------------------------
  /// @brief Reads a column from a file.
  /// @param path The path to the file.
  /// @param column_id The id to the column to be read.
  /// @param delimiter The delimiter of columns in the file.
  /// @return An in-memory representation of the read column.
  static Column fromFile(const char *path, u32 column_id, char delimiter) {
    auto file = utils::MemoryMappedFile(path);
    Column column(std::vector<T>{});
    column.readFile(file, column_id, delimiter);
    return column;
  }
  //---------------------------------------------------------------------------
  void padToMultipleOf(u32 length) {
    u16 rest = length - (raw_data.size() % length);
    for (u16 i = 0; i < rest; ++i) {
      raw_data.push_back(raw_data[raw_data.size() - 1]);
    }
  }
  //---------------------------------------------------------------------------
  T *data() { return raw_data.data(); }
  //---------------------------------------------------------------------------
  u32 size() { return raw_data.size(); }

private:
  void readFile(utils::MemoryMappedFile &file, u32 column_id, char delimiter) {
    const char *iter = file.begin();
    const char *end = file.end();

    while (iter != end) {
      auto value_begin =
          utils::jumpToIthDelimiter(iter, end, delimiter, column_id);
      if (column_id > 0)
        ++value_begin;
      iter = utils::jumpToIthDelimiter(value_begin, end, delimiter, 1);

      T value;
      std::from_chars(value_begin, iter, value);
      raw_data.push_back(value);

      iter = utils::jumpToIthDelimiter(value_begin, end, '\n', 1);
      if (iter == end)
        return;
      ++iter;
    }
  }
  //---------------------------------------------------------------------------
  vector<T> raw_data;
};
//---------------------------------------------------------------------------
} // namespace storage
//---------------------------------------------------------------------------
} // namespace compression
