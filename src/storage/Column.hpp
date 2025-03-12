#pragma once
//---------------------------------------------------------------------------
#include <charconv>
//---------------------------------------------------------------------------
#include "common/Units.hpp"
#include "common/Utils.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace storage {
//---------------------------------------------------------------------------
class Column {
  public:
    //---------------------------------------------------------------------------
    Column() = delete;
    //---------------------------------------------------------------------------
    explicit Column(vector<INTEGER> data) : raw_data(std::move(data)) {}
    //---------------------------------------------------------------------------
    /// @brief Reads a column from a file.
    /// @param path The path to the file.
    /// @param column_id The id to the column to be read.
    /// @param type The type to the column to be read.
    /// @param delimiter The delimiter of columns in the file.
    /// @return A column.
    static Column fromFile(const char* path, u32 column_id, char delimiter) {
      auto file = utils::MemoryMappedFile(path);
      Column column(std::vector<INTEGER>{});
      column.readFile(file, column_id, delimiter);
      return column;
    }
    //---------------------------------------------------------------------------
    INTEGER* data();
    //---------------------------------------------------------------------------
    u32 size();

  private:
    void readFile(utils::MemoryMappedFile& file, u32 column_id, char delimiter);
    //---------------------------------------------------------------------------
    vector<INTEGER> raw_data;
};
//---------------------------------------------------------------------------
}  // namespace storage
//---------------------------------------------------------------------------
}  // namespace compression
