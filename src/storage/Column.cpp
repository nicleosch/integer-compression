#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace storage {
//---------------------------------------------------------------------------
void Column::readFile(utils::MemoryMappedFile& file, u32 column_id, char delimiter) {
  const char* iter = file.begin();
  const char* end = file.end();

  while(iter != end) {
    auto value_begin = utils::jumpToIthDelimiter(iter, end, delimiter, column_id);
    iter = utils::jumpToIthDelimiter(value_begin, end, delimiter, 1);
    
    INTEGER value;
    std::from_chars(value_begin + 1, iter, value);
    raw_data.push_back(value);

    iter = utils::jumpToIthDelimiter(value_begin, end, '\n', 1);
    ++iter;
  }
}
//---------------------------------------------------------------------------
INTEGER* Column::data() {
  return raw_data.data();
}
//---------------------------------------------------------------------------
u32 Column::size() {
  return raw_data.size();
}
//---------------------------------------------------------------------------
}  // namespace storage
//---------------------------------------------------------------------------
}  // namespace compression