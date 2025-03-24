#include "storage/Column.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
namespace storage {
//---------------------------------------------------------------------------
void Column::readFile(utils::MemoryMappedFile &file, u32 column_id,
                      char delimiter) {
  const char *iter = file.begin();
  const char *end = file.end();

  while (iter != end) {
    auto value_begin =
        utils::jumpToIthDelimiter(iter, end, delimiter, column_id);
    if (column_id > 0)
      ++value_begin;
    iter = utils::jumpToIthDelimiter(value_begin, end, delimiter, 1);

    INTEGER value;
    std::from_chars(value_begin, iter, value);
    raw_data.push_back(value);

    iter = utils::jumpToIthDelimiter(value_begin, end, '\n', 1);
    if(iter == end) return;
    ++iter;
  }
}
//---------------------------------------------------------------------------
void Column::padToMultipleOf(u16 length) {
  u16 rest = length - (raw_data.size() % length);
  for (u16 i = 0; i < rest; ++i) {
    raw_data.push_back(0);
  }
}
//---------------------------------------------------------------------------
INTEGER *Column::data() { return raw_data.data(); }
//---------------------------------------------------------------------------
u32 Column::size() { return raw_data.size(); }
//---------------------------------------------------------------------------
} // namespace storage
//---------------------------------------------------------------------------
} // namespace compression