#include "compression/Decompressor.hpp"
#include "schemes/FOR.hpp"
#include "schemes/FORn.hpp"
//---------------------------------------------------------------------------
namespace compression {
//---------------------------------------------------------------------------
void decompressor::frameOfReference(vector<INTEGER>& dest, u8* src, u32 total_size) {
  // allocate space
  dest.reserve(total_size);

  // compress
  FOR cFor;
  cFor.decompress(dest.data(), src, total_size, 0);
};
//---------------------------------------------------------------------------
void decompressor::frameOfReferenceN(vector<INTEGER>& dest, u8* src, u32 total_size, u32 block_size) {
  // allocate space
  dest.reserve(total_size);

  // compress
  FORn cForN;
  cForN.decompress(dest.data(), src, total_size, block_size);
}
//---------------------------------------------------------------------------
}  // compression