# ----------------
# Helpers

def howmany(bit):
    """ how many values are we going to pack? """
    return 64

def howmanywords(bit):
    return int((howmany(bit) * bit + 127)/128)

def howmanybytes(bit):
    return howmanywords(bit) * 16

def plurial(number):
    if(number != 1):
        return "s"
    else :
        return ""

comp_types = ["eq", "neq", "gt", "lt"]

# ----------------
# Code Generation

print("#include <cstring>")
print("//---------------------------------------------------------------------------")
print("#include \"Block64BitPacking.hpp\"")
print("//---------------------------------------------------------------------------")
print("namespace compression {")
print("//---------------------------------------------------------------------------")
print("namespace bitpacking {")
print("//---------------------------------------------------------------------------")
print("namespace simd32 {")
print("//---------------------------------------------------------------------------")
print("namespace block64 {")
print("//---------------------------------------------------------------------------")
print("""typedef void (*simdpackblockfnc)(const u32 *pin, __m128i *compressed);""")
print("""typedef void (*simdunpackblockfnc)(const __m128i *compressed, u32 *out);""")

# Packing

print("")
print("static void simdpackblock0(const u32 *pin, __m128i *compressed) {")
print("  (void)compressed;")
print("  (void) pin; /* we consumed {0} 32-bit integer{1} */ ".format(howmany(0),plurial(howmany(0))))
print("}")
print("")

for bit in range(1,33): # 32 possible bit-sizes
    print("")
    print("/* we are going to pack {0} {1}-bit values, touching {2} 128-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void simdpackblock{0}(const u32 *pin, __m128i * compressed) {{".format(bit))
    print("  const __m128i * in = (const __m128i *)  pin;")
    print("  /* we are going to touch  {0} 128-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
      print("  __m128i w0;")
    else:
      print("  __m128i w0, w1;")
    if( (bit & (bit-1)) != 0) : print("  __m128i tmp; /* used to store inputs at word boundary */")
    oldword = 0
    for j in range(int(howmany(bit)/4)): # 16 loads (4 integers per load -> 4 * 16 = 64 -> 64 loaded integers in total)
      firstword = int(j * bit / 32)
      if(firstword > oldword): # overflow? -> store the fully occupied word
        print("  _mm_storeu_si128(compressed + {0}, w{1});".format(oldword,oldword%2))
        oldword = firstword
      secondword = int((j * bit + bit - 1)/32)
      firstshift = (j*bit) % 32 # how much to shift the bits to the left?
      if( firstword == secondword): # we can fit all bits into the first word
          if(firstshift == 0):
            print("  w{0} = _mm_loadu_si128 (in + {1});".format(firstword%2,j))
          else:
            print("  w{0} = _mm_or_si128(w{0},_mm_slli_epi32(_mm_loadu_si128 (in + {1}) , {2}));".format(firstword%2,j,firstshift))
      else: # bits need to be wrapped around words
          print("  tmp = _mm_loadu_si128 (in + {0});".format(j))
          print("  w{0} = _mm_or_si128(w{0},_mm_slli_epi32(tmp , {2}));".format(firstword%2,j,firstshift))
          secondshift = 32-firstshift # shift to write the carry into the new word
          print("  w{0} = _mm_srli_epi32(tmp,{2});".format(secondword%2,j,secondshift))
    if(firstword % 2 != secondword % 2):
        print("  _mm_storeu_si128(compressed + {0}, w{1});".format(oldword,oldword%2))
    print("  _mm_storeu_si128(compressed + {0}, w{1});".format(secondword,secondword%2))
    print("}")
    print("")

print("")
print("static void simdpackblockmask0(const u32 * pin, __m128i * compressed) {")
print("  (void)compressed;")
print("  (void) pin; /* we consumed {0} 32-bit integer{1} */ ".format(howmany(0),plurial(howmany(0))))
print("}")
print("")

# Packing (masking included)

for bit in range(1,33):
    print("")
    print("/* we are going to pack {0} {1}-bit values, touching {2} 128-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void simdpackblockmask{0}(const u32 * pin, __m128i * compressed) {{".format(bit))
    print("  /* we are going to touch  {0} 128-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
      print("  __m128i w0;")
    else:
      print("  __m128i w0, w1;")
    print("  const __m128i * in = (const __m128i *) pin;")
    if(bit < 32): print("  const __m128i mask = _mm_set1_epi32({0});".format((1<<bit)-1));
    def maskfnc(x):
        if(bit == 32): return x
        return " _mm_and_si128 ( mask, {0}) ".format(x)
    if( (bit & (bit-1)) != 0) : print("  __m128i tmp; /* used to store inputs at word boundary */")
    oldword = 0
    for j in range(int(howmany(bit)/4)):
      firstword = int(j * bit / 32)
      if(firstword > oldword):
        print("  _mm_storeu_si128(compressed + {0}, w{1});".format(oldword,oldword%2))
        oldword = firstword
      secondword = int((j * bit + bit - 1)/32)
      firstshift = (j*bit) % 32
      loadstr = maskfnc(" _mm_loadu_si128 (in + {0}) ".format(j))
      if( firstword == secondword):
          if(firstshift == 0):
            print("  w{0} = {1};".format(firstword%2,loadstr))
          else:
            print("  w{0} = _mm_or_si128(w{0},_mm_slli_epi32({1} , {2}));".format(firstword%2,loadstr,firstshift))
      else:
          print("  tmp = {0};".format(loadstr))
          print("  w{0} = _mm_or_si128(w{0},_mm_slli_epi32(tmp , {2}));".format(firstword%2,j,firstshift))
          secondshift = 32-firstshift
          print("  w{0} = _mm_srli_epi32(tmp,{2});".format(secondword%2,j,secondshift))
    if(firstword % 2 != secondword % 2):
        print("  _mm_storeu_si128(compressed + {0}, w{1});".format(oldword,oldword%2))
    print("  _mm_storeu_si128(compressed + {0}, w{1});".format(secondword,secondword%2))
    print("}")
    print("")

print("static void simdunpackblock0(const __m128i * compressed, u32 * pout) {")
print("  (void) compressed;")
print("  memset(pout,0,{0});".format(howmany(0)))
print("}")
print("")

# Unpacking

for bit in range(1,33):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 128-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void simdunpackblock{0}(const __m128i * compressed, u32 * pout) {{".format(bit))
    print("  /* we are going to access  {0} 128-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
      print("  __m128i w0;")
    else:
      print("  __m128i w0, w1;")
    print("  __m128i * out = (__m128i *) pout;")
    if(bit < 32): print("  const __m128i mask = _mm_set1_epi32({0});".format((1<<bit)-1));
    maskstr = " _mm_and_si128 ( mask, {0}) "
    if (bit == 32) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm_loadu_si128 (compressed);")
    for j in range(int(howmany(bit)/4)):
      firstword = int(j * bit / 32)
      secondword = int((j * bit + bit - 1)/32)
      if(secondword > oldword):
        print("  w{0} = _mm_loadu_si128 (compressed + {1});".format(secondword%2,secondword))
        oldword = secondword
      firstshift = (j*bit) % 32
      firstshiftstr = "_mm_srli_epi32( w{0} , "+str(firstshift)+") "
      if(firstshift == 0):
          firstshiftstr =" w{0} " # no need
      wfirst = firstshiftstr.format(firstword%2)
      if( firstword == secondword):
          if(firstshift + bit != 32):
            wfirst  = maskstr.format(wfirst)
          print("  _mm_storeu_si128(out + {0}, {1});".format(j,wfirst))
      else:
          secondshift = (32-firstshift)
          wsecond = "_mm_slli_epi32( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
          wfirstorsecond = " _mm_or_si128 ({0},{1}) ".format(wfirst,wsecond)
          wfirstorsecond = maskstr.format(wfirstorsecond)
          print("  _mm_storeu_si128(out + {0},\n    {1});".format(j,wfirstorsecond))
    print("}")
    print("")

print("static simdpackblockfnc simdfuncPackArr[] = {")
for bit in range(0,32):
  print("&simdpackblock{0},".format(bit))
print("&simdpackblock32")
print("};")

print("static simdpackblockfnc simdfuncPackMaskArr[] = {")
for bit in range(0,32):
  print("&simdpackblockmask{0},".format(bit))
print("&simdpackblockmask32")
print("};")

print("static simdunpackblockfnc simdfuncUnpackArr[] = {")
for bit in range(0,32):
  print("&simdunpackblock{0},".format(bit))
print("&simdunpackblock32")
print("};")
print("")

print("//---------------------------------------------------------------------------")
print("void pack(const u32 *in, __m128i *out, const u8 bit) {")
print("  simdfuncPackArr[bit](in, out);")
print("}")
print("//---------------------------------------------------------------------------")
print("void packmask(const u32 *in, __m128i *out, const u8 bit) {")
print("  simdfuncPackMaskArr[bit](in, out);")
print("}")
print("//---------------------------------------------------------------------------")
print("void unpack(const __m128i *in, u32 *out, const u8 bit) {")
print("  simdfuncUnpackArr[bit](in, out);")
print("}")
print("//---------------------------------------------------------------------------")


# Equality
print("static void filtereq0(const __m128i *in, u32 *matches, const INTEGER comp) {")
print("  if (comp == 0)")
print("     memset(matches, 1, 64 * sizeof(*matches));")
print("  else")
print("     memset(matches, 0, 64 * sizeof(*matches));")
print("}")
print("")

for bit in range(1,33):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 128-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void filtereq{0}(const __m128i *in, u32 *matches, const INTEGER comp) {{".format(bit))
    print("  /* we are going to access  {0} 128-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
        print("  __m128i w0;")
    else:
        print("  __m128i w0, w1;")
    print("  auto out = reinterpret_cast<__m128i *>(matches);")
    if(bit < 32): print("  const __m128i mask = _mm_set1_epi32({0});".format((1<<bit)-1))
    print("  const __m128i broadcomp = _mm_set1_epi32(comp);")
    maskstr = " _mm_and_si128 ( mask, {0}) "
    if (bit == 32) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm_lddqu_si128 (in);")
    for j in range(int(howmany(bit)/4)):
        firstword = int(j * bit / 32)
        secondword = int((j * bit + bit - 1)/32)
        if(secondword > oldword):
            print("  w{0} = _mm_lddqu_si128 (in + {1});".format(secondword%2,secondword))
            oldword = secondword
        firstshift = (j*bit) % 32
        firstshiftstr = "_mm_srli_epi32( w{0} , "+str(firstshift)+") "
        if(firstshift == 0):
            firstshiftstr =" w{0} " # no need
        wfirst = firstshiftstr.format(firstword%2)
        if( firstword == secondword):
            if(firstshift + bit != 32):
                wfirst  = maskstr.format(wfirst)
            print("  _mm_storeu_si128(out + {0}, _mm_cmpeq_epi32({1}, broadcomp));".format(j,wfirst))
        else:
            secondshift = (32-firstshift)
            wsecond = "_mm_slli_epi32( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
            wfirstorsecond = " _mm_or_si128 ({0},{1}) ".format(wfirst,wsecond)
            wfirstorsecond = maskstr.format(wfirstorsecond)
            print("  _mm_storeu_si128(out + {0},\n    _mm_cmpeq_epi32({1}, broadcomp));".format(j,wfirstorsecond))
    print("}")
    print("")

#Inequality
print("static void filterneq0(const __m128i *in, u32 *matches, const INTEGER comp) {")
print("  if (comp == 0)")
print("     memset(matches, 0, 64 * sizeof(*matches));")
print("  else")
print("     memset(matches, 1, 64 * sizeof(*matches));")
print("}")
print("")

for bit in range(1,33):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 128-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void filterneq{0}(const __m128i *in, u32 *matches, const INTEGER comp) {{".format(bit))
    print("  /* we are going to access  {0} 128-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
        print("  __m128i w0;")
    else:
        print("  __m128i w0, w1;")
    print("  auto out = reinterpret_cast<__m128i *>(matches);")
    if(bit < 32): print("  const __m128i mask = _mm_set1_epi32({0});".format((1<<bit)-1))
    print("  const __m128i broadcomp = _mm_set1_epi32(comp);")
    maskstr = " _mm_and_si128 ( mask, {0}) "
    if (bit == 32) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm_lddqu_si128 (in);")
    for j in range(int(howmany(bit)/4)):
        firstword = int(j * bit / 32)
        secondword = int((j * bit + bit - 1)/32)
        if(secondword > oldword):
            print("  w{0} = _mm_lddqu_si128 (in + {1});".format(secondword%2,secondword))
            oldword = secondword
        firstshift = (j*bit) % 32
        firstshiftstr = "_mm_srli_epi32( w{0} , "+str(firstshift)+") "
        if(firstshift == 0):
            firstshiftstr =" w{0} " # no need
        wfirst = firstshiftstr.format(firstword%2)
        if( firstword == secondword):
            if(firstshift + bit != 32):
                wfirst  = maskstr.format(wfirst)
            print("  _mm_storeu_si128(out + {0}, _mm_xor_si128(_mm_cmpeq_epi32({1}, broadcomp), _mm_set1_epi32(-1)));".format(j,wfirst))
        else:
            secondshift = (32-firstshift)
            wsecond = "_mm_slli_epi32( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
            wfirstorsecond = " _mm_or_si128 ({0},{1}) ".format(wfirst,wsecond)
            wfirstorsecond = maskstr.format(wfirstorsecond)
            print("  _mm_storeu_si128(out + {0},\n     _mm_xor_si128(_mm_cmpeq_epi32({1}, broadcomp), _mm_set1_epi32(-1)));".format(j,wfirstorsecond))
    print("}")
    print("")

# GreaterThan
print("static void filtergt0(const __m128i *in, u32 *matches, const INTEGER comp) {")
print("  if (comp < 0)")
print("     memset(matches, 1, 64 * sizeof(*matches));")
print("  else")
print("     memset(matches, 0, 64 * sizeof(*matches));")
print("}")
print("")

for bit in range(1,33):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 128-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void filtergt{0}(const __m128i *in, u32 *matches, const INTEGER comp) {{".format(bit))
    print("  /* we are going to access  {0} 128-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
        print("  __m128i w0;")
    else:
        print("  __m128i w0, w1;")
    print("  auto out = reinterpret_cast<__m128i *>(matches);")
    if(bit < 32): print("  const __m128i mask = _mm_set1_epi32({0});".format((1<<bit)-1))
    print("  const __m128i broadcomp = _mm_set1_epi32(comp);")
    maskstr = " _mm_and_si128 ( mask, {0}) "
    if (bit == 32) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm_lddqu_si128 (in);")
    for j in range(int(howmany(bit)/4)):
        firstword = int(j * bit / 32)
        secondword = int((j * bit + bit - 1)/32)
        if(secondword > oldword):
            print("  w{0} = _mm_lddqu_si128 (in + {1});".format(secondword%2,secondword))
            oldword = secondword
        firstshift = (j*bit) % 32
        firstshiftstr = "_mm_srli_epi32( w{0} , "+str(firstshift)+") "
        if(firstshift == 0):
            firstshiftstr =" w{0} " # no need
        wfirst = firstshiftstr.format(firstword%2)
        if( firstword == secondword):
            if(firstshift + bit != 32):
                wfirst  = maskstr.format(wfirst)
            print("  _mm_storeu_si128(out + {0}, _mm_cmpgt_epi32({1}, broadcomp));".format(j,wfirst))
        else:
            secondshift = (32-firstshift)
            wsecond = "_mm_slli_epi32( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
            wfirstorsecond = " _mm_or_si128 ({0},{1}) ".format(wfirst,wsecond)
            wfirstorsecond = maskstr.format(wfirstorsecond)
            print("  _mm_storeu_si128(out + {0},\n    _mm_cmpgt_epi32({1}, broadcomp));".format(j,wfirstorsecond))
    print("}")
    print("")

# LessThan
print("static void filterlt0(const __m128i *in, u32 *matches, const INTEGER comp) {")
print("  if (comp > 0)")
print("     memset(matches, 1, 64 * sizeof(*matches));")
print("  else")
print("     memset(matches, 0, 64 * sizeof(*matches));")
print("}")
print("")

for bit in range(1,33):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 128-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void filterlt{0}(const __m128i *in, u32 *matches, const INTEGER comp) {{".format(bit))
    print("  /* we are going to access  {0} 128-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
        print("  __m128i w0;")
    else:
        print("  __m128i w0, w1;")
    print("  auto out = reinterpret_cast<__m128i *>(matches);")
    if(bit < 32): print("  const __m128i mask = _mm_set1_epi32({0});".format((1<<bit)-1))
    print("  const __m128i broadcomp = _mm_set1_epi32(comp);")
    maskstr = " _mm_and_si128 ( mask, {0}) "
    if (bit == 32) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm_lddqu_si128 (in);")
    for j in range(int(howmany(bit)/4)):
        firstword = int(j * bit / 32)
        secondword = int((j * bit + bit - 1)/32)
        if(secondword > oldword):
            print("  w{0} = _mm_lddqu_si128 (in + {1});".format(secondword%2,secondword))
            oldword = secondword
        firstshift = (j*bit) % 32
        firstshiftstr = "_mm_srli_epi32( w{0} , "+str(firstshift)+") "
        if(firstshift == 0):
            firstshiftstr =" w{0} " # no need
        wfirst = firstshiftstr.format(firstword%2)
        if( firstword == secondword):
            if(firstshift + bit != 32):
                wfirst  = maskstr.format(wfirst)
            print("  _mm_storeu_si128(out + {0}, _mm_cmpgt_epi32(broadcomp, {1}));".format(j,wfirst))
        else:
            secondshift = (32-firstshift)
            wsecond = "_mm_slli_epi32( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
            wfirstorsecond = " _mm_or_si128 ({0},{1}) ".format(wfirst,wsecond)
            wfirstorsecond = maskstr.format(wfirstorsecond)
            print("  _mm_storeu_si128(out + {0},\n    _mm_cmpgt_epi32(broadcomp, {1}));".format(j,wfirstorsecond))
    print("}")
    print("")

for ct in comp_types:
    print("void filter{0}(const __m128i *in, u32 *matches, const INTEGER comp, const u8 bit) {{".format(ct))
    print(" switch(bit) {")
    for bit in range(0,33):
        print(" case {0}:".format(bit))
        print("     filter{0}{1}(in, matches, comp);".format(ct,bit))
        print("     break;")
    print(" }")
    print("}")
    print("")


print("void filter(const __m128i *in, u32 *matches, const u8 bit, const algebra::Predicate<INTEGER> &predicate) {")
print(" const INTEGER comp = predicate.getValue();")
print(" switch (predicate.getType()) {")
print(" case algebra::PredicateType::EQ:")
print("     filtereq(in, matches, comp, bit);")
print("     break;")
print(" case algebra::PredicateType::INEQ:")
print("     filterneq(in, matches, comp, bit);")
print("     break;")
print(" case algebra::PredicateType::GT:")
print("     filtergt(in, matches, comp, bit);")
print("     break;")
print(" case algebra::PredicateType::LT:")
print("     filterlt(in, matches, comp, bit);")
print("     break;")
print(" default:")
print("     break;")
print(" }")
print("}")


print("//---------------------------------------------------------------------------")
print("}")
print("//---------------------------------------------------------------------------")
print("}")
print("//---------------------------------------------------------------------------")
print("}")
print("//---------------------------------------------------------------------------")
print("}")
