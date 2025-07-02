# ----------------
# Helpers

def howmany(bit):
    """ how many values are we going to pack? """
    return 256

def howmanywords(bit):
    return int((howmany(bit) * bit + 255)/256)

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

print("#include <simdcomp.h>")
print("#include <cstring>")
print("//---------------------------------------------------------------------------")
print("#include \"AVXBitPacking.hpp\"")
print("//---------------------------------------------------------------------------")
print("namespace compression {")
print("//---------------------------------------------------------------------------")
print("namespace bitpacking {")
print("//---------------------------------------------------------------------------")
print("namespace simd32 {")
print("//---------------------------------------------------------------------------")
print("namespace avx {")
print("//---------------------------------------------------------------------------")
print("")
print("void pack(const u32 *in, __m256i *out, const u8 bit) {")
print(" return avxpackwithoutmask(in, out, bit);")
print("}")
print("")
print("void packmask(const u32 *in, __m256i *out, const u8 bit) {")
print(" return avxpack(in, out, bit);")
print("}")
print("")
print("void unpack(const __m256i *in, u32 *out, const u8 bit) {")
print(" avxunpack(in, out, bit);")
print("}")
print("")

# Equality
print("static void filtereq0(const __m256i *in, u32 *matches, const INTEGER comp) {")
print("  if (comp == 0)")
print("     memset(matches, 1, 256 * sizeof(*matches));")
print("  else")
print("     memset(matches, 0, 256 * sizeof(*matches));")
print("}")
print("")

for bit in range(1,33):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 256-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void filtereq{0}(const __m256i *in, u32 *matches, const INTEGER comp) {{".format(bit))
    print("  /* we are going to access  {0} 256-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
        print("  __m256i w0;")
    else:
        print("  __m256i w0, w1;")
    print("  auto out = reinterpret_cast<__m256i *>(matches);")
    if(bit < 32): print("  const __m256i mask = _mm256_set1_epi32({0});".format((1<<bit)-1))
    print("  const __m256i broadcomp = _mm256_set1_epi32(comp);")
    maskstr = " _mm256_and_si256 ( mask, {0}) "
    if (bit == 32) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm256_lddqu_si256 (in);")
    for j in range(int(howmany(bit)/8)):
        firstword = int(j * bit / 32)
        secondword = int((j * bit + bit - 1)/32)
        if(secondword > oldword):
            print("  w{0} = _mm256_lddqu_si256 (in + {1});".format(secondword%2,secondword))
            oldword = secondword
        firstshift = (j*bit) % 32
        firstshiftstr = "_mm256_srli_epi32( w{0} , "+str(firstshift)+") "
        if(firstshift == 0):
            firstshiftstr =" w{0} " # no need
        wfirst = firstshiftstr.format(firstword%2)
        if( firstword == secondword):
            if(firstshift + bit != 32):
                wfirst  = maskstr.format(wfirst)
            print("  _mm256_storeu_si256(out + {0}, _mm256_cmpeq_epi32({1}, broadcomp));".format(j,wfirst))
        else:
            secondshift = (32-firstshift)
            wsecond = "_mm256_slli_epi32( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
            wfirstorsecond = " _mm256_or_si256 ({0},{1}) ".format(wfirst,wsecond)
            wfirstorsecond = maskstr.format(wfirstorsecond)
            print("  _mm256_storeu_si256(out + {0},\n    _mm256_cmpeq_epi32({1}, broadcomp));".format(j,wfirstorsecond))
    print("}")
    print("")

#Inequality
print("static void filterneq0(const __m256i *in, u32 *matches, const INTEGER comp) {")
print("  if (comp == 0)")
print("     memset(matches, 0, 256 * sizeof(*matches));")
print("  else")
print("     memset(matches, 1, 256 * sizeof(*matches));")
print("}")
print("")

for bit in range(1,33):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 256-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void filterneq{0}(const __m256i *in, u32 *matches, const INTEGER comp) {{".format(bit))
    print("  /* we are going to access  {0} 256-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
        print("  __m256i w0;")
    else:
        print("  __m256i w0, w1;")
    print("  auto out = reinterpret_cast<__m256i *>(matches);")
    if(bit < 32): print("  const __m256i mask = _mm256_set1_epi32({0});".format((1<<bit)-1))
    print("  const __m256i broadcomp = _mm256_set1_epi32(comp);")
    maskstr = " _mm256_and_si256 ( mask, {0}) "
    if (bit == 32) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm256_lddqu_si256 (in);")
    for j in range(int(howmany(bit)/8)):
        firstword = int(j * bit / 32)
        secondword = int((j * bit + bit - 1)/32)
        if(secondword > oldword):
            print("  w{0} = _mm256_lddqu_si256 (in + {1});".format(secondword%2,secondword))
            oldword = secondword
        firstshift = (j*bit) % 32
        firstshiftstr = "_mm256_srli_epi32( w{0} , "+str(firstshift)+") "
        if(firstshift == 0):
            firstshiftstr =" w{0} " # no need
        wfirst = firstshiftstr.format(firstword%2)
        if( firstword == secondword):
            if(firstshift + bit != 32):
                wfirst  = maskstr.format(wfirst)
            print("  _mm256_storeu_si256(out + {0}, _mm256_xor_si256(_mm256_cmpeq_epi32({1}, broadcomp), _mm256_set1_epi32(-1)));".format(j,wfirst))
        else:
            secondshift = (32-firstshift)
            wsecond = "_mm256_slli_epi32( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
            wfirstorsecond = " _mm256_or_si256 ({0},{1}) ".format(wfirst,wsecond)
            wfirstorsecond = maskstr.format(wfirstorsecond)
            print("  _mm256_storeu_si256(out + {0},\n     _mm256_xor_si256(_mm256_cmpeq_epi32({1}, broadcomp), _mm256_set1_epi32(-1)));".format(j,wfirstorsecond))
    print("}")
    print("")

# GreaterThan
print("static void filtergt0(const __m256i *in, u32 *matches, const INTEGER comp) {")
print("  if (comp == 0)")
print("     memset(matches, 1, 256 * sizeof(*matches));")
print("  else")
print("     memset(matches, 0, 256 * sizeof(*matches));")
print("}")
print("")

for bit in range(1,33):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 256-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void filtergt{0}(const __m256i *in, u32 *matches, const INTEGER comp) {{".format(bit))
    print("  /* we are going to access  {0} 256-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
        print("  __m256i w0;")
    else:
        print("  __m256i w0, w1;")
    print("  auto out = reinterpret_cast<__m256i *>(matches);")
    if(bit < 32): print("  const __m256i mask = _mm256_set1_epi32({0});".format((1<<bit)-1))
    print("  const __m256i broadcomp = _mm256_set1_epi32(comp);")
    maskstr = " _mm256_and_si256 ( mask, {0}) "
    if (bit == 32) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm256_lddqu_si256 (in);")
    for j in range(int(howmany(bit)/8)):
        firstword = int(j * bit / 32)
        secondword = int((j * bit + bit - 1)/32)
        if(secondword > oldword):
            print("  w{0} = _mm256_lddqu_si256 (in + {1});".format(secondword%2,secondword))
            oldword = secondword
        firstshift = (j*bit) % 32
        firstshiftstr = "_mm256_srli_epi32( w{0} , "+str(firstshift)+") "
        if(firstshift == 0):
            firstshiftstr =" w{0} " # no need
        wfirst = firstshiftstr.format(firstword%2)
        if( firstword == secondword):
            if(firstshift + bit != 32):
                wfirst  = maskstr.format(wfirst)
            print("  _mm256_storeu_si256(out + {0}, _mm256_cmpgt_epi32({1}, broadcomp));".format(j,wfirst))
        else:
            secondshift = (32-firstshift)
            wsecond = "_mm256_slli_epi32( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
            wfirstorsecond = " _mm256_or_si256 ({0},{1}) ".format(wfirst,wsecond)
            wfirstorsecond = maskstr.format(wfirstorsecond)
            print("  _mm256_storeu_si256(out + {0},\n    _mm256_cmpgt_epi32({1}, broadcomp));".format(j,wfirstorsecond))
    print("}")
    print("")

# LessThan
print("static void filterlt0(const __m256i *in, u32 *matches, const INTEGER comp) {")
print("  if (comp == 0)")
print("     memset(matches, 1, 256 * sizeof(*matches));")
print("  else")
print("     memset(matches, 0, 256 * sizeof(*matches));")
print("}")
print("")

for bit in range(1,33):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 256-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void filterlt{0}(const __m256i *in, u32 *matches, const INTEGER comp) {{".format(bit))
    print("  /* we are going to access  {0} 256-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
        print("  __m256i w0;")
    else:
        print("  __m256i w0, w1;")
    print("  auto out = reinterpret_cast<__m256i *>(matches);")
    if(bit < 32): print("  const __m256i mask = _mm256_set1_epi32({0});".format((1<<bit)-1))
    print("  const __m256i broadcomp = _mm256_set1_epi32(comp);")
    maskstr = " _mm256_and_si256 ( mask, {0}) "
    if (bit == 32) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm256_lddqu_si256 (in);")
    for j in range(int(howmany(bit)/8)):
        firstword = int(j * bit / 32)
        secondword = int((j * bit + bit - 1)/32)
        if(secondword > oldword):
            print("  w{0} = _mm256_lddqu_si256 (in + {1});".format(secondword%2,secondword))
            oldword = secondword
        firstshift = (j*bit) % 32
        firstshiftstr = "_mm256_srli_epi32( w{0} , "+str(firstshift)+") "
        if(firstshift == 0):
            firstshiftstr =" w{0} " # no need
        wfirst = firstshiftstr.format(firstword%2)
        if( firstword == secondword):
            if(firstshift + bit != 32):
                wfirst  = maskstr.format(wfirst)
            print("  _mm256_storeu_si256(out + {0}, _mm256_cmpgt_epi32(broadcomp, {1}));".format(j,wfirst))
        else:
            secondshift = (32-firstshift)
            wsecond = "_mm256_slli_epi32( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
            wfirstorsecond = " _mm256_or_si256 ({0},{1}) ".format(wfirst,wsecond)
            wfirstorsecond = maskstr.format(wfirstorsecond)
            print("  _mm256_storeu_si256(out + {0},\n    _mm256_cmpgt_epi32(broadcomp, {1}));".format(j,wfirstorsecond))
    print("}")
    print("")

for ct in comp_types:
    print("void filter{0}(const __m256i *in, u32 *matches, const INTEGER comp, const u8 bit) {{".format(ct))
    print(" switch(bit) {")
    for bit in range(0,33):
        print(" case {0}:".format(bit))
        print("     filter{0}{1}(in, matches, comp);".format(ct,bit))
        print("     break;")
    print(" }")
    print("}")
    print("")


print("void filter(const __m256i *in, u32 *matches, const u8 bit, const algebra::Predicate<INTEGER> &predicate) {")
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