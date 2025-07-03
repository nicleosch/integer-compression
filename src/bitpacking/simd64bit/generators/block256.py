# Generates code for AVXBitPacking.cpp

def howmany(bit):
    """ how many values are we going to pack? """
    return 256

def howmanywords(bit):
    return int((howmany(bit) * bit + 255)/256)

def howmanybytes(bit):
    return howmanywords(bit) * 32

print("""typedef void (*avxpackblockfnc)(const u64 *pin, __m256i *compressed);""")
print("""typedef void (*avxunpackblockfnc)(const __m256i *compressed, u64 *out);""")


def plurial(number):
    if(number != 1):
        return "s"
    else :
        return ""

print("")
print("static void avxpackblock0(const u64 *pin, __m256i *compressed) {")
print("  (void)compressed;")
print("  (void) pin; /* we consumed {0} 64-bit integer{1} */ ".format(howmany(0),plurial(howmany(0))))
print("}")
print("")

for bit in range(1,65): # 64 possible bit-sizes
    print("")
    print("/* we are going to pack {0} {1}-bit values, touching {2} 256-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void avxpackblock{0}(const u64 *pin, __m256i * compressed) {{".format(bit))
    print("  const __m256i * in = (const __m256i *)  pin;")
    print("  /* we are going to touch  {0} 256-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
      print("  __m256i w0;")
    else:
      print("  __m256i w0, w1;")
    if( (bit & (bit-1)) != 0) : print("  __m256i tmp; /* used to store inputs at word boundary */")
    oldword = 0
    for j in range(int(howmany(bit)/4)): # 64 loads (4 integers per load -> 4 * 64 = 256 -> 256 loaded integers in total)
      firstword = int(j * bit / 64)
      if(firstword > oldword): # overflow? -> store the fully occupied word
        print("  _mm256_storeu_si256(compressed + {0}, w{1});".format(oldword,oldword%2))
        oldword = firstword
      secondword = int((j * bit + bit - 1)/64)
      firstshift = (j*bit) % 64 # how much to shift the bits to the left?
      if( firstword == secondword): # we can fit all bits into the first word
          if(firstshift == 0):
            print("  w{0} = _mm256_lddqu_si256 (in + {1});".format(firstword%2,j))
          else:
            print("  w{0} = _mm256_or_si256(w{0},_mm256_slli_epi64(_mm256_lddqu_si256 (in + {1}) , {2}));".format(firstword%2,j,firstshift))
      else: # bits need to be wrapped around words
          print("  tmp = _mm256_lddqu_si256 (in + {0});".format(j))
          print("  w{0} = _mm256_or_si256(w{0},_mm256_slli_epi64(tmp , {2}));".format(firstword%2,j,firstshift))
          secondshift = 64-firstshift # shift to write the carry into the new word
          print("  w{0} = _mm256_srli_epi64(tmp,{2});".format(secondword%2,j,secondshift))
    print("  _mm256_storeu_si256(compressed + {0}, w{1});".format(secondword,secondword%2))
    print("}")
    print("")

print("")
print("static void avxpackblockmask0(const u64 * pin, __m256i * compressed) {")
print("  (void)compressed;")
print("  (void) pin; /* we consumed {0} 64-bit integer{1} */ ".format(howmany(0),plurial(howmany(0))))
print("}")
print("")

for bit in range(1,65):
    print("")
    print("/* we are going to pack {0} {1}-bit values, touching {2} 256-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void avxpackblockmask{0}(const u64 * pin, __m256i * compressed) {{".format(bit))
    print("  /* we are going to touch  {0} 256-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
      print("  __m256i w0;")
    else:
      print("  __m256i w0, w1;")
    print("  const __m256i * in = (const __m256i *) pin;")
    if(bit < 64): print("  const __m256i mask = _mm256_set1_epi64x({0});".format((1<<bit)-1));
    def maskfnc(x):
        if(bit == 64): return x
        return " _mm256_and_si256 ( mask, {0}) ".format(x)
    if( (bit & (bit-1)) != 0) : print("  __m256i tmp; /* used to store inputs at word boundary */")
    oldword = 0
    for j in range(int(howmany(bit)/4)):
      firstword = int(j * bit / 64)
      if(firstword > oldword):
        print("  _mm256_storeu_si256(compressed + {0}, w{1});".format(oldword,oldword%2))
        oldword = firstword
      secondword = int((j * bit + bit - 1)/64)
      firstshift = (j*bit) % 64
      loadstr = maskfnc(" _mm256_lddqu_si256 (in + {0}) ".format(j))
      if( firstword == secondword):
          if(firstshift == 0):
            print("  w{0} = {1};".format(firstword%2,loadstr))
          else:
            print("  w{0} = _mm256_or_si256(w{0},_mm256_slli_epi64({1} , {2}));".format(firstword%2,loadstr,firstshift))
      else:
          print("  tmp = {0};".format(loadstr))
          print("  w{0} = _mm256_or_si256(w{0},_mm256_slli_epi64(tmp , {2}));".format(firstword%2,j,firstshift))
          secondshift = 64-firstshift
          print("  w{0} = _mm256_srli_epi64(tmp,{2});".format(secondword%2,j,secondshift))
    print("  _mm256_storeu_si256(compressed + {0}, w{1});".format(secondword,secondword%2))
    print("}")
    print("")

print("static void avxunpackblock0(const __m256i * compressed, u64 * pout) {")
print("  (void) compressed;")
print("  memset(pout,0,{0});".format(howmany(0)))
print("}")
print("")

for bit in range(1,65):
    print("")
    print("/* we packed {0} {1}-bit values, touching {2} 256-bit words, using {3} bytes */ ".format(howmany(bit),bit,howmanywords(bit),howmanybytes(bit)))
    print("static void avxunpackblock{0}(const __m256i * compressed, u64 * pout) {{".format(bit))
    print("  /* we are going to access  {0} 256-bit word{1} */ ".format(howmanywords(bit),plurial(howmanywords(bit))))
    if(howmanywords(bit) == 1):
      print("  __m256i w0;")
    else:
      print("  __m256i w0, w1;")
    print("  __m256i * out = (__m256i *) pout;")
    if(bit < 64): print("  const __m256i mask = _mm256_set1_epi64x({0});".format((1<<bit)-1));
    maskstr = " _mm256_and_si256 ( mask, {0}) "
    if (bit == 64) : maskstr = " {0} " # no need
    oldword = 0
    print("  w0 = _mm256_lddqu_si256 (compressed);")
    for j in range(int(howmany(bit)/4)):
      firstword = int(j * bit / 64)
      secondword = int((j * bit + bit - 1)/64)
      if(secondword > oldword):
        print("  w{0} = _mm256_lddqu_si256 (compressed + {1});".format(secondword%2,secondword))
        oldword = secondword
      firstshift = (j*bit) % 64
      firstshiftstr = "_mm256_srli_epi64( w{0} , "+str(firstshift)+") "
      if(firstshift == 0):
          firstshiftstr =" w{0} " # no need
      wfirst = firstshiftstr.format(firstword%2)
      if( firstword == secondword):
          if(firstshift + bit != 64):
            wfirst  = maskstr.format(wfirst)
          print("  _mm256_storeu_si256(out + {0}, {1});".format(j,wfirst))
      else:
          secondshift = (64-firstshift)
          wsecond = "_mm256_slli_epi64( w{0} , {1} ) ".format((firstword+1)%2,secondshift)
          wfirstorsecond = " _mm256_or_si256 ({0},{1}) ".format(wfirst,wsecond)
          wfirstorsecond = maskstr.format(wfirstorsecond)
          print("  _mm256_storeu_si256(out + {0},\n    {1});".format(j,wfirstorsecond))
    print("}")
    print("")

print("static avxpackblockfnc avxfuncPackArr[] = {")
for bit in range(0,64):
  print("&avxpackblock{0},".format(bit))
print("&avxpackblock64")
print("};")

print("static avxpackblockfnc avxfuncPackMaskArr[] = {")
for bit in range(0,64):
  print("&avxpackblockmask{0},".format(bit))
print("&avxpackblockmask64")
print("};")


print("static avxunpackblockfnc avxfuncUnpackArr[] = {")
for bit in range(0,64):
  print("&avxunpackblock{0},".format(bit))
print("&avxunpackblock64")
print("};")
