#!/bin/bash
# build_g5.sh - Build BitNet for Power Mac G5 (big-endian PowerPC AltiVec)
#
# Requirements:
#   - Mac OS X 10.5 Leopard (or Linux ppc64be)
#   - GCC 10+ with C++17 support
#   - Model file: bitnet_b1_58-large converted to GGUF I2_S format
#
# Two levels of AltiVec SIMD:
#   1. Dot-product kernels (ggml-bitnet-mad.cpp) - vec_msum, vec_ld, vec_splat_u8
#   2. Framework vectorization (ggml.c GGML_SIMD + ggml-quants.c quantize_row_i8_s)
#      - vec_ld/vec_st, vec_madd, vec_abs, vec_round, vec_packs
#      - Applied via g5-altivec-framework.patch
#
# Usage:
#   ./patches/build_g5.sh [GCC_PREFIX]
#
# Example:
#   ./patches/build_g5.sh /usr/local/gcc-10/bin
#   ./patches/build_g5.sh   # uses gcc/g++ from PATH

set -e

GCC_PREFIX="${1:-}"
if [ -n "$GCC_PREFIX" ]; then
    CC="${GCC_PREFIX}/gcc"
    CXX="${GCC_PREFIX}/g++"
else
    CC="gcc"
    CXX="g++"
fi

echo "=== BitNet G5 AltiVec Build ==="
echo "CC:  $CC"
echo "CXX: $CXX"
echo ""

# Step 1: Apply big-endian patches to llama.cpp submodule
echo ">>> Step 1: Applying big-endian patches..."
cd 3rdparty/llama.cpp
if git diff --quiet HEAD 2>/dev/null; then
    git apply ../../patches/g5-big-endian.patch
    echo "    Applied g5-big-endian.patch"
    git apply ../../patches/g5-altivec-framework.patch
    echo "    Applied g5-altivec-framework.patch"
else
    echo "    Submodule already has local changes, skipping patch"
fi

# Step 2: Copy regex compatibility header
echo ">>> Step 2: Installing regex-ppc.h..."
cp ../../patches/regex-ppc.h common/regex-ppc.h
echo "    Installed common/regex-ppc.h"

# Step 3: Build using Makefile with G5 AltiVec flags
# C code uses -O3 (safe on PPC with GCC 10). C++ uses -Os because GCC 10.5
# has miscompile bugs at -O2/-O3 on PPC that cause Bus errors in arg.cpp,
# llama.cpp, and llama-vocab.cpp (aggressive vector register spills hit
# Mach-O ABI stack alignment issues).
# -include common/regex-ppc.h replaces broken std::regex on PPC BE
# -lm required for roundf() in AltiVec quantize path
echo ">>> Step 3: Building llama-cli with AltiVec flags..."
echo "    (This takes several minutes on dual G5)"
echo "    NOTE: Use -t 1 for inference (single thread is faster due to"
echo "          barrier overhead on 870 graph nodes per token)"

make -j2 \
    CC="$CC" \
    CXX="$CXX" \
    GGML_NO_METAL=1 \
    LLAMA_NO_ACCELERATE=1 \
    LLAMA_NO_LLAMAFILE=1 \
    "GGML_NO_OPENMP=" \
    MK_CFLAGS="-mcpu=970 -maltivec -O3 -I ggml/include" \
    MK_CXXFLAGS="-mcpu=970 -maltivec -Os -std=gnu++17 -I ggml/include -include common/regex-ppc.h" \
    MK_LDFLAGS="-L$(dirname $CC)/../lib -lgomp -lm" \
    llama-cli

echo ""
echo "=== Build complete ==="
echo ""
echo "Run inference with:"
echo "  ./3rdparty/llama.cpp/llama-cli \\"
echo "    -m <model>.gguf \\"
echo "    -p \"Once upon a time\" \\"
echo "    -n 30 -t 1 --no-warmup --no-mmap"
echo ""
echo "Performance: pp6 ~5.1 t/s, tg ~1.5 t/s (AltiVec + framework SIMD, -t 1)"
echo ""
echo "AltiVec SIMD coverage:"
echo "  - Dot product kernels (ggml-bitnet-mad.cpp): 16x raw speedup"
echo "  - Framework ops (ggml_vec_scale/add/mul/dot/mad): ~4x via GGML_SIMD"
echo "  - Activation quantize (quantize_row_i8_s): ~4x via vec_abs/vec_packs"
echo ""
echo "End-to-end speedup is limited by -Os C++ (GCC miscompile workaround)"
echo "and 870 barrier syncs per token at single-thread."
