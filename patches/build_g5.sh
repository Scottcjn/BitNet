#!/bin/bash
# build_g5.sh - Build BitNet for Power Mac G5 (big-endian PowerPC)
#
# Requirements:
#   - Mac OS X 10.5 Leopard (or Linux ppc64be)
#   - GCC 10+ with C++17 support
#   - Model file: bitnet_b1_58-large converted to GGUF I2_S format
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

echo "=== BitNet G5 Big-Endian Build ==="
echo "CC:  $CC"
echo "CXX: $CXX"
echo ""

# Step 1: Apply big-endian patches to llama.cpp submodule
echo ">>> Step 1: Applying big-endian patches..."
cd 3rdparty/llama.cpp
if git diff --quiet HEAD 2>/dev/null; then
    git apply ../../patches/g5-big-endian.patch
    echo "    Applied g5-big-endian.patch"
else
    echo "    Submodule already has local changes, skipping patch"
fi

# Step 2: Copy regex compatibility header
echo ">>> Step 2: Installing regex-ppc.h..."
cp ../../patches/regex-ppc.h common/regex-ppc.h
echo "    Installed common/regex-ppc.h"

# Step 3: Build using Makefile with G5-safe flags
# -Os is the highest safe optimization level on G5 (O2/O3 cause Bus errors)
# -fno-strict-aliasing prevents misaligned access from type-punning
# -include common/regex-ppc.h replaces broken std::regex on PPC BE
echo ">>> Step 3: Building llama-cli with G5 flags..."
echo "    (This takes several minutes on dual G5)"

OPTLEVEL="${OPTLEVEL:-Os}"

make -j2 \
    CC="$CC" \
    CXX="$CXX" \
    GGML_NO_METAL=1 \
    LLAMA_NO_ACCELERATE=1 \
    LLAMA_NO_LLAMAFILE=1 \
    "GGML_NO_OPENMP=" \
    MK_CFLAGS="-mcpu=970 -maltivec -${OPTLEVEL} -fno-strict-aliasing -I ggml/include" \
    MK_CXXFLAGS="-mcpu=970 -maltivec -${OPTLEVEL} -fno-strict-aliasing -std=gnu++17 -I ggml/include -include common/regex-ppc.h" \
    MK_LDFLAGS="-L$(dirname $CC)/../lib -lgomp" \
    llama-cli

echo ""
echo "=== Build complete ==="
echo ""
echo "Run inference with:"
echo "  ./3rdparty/llama.cpp/llama-cli \\"
echo "    -m <model>.gguf \\"
echo "    -p \"Once upon a time\" \\"
echo "    -n 30 -t 2 --no-warmup --no-mmap"
echo ""
echo "Benchmarks (Power Mac G5 Dual 2.0GHz, -Os):"
echo "  bitnet_b1_58-large (728M): pp5 = 4.31 t/s, tg = 1.61 t/s"
