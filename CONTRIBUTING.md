# Contributing to BitNet.cpp

Thank you for contributing to BitNet.cpp, Microsoft's 1-bit LLM inference framework running on CPU.

## Project Overview

BitNet.cpp provides hardware-accelerated inference for 1-bit LLMs (like BitNet-b1.58-2B-4T) on CPU via llama.cpp compatibility. It supports x86_64 (AVX2/AVX512) and ARM64 (NEON) architectures.

## Development Setup

### Prerequisites

- C++17 compatible compiler (GCC 9+, Clang 12+, MSVC 2019+)
- CMake 3.15+
- Python 3.10+ (for model conversion)

### Building

```bash
git clone https://github.com/Scottcjn/BitNet.git
cd BitNet
mkdir build && cd build
cmake ..
make -j$(nproc)

# Or use the build script
./bitnet.sh build
```

### Downloading Models

```bash
# Download a 1-bit model from Hugging Face
python download-model.py microsoft/BitNet-b1.58-2B-4T
```

### Running

```bash
# Interactive mode
./bitnet -m models/bitnet-b1.58-2b-4t.gguf -p "Your prompt here"

# Chat mode
./bitnet -m models/bitnet-b1.58-2b-4t.gguf --color -ins -r "User:"
```

## Code Style

- C++17 standard
- Follow LLVM/Google style guide
- Use `uint64_t`/`size_t` for bit operations
- SIMD intrinsics in separate compilation units
- Header-only utilities allowed

## Testing

```bash
# Build tests
cmake .. -DBITNET_BUILD_TESTS=ON
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Benchmark
./benchmark --model models/bitnet-b1.58-2b-4t.gguf --n-prompt 512 --n-gen 128
```

## Architecture Notes

- llama.cpp compatible model format (GGUF)
- Weight-only quantization (WOQ) for 1.58-bit models
- SIMD kernels: AVX2 (x86), NEON (ARM64)
- KV cache and prompt processing fully supported

## Submitting Changes

1. Fork the repository
2. Create a branch: `git checkout -b fix/your-fix`
3. Ensure code compiles and tests pass
4. Submit a pull request

## Ideas for Contributions

- Additional architecture support (POWER8/POWER9 via VSX)
- More SIMD kernel optimizations
- New quantization formats
- WASM/WebGPU backend
- Memory footprint improvements
