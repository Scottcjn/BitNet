# Contributing to bitnet.cpp

Thanks for helping improve bitnet.cpp. This project provides optimized CPU and
GPU inference support for 1-bit LLMs, including BitNet b1.58 models and the
POWER8/PowerPC port maintained in this fork.

## Good First Contributions

Helpful contributions include:

- Documentation fixes for setup, model conversion, inference, and benchmarking.
- Reproducible notes for specific operating systems, compilers, CPUs, or GPUs.
- Small tests for pure Python helpers and command-line behavior.
- Build script improvements for CMake, PowerPC, POWER8, or GPU workflows.
- Clarifications to kernel documentation, tuning guidance, and benchmark notes.

Please keep each pull request focused on one topic. For larger changes, open an
issue first so maintainers can discuss the target hardware, model format,
dependencies, and expected validation.

## Working Locally

Start from a fresh branch:

```bash
git checkout -b your-change-name
```

Install Python dependencies before working on scripts:

```bash
pip install -r requirements.txt
```

If your change touches GPU utilities, also review `gpu/requirements.txt`.
Do not commit downloaded models, generated benchmark outputs, local build
directories, or machine-specific logs.

## Validation

Choose checks that match the files you changed:

- Documentation-only changes: run `git diff --check` and review Markdown
  rendering where practical.
- Python helper changes: run the focused `pytest` command for the affected
  tests, or add a small test when behavior changes.
- Build or kernel changes: include the CMake or build command used, compiler
  version, target architecture, and hardware tested.
- Benchmark changes: include the model, quantization type, thread count, and
  command used to reproduce the result.

If you cannot run a full model download or hardware-specific benchmark, state
that clearly in the pull request and describe the lighter validation you did
run.

## Pull Request Checklist

- Explain the problem and the scope of the fix.
- List the exact validation commands you ran.
- Keep unrelated formatting changes out of the diff.
- Avoid committing generated files, model weights, caches, or local logs.
- Follow the Microsoft Open Source Code of Conduct in `CODE_OF_CONDUCT.md`.

## Issue Reports

For ordinary build, documentation, or usage issues, include:

- Operating system and architecture.
- Compiler and CMake versions.
- Python version and environment manager, if relevant.
- Model name, quantization type, and command used.
- The complete error message or the smallest useful excerpt.

Specific reproduction details are especially important for PowerPC, POWER8,
GPU, and model-conversion issues, where behavior can vary by platform.
