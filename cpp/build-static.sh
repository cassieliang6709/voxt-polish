#!/usr/bin/env bash
# Build voxt_polish_cpp as one self-contained binary (no Homebrew at run
# time): llama.cpp at the commit the engine was written against, built in
# statically with its Metal shaders embedded. Output: cpp/build-static/voxt_polish_cpp
set -euo pipefail
LLAMA_COMMIT=b1b770d90d94cad9e9ea5b993bdbc383a522fb28
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/third_party/llama.cpp"
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$SRC" && git -C "$SRC" init -q
  git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if [ "$(git -C "$SRC" rev-parse -q --verify HEAD || true)" != "$LLAMA_COMMIT" ]; then
  git -C "$SRC" fetch -q --depth 1 origin "$LLAMA_COMMIT"
  git -C "$SRC" checkout -q FETCH_HEAD
fi
cmake -S "$HERE" -B "$HERE/build-static" -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR="$SRC" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 >/dev/null
cmake --build "$HERE/build-static" --target voxt_polish_cpp -j
otool -L "$HERE/build-static/voxt_polish_cpp"
