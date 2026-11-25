#!/bin/zsh
# Launch a standalone Qwen3-0.6B polish llama-server + optional guardrail gateway on test ports
# (default: llama-server on 8091, gateway on 8092) without touching any live service on 8090.
#
# Usage:
#   ./start_polish_0.6b.sh <model.gguf> [--no-spec] [--dict /path/to/dictionary.json]

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model.gguf> [--no-spec] [--dict /path/to/dictionary.json]}"
shift

# Default: ngram-simple ON (measured 2.47x speedup on Q8_0 with zero parity loss)
USE_SPEC=1
DICT_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec) USE_SPEC=1; shift ;;
    --no-spec) USE_SPEC=0; shift ;;
    --dict) DICT_PATH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

REPO_DIR="${0:A:h}"
LLAMA_PORT="${LLAMA_PORT:-8091}"
GW_PORT="${GW_PORT:-8092}"
LLAMA_SERVER="${LLAMA_SERVER:-/opt/homebrew/bin/llama-server}"
BADCASE_LOG="${BADCASE_LOG:-/tmp/voxt_polish_badcases.jsonl}"

SPEC_ARGS=()
if [[ "$USE_SPEC" -eq 1 ]]; then
  SPEC_ARGS=(--spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m 6 --spec-draft-n-max 6)
fi

echo "Starting llama-server on 127.0.0.1:${LLAMA_PORT} (model=$(basename "$MODEL_PATH"), spec=${USE_SPEC})..."
"$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --alias qwen3-0.6b-polish \
  -ngl 99 -c 4096 -np 2 \
  --host 127.0.0.1 --port "$LLAMA_PORT" \
  --reasoning off --reasoning-budget 0 \
  --load-mode mmap+mlock -ub 1024 -fa off --cache-reuse 64 -kvu \
  "${SPEC_ARGS[@]}" &
LLAMA_PID=$!

cleanup() {
  kill "$LLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 60); do
  if curl -sf -m 1 "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "llama-server healthy on :${LLAMA_PORT}. Starting guardrail gateway on :${GW_PORT}..."
GW_ARGS=(serve --port "$GW_PORT" --upstream "http://127.0.0.1:${LLAMA_PORT}/v1" --badcase-log "$BADCASE_LOG")
if [[ -n "$DICT_PATH" ]]; then
  GW_ARGS+=(--dict "$DICT_PATH")
fi
python3 "$REPO_DIR/polish_gateway.py" "${GW_ARGS[@]}"
