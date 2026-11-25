#!/bin/zsh
# Convert an HF Qwen3-0.6B polish checkpoint to F16 / Q8_0 / Q4_0 GGUF,
# then run the full behavioral parity + ngram-simple ON/OFF + gateway benchmark on port 8091
# (leaving any live production server on port 8090 untouched).
#
# Usage:
#   ./run_local_parity_suite.sh <hf_ckpt_dir> <eval_compare_jsonl> <out_dir> [dict_json]

set -euo pipefail

CKPT_DIR="${1:?Usage: $0 <hf_ckpt_dir> <eval_compare_jsonl> <out_dir> [dict_json]}"
COMPARE_JSONL="${2:?Need 4090 eval_compare.jsonl}"
OUT_DIR="${3:?Need output directory}"
DICT_JSON="${4:-}"

REPO_DIR="${0:A:h}"
CONVERT_PY="${CONVERT_PY:-${HOME}/Developer/05_External_Clones/llama.cpp/convert_hf_to_gguf.py}"
GGUF_PYTHON="${GGUF_PYTHON:-${HOME}/Developer/vllm-metal-env/.venv/bin/python}"
LLAMA_QUANTIZE="${LLAMA_QUANTIZE:-/opt/homebrew/bin/llama-quantize}"
LLAMA_SERVER="${LLAMA_SERVER:-/opt/homebrew/bin/llama-server}"
PORT="${TEST_PORT:-8091}"
SYS_PROMPT="${REPO_DIR}/prompts/sft_system.txt"

mkdir -p "$OUT_DIR"

F16_GGUF="$OUT_DIR/qwen3-0.6b-polish-f16.gguf"
Q8_GGUF="$OUT_DIR/qwen3-0.6b-polish-q8_0.gguf"
Q4_GGUF="$OUT_DIR/qwen3-0.6b-polish-q4_0.gguf"

if [[ ! -f "$F16_GGUF" ]]; then
  echo "=== [1/3] Converting HF checkpoint -> F16 GGUF ==="
  "$GGUF_PYTHON" "$CONVERT_PY" "$CKPT_DIR" --outfile "$F16_GGUF" --outtype f16
fi

if [[ ! -f "$Q8_GGUF" ]]; then
  echo "=== [2/3] Quantizing F16 -> Q8_0 GGUF ==="
  "$LLAMA_QUANTIZE" "$F16_GGUF" "$Q8_GGUF" Q8_0
fi

if [[ ! -f "$Q4_GGUF" ]]; then
  echo "=== [3/3] Quantizing F16 -> Q4_0 GGUF ==="
  "$LLAMA_QUANTIZE" "$F16_GGUF" "$Q4_GGUF" Q4_0
fi

ls -lh "$F16_GGUF" "$Q8_GGUF" "$Q4_GGUF"

run_case() {
  local label="$1"
  local model="$2"
  shift 2
  echo ""
  echo "=== Running parity & latency benchmark: $label ==="
  "$LLAMA_SERVER" \
    -m "$model" \
    --alias polish \
    -ngl 99 -c 4096 -np 2 \
    --host 127.0.0.1 --port "$PORT" \
    --reasoning off --reasoning-budget 0 \
    --load-mode mmap+mlock -ub 1024 --cache-reuse 64 -kvu \
    "$@" > "$OUT_DIR/server_${label}.log" 2>&1 &
  local pid=$!
  trap "kill $pid 2>/dev/null || true" EXIT

  for i in $(seq 1 60); do
    if curl -sf -m 1 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done

  python3 "$REPO_DIR/eval_parity.py" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --compare-jsonl "$COMPARE_JSONL" \
    --system-prompt "$SYS_PROMPT" \
    --label "$label" \
    --out "$OUT_DIR/parity_${label}.json"

  if [[ -n "$DICT_JSON" && -f "$DICT_JSON" && ( "$label" == "q8_0_nospec_faoff" || "$label" == "q4_0_nospec_faoff" ) ]]; then
    echo "--- Running gateway (+dict +spacing +guardrail) on $label ---"
    python3 "$REPO_DIR/polish_gateway.py" eval \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --compare-jsonl "$COMPARE_JSONL" \
      --system-prompt "$SYS_PROMPT" \
      --dict "$DICT_JSON" \
      --label "${label}_gateway" \
      --out "$OUT_DIR/parity_${label}_gateway.json"
  fi

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - EXIT
  sleep 1
}

# 1. F16 parity gate (CUDA bf16 vs Metal F16, isolates backend/template bugs)
run_case "f16_nospec_faoff" "$F16_GGUF" -fa off

# 2. Q8_0 ablation: FA off vs FA auto, ngram-simple OFF vs ON (n=1, m=6 and n=1, m=4)
run_case "q8_0_nospec_faoff" "$Q8_GGUF" -fa off
run_case "q8_0_nospec_faon"  "$Q8_GGUF" -fa on
run_case "q8_0_ngram_m6"     "$Q8_GGUF" -fa off --spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m 6 --spec-draft-n-max 6
run_case "q8_0_ngram_m4"     "$Q8_GGUF" -fa off --spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m 4 --spec-draft-n-max 4

# 3. Q4_0 ablation: ngram-simple OFF vs ON
run_case "q4_0_nospec_faoff" "$Q4_GGUF" -fa off
run_case "q4_0_ngram_m6"     "$Q4_GGUF" -fa off --spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m 6 --spec-draft-n-max 6

echo ""
echo "=== ALL PARITY & ABLATION CASES COMPLETE ==="
python3 - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
out_dir = Path(sys.argv[1])
for p in sorted(out_dir.glob("parity_*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))["summary"]
    print(f"{d['label']:26s} | sim_gt={d['mean_char_sim_gt']:.4f} | p0_pres={d['p0_preserve_recall']:.4f}({d['p0_preserve_error_rows']}err) | parity_bf16={d.get('parity_mean_sim_vs_bf16',0):.4f} | med={d['warm_median_s']:.3f}s p90={d['warm_p90_s']:.3f}s | {d['gen_tok_per_s']:.1f} tok/s | drop={d['under_drops']} exp={d['over_expands']} think={d['think_leaks']}")
EOF
