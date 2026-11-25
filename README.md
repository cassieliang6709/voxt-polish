# voxt-polish

End-to-end on-device voice-dictation polishing stack for Apple Silicon: data curation & PII scrubbing, full-parameter bf16 SFT of `Qwen3-0.6B` with P0 entity-preservation early stopping, 3-tier GGUF behavioral train-inference parity gate, a zero-Python **C++20 embedded `libllama` runtime**, and **CoreML Apple Neural Engine (ANE)** Silero VAD offloading.

> **Privacy & Weight Distribution Note:** This public repository contains **all source code, prompt templates, synthetic datasets (`examples/`), and anonymized aggregate metrics** — ready to clone, build, and run directly. Personal dictation corpora, raw recordings, personal dictionaries, logs, and personal fine-tuned weights are excluded via `.gitignore` (see [Quick Start](#quick-start) below to run immediately with any public Hugging Face `Qwen3` GGUF or train your own on `examples/`).

## Components

- `cpp/` (`CMakeLists.txt`, `polish_engine.hpp/.cpp`, `main.cpp`): zero-Python **C++20 embedded `libllama` inference engine & OpenAI-compatible HTTP/SSE server** — pins system prompt KV `[0, N_sys)` with incremental LCP prefix reuse (`llama_memory_seq_rm`), hard-masks `<think>`/`</think>` via a custom `llama_sampler_i`, runs in-process `ngram-simple` speculative decoding (`n=1, m=6`), and integrates deterministic dictionary replacement + length circuit-breaker guardrails.
- `apple_silicon/` (`convert_silero_coreml.py`, `convert_silero_coreml_batched.py`, `ane_vad_bench.swift`, `ane_vad_bench_b32.swift`, `run_heterogeneous_bench.py`): **Apple Silicon Metal GPU contention suite** (MLX `qwen3_asr_server` vs GGML `voxt_polish_cpp`) and **32-chunk batched CoreML FP16 Silero VAD v6** offload to Apple Neural Engine (ANE).
- `clean_template.py` & `skills/dictation-data-curation/SKILL.md`: deterministic cleanup (filler words, stray question marks, CJK/ASCII spacing, inline numbered lists, entity-mismatch flagging) with external `--patches` JSON.
- `train_polish_sft.py`: pure PyTorch + `transformers` full-parameter bf16 SFT (peak VRAM `6.39 GiB` on RTX 4090), prompt loss masking, `enable_thinking=False`, chronological holdout split, `--anchor-pairs`, and `--eval-each-epoch` early stopping by `p0_preserve_recall`.
- `prompts/sft_system.txt`: single-source 70-character system prompt shared verbatim between training, offline eval, Python gateway, and C++20 serving.
- `eval_parity.py` & `run_local_parity_suite.sh`: converts HF checkpoint to `F16 / Q8_0 / Q4_0` GGUF and runs the behavioral parity gate (`p0_preserve_recall`, `under_drops`, `over_expands`, `<think>` leaks, char similarity vs ground truth & bf16, latency, tok/s).
- `polish_gateway.py` & `start_polish_0.6b.sh`: Python reference OpenAI-compatible local gateway wrapper & launch script.
- `bench/` & `keepwarm.plist`: Apple Silicon serving benchmarks, anonymized low-level results (`bench/apple_silicon_lowlevel_results.json`), and 30-second dual-slot GPU keep-warm (`slot 1` ping preserves `slot 0` prefix cache).

## Quick Start

### Prerequisites (macOS Apple Silicon)

```bash
brew install llama.cpp cmake
git clone https://github.com/cassieliang6709/voxt-polish.git
cd voxt-polish
```

### 1. Build & Run the C++20 Embedded `libllama` Engine (60 seconds, zero Python)

Use any public `Qwen3` GGUF (e.g., `Qwen3-0.6B-Q8_0.gguf` or `Qwen3-4B-Instruct-2507-Q4_0.gguf` from Hugging Face) or your own fine-tuned checkpoint:

```bash
# Build native C++20 binary linked directly against Homebrew libllama
cmake -B cpp/build cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j

# Option A: Quick single-utterance smoke test
./cpp/build/voxt_polish_cpp \
  --model /path/to/qwen3-0.6b-q8_0.gguf \
  --system-prompt prompts/sft_system.txt \
  --dict examples/sample_dictionary.json

# Option B: Benchmark 12-step incremental streaming LCP KV reuse
./cpp/build/voxt_polish_cpp \
  --model /path/to/qwen3-0.6b-q8_0.gguf \
  --system-prompt prompts/sft_system.txt \
  --bench-streaming

# Option C: Launch standalone OpenAI-compatible HTTP/SSE server (drop-in for Voxt / Utter)
./cpp/build/voxt_polish_cpp \
  --model /path/to/qwen3-0.6b-q8_0.gguf \
  --system-prompt prompts/sft_system.txt \
  --dict examples/sample_dictionary.json \
  --draft-max 6 --serve --port 8093
```

Verify the server:

```bash
curl -s http://127.0.0.1:8093/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b-polish","messages":[{"role":"user","content":"嗯那个我觉得 llama dot cpp 的投机解码挺快的"}]}'
```

### 2. Clean Synthetic Data, Fine-Tune `Qwen3-0.6B`, and Run 3-Tier GGUF Parity Gate

```bash
# 1. Deterministic cleanup on bundled synthetic dictation pairs
python3 clean_template.py \
  --in-jsonl examples/synthetic_pairs.jsonl \
  --out-jsonl /tmp/synthetic_clean.jsonl

# 2. Full-parameter bf16 SFT (requires CUDA GPU, ~6.4 GiB VRAM)
python3 train_polish_sft.py \
  --model Qwen/Qwen3-0.6B \
  --pairs /path/to/your_clean_pairs.jsonl \
  --anchor-pairs examples/anchor_pairs.jsonl \
  --out-dir /tmp/qwen3-0.6b-polish \
  --epochs 4 --lr 2e-5 --eval-each-epoch

# 3. Export F16 / Q8_0 / Q4_0 GGUF and run behavioral parity + ngram-simple ablation
./run_local_parity_suite.sh \
  /tmp/qwen3-0.6b-polish/checkpoint \
  /tmp/qwen3-0.6b-polish/eval_compare.jsonl \
  /tmp/gguf_bench \
  examples/sample_dictionary.json
```

### 3. Convert & Benchmark 32-Chunk Batched CoreML ANE Silero VAD (Apple Silicon)

```bash
uv venv apple_silicon/.venv --python 3.12
uv pip install --python apple_silicon/.venv/bin/python torch coremltools safetensors numpy

# Export 32-chunk (1.024s) CoreML FP16 MLProgram targeting Apple Neural Engine
apple_silicon/.venv/bin/python apple_silicon/convert_silero_coreml_batched.py \
  --safetensors /path/to/silero-vad-v6/model.safetensors \
  --out apple_silicon/SileroVADv6_B32.mlpackage

xcrun coremlcompiler compile apple_silicon/SileroVADv6_B32.mlpackage apple_silicon/
swiftc -O apple_silicon/ane_vad_bench_b32.swift -o apple_silicon/ane_vad_bench_b32

# Run on any 16 kHz mono PCM16 WAV files
./apple_silicon/ane_vad_bench_b32 apple_silicon/SileroVADv6_B32.mlmodelc cpu_and_ne /path/to/*.wav
```

---

## Anonymized Experimental Results

### 1. Full-Parameter bf16 SFT Progression (40 Chronological Clean Holdout Pairs, RTX 4090)

| Arm | Mean Char Sim | P0-Preserve (ASR Entities) | Filler「然后」 | Under-Drops / Over-Expands | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Zero-shot `Qwen3-0.6B` | 0.0726 | 79.7% (47/59) | 0 | 0 / 30 | 75% answered as chatbot or translated |
| Run 1 (325 raw pairs, 4 ep) | 0.8217 | 96.6% (57/59) | 26 | 0 / 0 | Heavy filler retention; `JD -> JIRA` error |
| Run 2 (324 clean pairs, 4 ep) | 0.8193 | 91.5% (54/59) | 3 (-88.5%) | 0 / 0 | Overfit at ep 4: 3 entity/clause regressions |
| Run 3 Best (PII-scrubbed 324 + 24 synthetic anchors, Ep 2) | 0.8316 | 100.0% (59/59) | 4 | 0 / 0 | Zero P0 entity errors; unseen self-correction 1/10 |
| **Run 4 Best (PII-scrubbed 324 + 50 anchors incl. self-correction, Ep 1)** | **0.8362** | **100.0% (59/59)** | **3** | **0 / 0** | **Unseen self-correction jumped 1/10 -> 7/10** |

Run 3 per-epoch curve (proving why training loss alone misleads on small datasets):
- Epoch 1 (`loss=0.9760`): `sim=0.8265`, `P0-Preserve=98.3%`, `under_drops=0`
- Epoch 2 (`loss=0.6105`): `sim=0.8316`, `P0-Preserve=100.0%`, `under_drops=0` (selected)
- Epoch 3 (`loss=0.4712`): `sim=0.8290`, `P0-Preserve=93.2%`, `under_drops=1`
- Epoch 4 (`loss=0.4259`): `sim=0.8288`, `P0-Preserve=93.2%`, `under_drops=1`

### 2. CUDA bf16 -> Apple Silicon Metal GGUF Parity & `ngram-simple` Ablation (40 Holdout Pairs)

| Variant | Size | Parity vs 4090 bf16 (Exact) | Sim vs GT | P0-Preserve | Warm Med / p90 | Gen tok/s | Draft Acc / Mean Len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `F16` no-spec (`-fa off`) | 1.1 GB | 0.9977 (32/40) | 0.8314 | 100.0% | 0.569s / 1.424s | 90.3 | — |
| `Q8_0` no-spec (`-fa off`) | 610 MB | 0.9958 (29/40) | 0.8319 | 100.0% | 0.551s / 1.331s | 99.7 | — |
| `Q8_0` no-spec (`-fa on`) | 610 MB | 0.9962 (30/40) | 0.8323 | 100.0% | 0.491s / 1.354s | 100.1 | — |
| `Q8_0` + `ngram-simple` (n=1, m=4) | 610 MB | 0.9958 (29/40) | 0.8319 | 100.0% | 0.249s / 0.624s | 215.1 | 47.2% / 2.89 |
| `Q8_0` + `ngram-simple` (n=1, m=6, `llama-server`) | 610 MB | 0.9958 (29/40) | 0.8319 | 100.0% | 0.223s / 0.568s | 232.1 | 47.1% / 3.82 |
| **`Q8_0` C++20 `voxt_polish_cpp` (`m=6` + dict/guard)** | **610 MB** | **0.9958 raw / 0.9929 guard** | **0.8325** | **100.0%** | **0.202s / 0.556s** | **232.6 raw** | **56.0% med / 0 fallbacks** |
| `Q4_0` + `ngram-simple` (n=1, m=6) | 364 MB | **0.9252 (6/40) FAIL** | 0.8103 | 100.0% | 0.211s / 0.567s | 227.4 | 37.0% / 3.22 |

Key engineering findings:
- **Why `Q8_0` beats `Q4_0` on 0.6B**: `Q4_0` failed the `>= 0.95` parity gate (`0.9252`, exact match dropped `29 -> 6/40`, `sim_gt` dropped `-2.16 pt`) and introduced a polarity inversion on holdout `#5` ("auto-delete" flipped to "do not auto-delete"), while offering virtually no speed gain over `Q8_0 + ngram-simple` (`0.211s` vs `0.223s`).
- **0.6B C++20 vs 4B `Q4_0` (6-sentence warm HTTP)**: `Qwen3-4B-Q4_0` (`2.38 GB`, `llama-server`) warm median `0.458–0.560s` (`57.5–66.8 tok/s`) vs `Qwen3-0.6B-SFT-Q8_0` in `voxt_polish_cpp` (`610 MB`, zero Python) warm median **`0.0819s`** (`p90 0.1146s`, `258.4 tok/s`).
- **Known limitation (honest tradeoff)**: Run 4 raised unseen mid-sentence self-correction pass rate from `1/10` to `7/10`; `3/10` edge cases remain on short numeric or English inline corrections, targeted by future user-edited DPO pairs.

### 3. End-to-End Real-Device Latency vs SGLang-Omni Native Voxt (M1 Pro 16 GB)

Measured on M1 Pro 16 GB with the official Voxt 2.1.3 Omni Runtime (`Qwen3-ASR 0.6B 4bit` + `Silero VAD v6`) replaying 5 real Chinese dictations (A–E, 7.4–38.7 s) at 1.0× speed via `VOXTNEW_HELPER_CAPTURE_FILE`, releasing the key 0.5 s after speech ends. Metric: **Stop-to-text latency (`stop→end`)**.

| Recording (`stop→end`) | Native Voxt (4B `Q4_0`, live OFF) | Utter + 4B `Q4_0` (live ON) | Utter + 0.6B (`llama-server` + Py) | Utter + 0.6B C++20 (`Pieces=60s`) | **Utter + 0.6B C++20 + VAD Pieces (`15s`)** |
| --- | ---: | ---: | ---: | ---: | ---: |
| A (7.4 s) | 0.80 s | 0.28 s | 0.31 s | 0.37 s | *(unchanged, `< 15s`)* |
| B (12.8 s, ASR revision) | 1.01 s | 1.44 s | 0.48 s | 0.50 s | *(unchanged, `< 15s`)* |
| C (16.5 s) | 1.31 s | 0.45 s | 0.52 s | 0.50 s | **0.21 s** *(ASR 155 ms, sim 0.981)* |
| D (23.9 s) | 1.60 s | 0.57 s | 0.64 s | 0.57 s | **0.35 s** *(ASR 246 ms, sim 0.979)* |
| E (38.7 s, longest) | 2.77 s *(max 4.66 s)* | 2.78 s *(max 8.67 s)* | 1.34 s *(zero spike)* | 1.29 s | **0.25 s** *(11.1× vs 4B; 3-cut sim 0.763)* |

### 4. Apple Silicon Metal GPU Contention & CoreML ANE Offload (`bench/apple_silicon_lowlevel_results.json`)

- **C++20 Streaming LCP KV Reuse (`--bench-streaming`)**: Across a 12-step growing live-polish session, pinning system KV `[0, N_sys)` plus incremental LCP reuse cut newly prefilled tokens from `340` to **`149` (`-56.2%`, `86.6%` KV reuse)** and total session compute from `785.3 ms` to **`668.1 ms`**.
- **Metal GPU Contention (MLX ASR vs GGML LLM)**: Running MLX `qwen3_asr_server` and GGML `voxt_polish_cpp` concurrently on M1 Pro slowed 38.7 s ASR by **`+57.2%`** (`858.8 -> 1,349.9 ms`) and 0.6B LLM by **`+100.1%`** (`99.8 -> 199.7 ms`).
- **32-Chunk Batched CoreML ANE Silero VAD v6**: Naive single-chunk (`B=1`) CoreML ANE dispatch took `1,174.6 ms` on 38.7 s audio due to per-call IPC overhead; unrolling 32 chunks (`1.024 s`) in one CoreML FP16 MIL graph dropped it to **`20.64 ms`** (`17.3 us/chunk`, RTF `0.0005`), 5-recording sum **`52.42 ms`** (**`8.05×`** faster than MLX Metal `silero_vad_server`'s `422.2 ms`, **`99.75%`** decision agreement vs PyTorch FP32 across 1,194 chunks, `0%` Metal GPU contention).
