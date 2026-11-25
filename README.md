# voxt-polish

End-to-end on-device voice-dictation polishing pipeline: data curation & PII scrubbing, full-parameter bf16 SFT of `Qwen3-0.6B` with P0 entity-preservation early stopping, 3-tier GGUF behavioral train-inference parity gate, `ngram-simple` speculative decoding ablation on Apple Silicon Metal, and a lightweight guardrail gateway.

> Privacy note: this repository contains code, prompt templates, synthetic examples (`examples/`), and anonymized aggregate metrics only. Personal dictation corpora, raw recordings, personal dictionaries, logs, and fine-tuned model weights are excluded via `.gitignore`.

## Components

- `clean_template.py` & `skills/dictation-data-curation/SKILL.md`: deterministic cleanup (filler words, stray question marks, CJK/ASCII spacing, inline numbered lists, entity-mismatch flagging) with external `--patches` JSON.
- `train_polish_sft.py`: pure PyTorch + `transformers` full-parameter bf16 SFT on RTX 4090 (peak VRAM 6.39 GiB), prompt loss masking, `enable_thinking=False`, chronological holdout split (latest 40 pairs), optional `--anchor-pairs`, and `--eval-each-epoch` early stopping by `p0_preserve_recall`.
- `prompts/sft_system.txt`: single-source 70-character system prompt shared verbatim between training, offline eval, and local serving.
- `eval_parity.py` & `run_local_parity_suite.sh`: converts HF checkpoint to `F16 / Q8_0 / Q4_0` GGUF and runs the behavioral parity gate (`p0_preserve_recall`, `under_drops`, `over_expands`, `<think>` leaks, char similarity vs ground truth and vs 4090 bf16, warm median/p90 latency, and tok/s).
- `polish_gateway.py` & `start_polish_0.6b.sh`: OpenAI-compatible local wrapper (deterministic dictionary alias pre/post replacement without mutating system prompt KV cache, CJK/ASCII spacing, and `<0.55x` / `>1.5x` / `<think>` circuit-breaker fallback with local bad-case logging).
- `bench/` & `keepwarm.plist`: Apple Silicon `llama.cpp` serving benchmarks and 30-second dual-slot GPU keep-warm (`slot 1` ping preserves `slot 0` prefix cache).

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
| **`Q8_0` + `ngram-simple` (n=1, m=6)** | **610 MB** | **0.9958 (29/40)** | **0.8319** | **100.0%** | **0.223s / 0.568s** | **232.1** | **47.1% / 3.82** |
| `Q8_0` + `polish_gateway.py` (no-spec) | 610 MB | 0.9929 (27/40) | 0.8325 | 100.0% | — | — | Ref entity recall `0.794 -> 0.810`, 0 fallbacks |
| `Q4_0` + `ngram-simple` (n=1, m=6) | 364 MB | **0.9252 (6/40) FAIL** | 0.8103 | 100.0% | 0.211s / 0.567s | 227.4 | 37.0% / 3.22 |

Key engineering findings:
- **Why `Q8_0` beats `Q4_0` on 0.6B**: `Q4_0` failed the `>= 0.95` parity gate (`0.9252`, exact match dropped `29 -> 6/40`, `sim_gt` dropped `-2.16 pt`) and introduced a polarity inversion on holdout `#5` ("auto-delete" flipped to "do not auto-delete"), while offering virtually no speed gain over `Q8_0 + ngram-simple` (`0.211s` vs `0.223s`, `227.4` vs `232.1 tok/s`).
- **`ngram-simple` speedup on 0.6B**: Even with light paraphrasing, `ngram-simple (n=1, m=6)` achieved 47.1% median draft acceptance (3.82 tokens/step), cutting `Q8_0` median latency by **2.47x** (`0.551s -> 0.223s`) and p90 by **2.34x** (`1.331s -> 0.568s`) with bit-identical outputs vs no-spec.
- **0.6B `Q8_0` vs 4B `Q4_0` (`voxt_like.py` 6-sentence benchmark)**: `Qwen3-4B-Q4_0` (2.38 GB) warm median `0.458–0.560s` (`p90 0.667–0.709s`, `57.5–66.8 tok/s`) vs `Qwen3-0.6B-SFT-Q8_0` (610 MB) warm median **`0.096–0.098s`** (`p90 0.142–0.143s`, `331–347 tok/s`) — **4.8x–5.8x faster** at **3.9x smaller footprint**.
- **Known limitation (honest tradeoff)**: Run 4 raised unseen mid-sentence self-correction pass rate from `1/10` to `7/10`; `3/10` edge cases remain on short numeric or English inline corrections, to be closed with real user-edited DPO pairs.

### 3. End-to-End Real-Device Latency vs SGLang-Omni Native Voxt (M1 Pro 16 GB)

Measured on the same M1 Pro 16 GB with the official Voxt 2.1.3 Omni Runtime (`Qwen3-ASR 0.6B 4bit`) replaying 5 real Chinese dictations (A–E, 7.4–38.7 s) at 1.0× speed via `VOXTNEW_HELPER_CAPTURE_FILE`, releasing the key 0.5 s after speech ends. Metric: **Stop-to-text latency (`stop→end`)**.

| Metric (`stop→end`) | SGLang-Omni Native Default (4B `Q4_0`, `liveCleanLocal=false`) | Utter Client + 4B `Q4_0` (`liveCleanLocal=true` + punct reuse) | Utter + `0.6B Q8_0` + `ngram-simple` + Gateway (`liveCleanLocal=true`) | Reduction vs Native |
| --- | ---: | ---: | ---: | ---: |
| Overall Median | 1.31 s *(30-run batch: 1.69 s)* | 0.57 s | **0.55 s** | **-58.0%** *(-67.5% vs 30-run)* |
| Overall Mean | 1.64 s *(30-run batch: 2.16 s)* | 1.59 s *(E 8.7s spike)* | **0.80 s** | **-51.0%** |
| Recording A (7.4 s) | 0.80 s | 0.28 s | **0.31 s** | **-61.3%** |
| Recording B (12.8 s, ASR revision) | 1.01 s | 1.44 s | **0.48 s** | **-52.5%** |
| Recording D (23.9 s) | 1.60 s | 0.57 s | **0.64 s** | **-60.0%** |
| Recording E (38.7 s, longest) | 2.77 s *(max 4.66 s)* | 2.78 s *(max 8.67 s)* | **1.34 s *(zero spike)*** | **-51.6%** |

- **What cuts end-to-end latency in half**: SGLang-Omni's native Voxt disables live clean-up for local (`127.0.0.1`) LLMs and invalidates reused prefixes on single punctuation diffs. Combining local live clean-up + punctuation-tolerant prefix reuse (`sgl-project/sglang-omni#4643` + Utter) with the 610 MB fine-tuned `Qwen3-0.6B Q8_0` + `ngram-simple` cuts both median (`-58.0%`) and mean (`-51.0%`) stop-to-text wait by more than half while eliminating the 4B memory-contention tail spikes.
- **Measured GPU-sharing tradeoff**: running local live clean-up alongside local ASR adds ~40 ms to ASR Final and ~60 ms to the first live caption, while keeping live preview cadence unchanged (`1.01 s`).

## Quick Start (Synthetic Data)

```bash
# 1. Clean synthetic dictation pairs
python3 clean_template.py \
  --in-jsonl examples/synthetic_pairs.jsonl \
  --out-jsonl /tmp/synthetic_clean.jsonl

# 2. Train Qwen3-0.6B full-param bf16 SFT with per-epoch P0-Preserve early stopping
python3 train_polish_sft.py \
  --model Qwen/Qwen3-0.6B \
  --pairs /path/to/pairs_clean_scrubbed.jsonl \
  --anchor-pairs examples/anchor_pairs.jsonl \
  --out-dir /tmp/qwen3-0.6b-polish \
  --epochs 4 --lr 2e-5 --eval-each-epoch

# 3. Convert to F16 / Q8_0 / Q4_0 GGUF and run behavioral parity + ngram-simple ablation
./run_local_parity_suite.sh \
  /path/to/checkpoint \
  /path/to/eval_compare.jsonl \
  /tmp/gguf_bench \
  examples/sample_dictionary.json

# 4. Launch standalone Q8_0 + ngram-simple server (8091) + guardrail gateway (8092)
./start_polish_0.6b.sh /path/to/qwen3-0.6b-polish-q8_0.gguf --dict examples/sample_dictionary.json
```
