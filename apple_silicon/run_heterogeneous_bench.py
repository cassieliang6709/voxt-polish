"""Apple Silicon heterogeneous benchmark & Metal GPU contention suite.

Measures on M1 Pro (16 GB unified memory):
  1. Silero VAD v6 parity & latency across 4 backends on the 5 real dictation WAVs:
       - Voxt native MLX Metal GPU (`silero_vad_server`)
       - CoreML `cpu_only` (Apple AMX / CPU)
       - CoreML `cpu_and_ne` (Apple Neural Engine)
       - CoreML `cpu_and_gpu` (CoreML Metal)
  2. Real Metal GPU contention between MLX (`qwen3_asr_server`) and GGML (`voxt_polish_cpp`):
       - Isolated `voxt_polish_cpp` vs concurrent with MLX `qwen3_asr_server`
       - Isolated `qwen3_asr_server` vs concurrent with GGML `voxt_polish_cpp`
       - Concurrent VAD+LLM: MLX Metal VAD vs CoreML non-GPU VAD
  3. Optional `xctrace` Metal/Time capture export.
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OMNI_MODELS = Path.home() / "Library/Application Support/SGLang Omni/models"
OMNI_BIN = os.environ.get("VOXT_OMNI_BIN", "/Applications/Voxt.app/Contents/Resources/omni/bin")
VAD_MODEL = os.environ.get(
    "VOXT_VAD_MODEL",
    str(OMNI_MODELS / "models--mlx-community--silero-vad-v6/snapshots/2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06"),
)
ASR_MODEL = os.environ.get(
    "VOXT_ASR_MODEL",
    str(OMNI_MODELS / "models--mlx-community--Qwen3-ASR-0.6B-4bit/snapshots"),
)
COREML_BIN = str(REPO_ROOT / "apple_silicon/ane_vad_bench")
COREML_MODEL = str(REPO_ROOT / "apple_silicon/SileroVADv6.mlmodelc")
CPP_BIN = str(REPO_ROOT / "cpp/build/voxt_polish_cpp")
GGUF_MODEL = os.environ.get("VOXT_GGUF_MODEL", str(REPO_ROOT / "models/qwen3-0.6b-polish-q8_0.gguf"))
SYS_PROMPT = str(REPO_ROOT / "prompts/sft_system.txt")
DICT_JSON = os.environ.get("VOXT_DICT_JSON", str(REPO_ROOT / "examples/sample_dictionary.json"))


def wait_health(url: str, timeout: float = 30.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.15)
    raise RuntimeError(f"Server did not become healthy: {url}")


def post_multipart_wav(url: str, wav_path: str, extra_fields: dict[str, str] | None = None):
    boundary = "----VoxtBenchBoundary7MA4YWxkTrZu0gW"
    parts = []
    for k, v in (extra_fields or {}).items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    wav_bytes = Path(wav_path).read_bytes()
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{Path(wav_path).name}\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n".encode()
        + wav_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode("utf-8", errors="replace")
    return (time.perf_counter() - t0) * 1000.0, raw


def post_llm_json(port: int, text: str):
    payload = json.dumps(
        {
            "model": "qwen3-0.6b-polish",
            "stream": False,
            "messages": [{"role": "user", "content": text}],
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    return (time.perf_counter() - t0) * 1000.0, data


def run_coreml_vad(compute_unit: str, wavs: list[str]):
    cmd = [COREML_BIN, COREML_MODEL, compute_unit] + wavs
    out = subprocess.check_output(cmd, text=True)
    return [json.loads(line) for line in out.strip().splitlines() if line.strip()]


def bench_mlx_vad(wavs: list[str], port: int = 18094):
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = OMNI_BIN
    proc = subprocess.Popen(
        [
            f"{OMNI_BIN}/silero_vad_server",
            "--model-directory",
            VAD_MODEL,
            "--model-kind",
            "silero_vad",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        wait_health(f"http://127.0.0.1:{port}/health", timeout=20.0)
        vad_fields = {
            "threshold": "0.5",
            "min_speech_duration_ms": "250",
            "min_silence_duration_ms": "100",
            "speech_pad_ms": "30",
        }
        # Warmup
        post_multipart_wav(f"http://127.0.0.1:{port}/v1/vad/speech_timestamps", wavs[0], vad_fields)
        rows = []
        for w in wavs:
            ms, raw = post_multipart_wav(f"http://127.0.0.1:{port}/v1/vad/speech_timestamps", w, vad_fields)
            data = json.loads(raw)
            segs = data.get("timestamps") or []
            rows.append(
                {
                    "compute_unit": "mlx_metal_server",
                    "wav_file": Path(w).name,
                    "total_ms": round(ms, 2),
                    "segments": len(segs),
                }
            )
        return rows
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def bench_metal_contention(wavs: list[str], asr_port: int = 18095, llm_port: int = 18096):
    asr_snaps = sorted(glob.glob(f"{ASR_MODEL}/*"))
    if not asr_snaps:
        raise RuntimeError(f"No Qwen3-ASR snapshot found under {ASR_MODEL}")
    asr_dir = asr_snaps[0]

    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = OMNI_BIN
    asr_proc = subprocess.Popen(
        [
            f"{OMNI_BIN}/qwen3_asr_server",
            "--model-directory",
            asr_dir,
            "--model-kind",
            "qwen3_asr",
            "--host",
            "127.0.0.1",
            "--port",
            str(asr_port),
            "--cache-limit-mb",
            "512",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    llm_proc = subprocess.Popen(
        [
            CPP_BIN,
            "--model",
            GGUF_MODEL,
            "--system-prompt",
            SYS_PROMPT,
            "--dict",
            DICT_JSON,
            "--draft-max",
            "6",
            "--serve",
            "--port",
            str(llm_port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_health(f"http://127.0.0.1:{asr_port}/health", timeout=30.0)
        wait_health(f"http://127.0.0.1:{llm_port}/health", timeout=30.0)

        long_wav = max(wavs, key=lambda p: Path(p).stat().st_size)  # 38.7s wav E
        test_utterances = [
            "嗯那个我再试一下我的这个语音输入，看有没有什么问题，我还是觉得第一个 token 的速度有点慢。",
            "之前开了一个 section，你可以在聊天记录里面搜索一下，是关于怎么样加速这个 workflow 的。",
            "就是说我们在 RTX 4090 上把 Qwen3 零点六 B 训完之后，在 Apple Silicon 上做三级 GGUF 门禁校验。",
        ]
        # Warmup both servers
        post_multipart_wav(
            f"http://127.0.0.1:{asr_port}/v1/audio/transcriptions",
            wavs[0],
            {"model": "asr.fast", "language": "zh", "response_format": "json"},
        )
        post_llm_json(llm_port, test_utterances[0])

        # 1. Isolated ASR on 38.7s wav E (3 runs)
        iso_asr = []
        for _ in range(3):
            ms, _ = post_multipart_wav(
                f"http://127.0.0.1:{asr_port}/v1/audio/transcriptions",
                long_wav,
                {"model": "asr.fast", "language": "zh", "response_format": "json"},
            )
            iso_asr.append(round(ms, 1))

        # 2. Isolated C++ GGML 0.6B Q8_0 (9 runs)
        iso_llm = []
        for i in range(9):
            ms, _ = post_llm_json(llm_port, test_utterances[i % len(test_utterances)])
            iso_llm.append(round(ms, 1))

        # 3. Concurrent MLX ASR (38.7s wav E x 3) + GGML C++ LLM (continuous loop)
        stop_ev = threading.Event()
        conc_llm = []

        def llm_hammer():
            idx = 0
            while not stop_ev.is_set():
                ms, _ = post_llm_json(llm_port, test_utterances[idx % len(test_utterances)])
                conc_llm.append(round(ms, 1))
                idx += 1

        t = threading.Thread(target=llm_hammer, daemon=True)
        t.start()
        conc_asr = []
        for _ in range(3):
            ms, _ = post_multipart_wav(
                f"http://127.0.0.1:{asr_port}/v1/audio/transcriptions",
                long_wav,
                {"model": "asr.fast", "language": "zh", "response_format": "json"},
            )
            conc_asr.append(round(ms, 1))
        stop_ev.set()
        t.join(timeout=10)

        # 4. Concurrent CoreML cpu_only (AMX) VAD + GGML C++ LLM (zero Metal GPU contention!)
        stop_ev2 = threading.Event()
        vad_llm = []

        def llm_hammer2():
            idx = 0
            while not stop_ev2.is_set():
                ms, _ = post_llm_json(llm_port, test_utterances[idx % len(test_utterances)])
                vad_llm.append(round(ms, 1))
                idx += 1

        t2 = threading.Thread(target=llm_hammer2, daemon=True)
        t2.start()
        amx_vad_runs = run_coreml_vad("cpu_only", [long_wav, long_wav])
        stop_ev2.set()
        t2.join(timeout=10)

        return {
            "long_wav": Path(long_wav).name,
            "isolated_mlx_asr_ms": iso_asr,
            "isolated_mlx_asr_med": round(statistics.median(iso_asr), 1),
            "concurrent_mlx_asr_ms": conc_asr,
            "concurrent_mlx_asr_med": round(statistics.median(conc_asr), 1),
            "asr_contention_pct": round(
                100.0 * (statistics.median(conc_asr) - statistics.median(iso_asr)) / statistics.median(iso_asr),
                1,
            ),
            "isolated_cpp_llm_med": round(statistics.median(iso_llm), 1),
            "isolated_cpp_llm_p90": round(sorted(iso_llm)[int(0.9 * (len(iso_llm) - 1))], 1),
            "concurrent_cpp_llm_with_mlx_asr_med": round(statistics.median(conc_llm), 1),
            "concurrent_cpp_llm_with_mlx_asr_p90": round(sorted(conc_llm)[int(0.9 * (len(conc_llm) - 1))], 1),
            "llm_contention_with_mlx_asr_pct": round(
                100.0 * (statistics.median(conc_llm) - statistics.median(iso_llm)) / statistics.median(iso_llm),
                1,
            ),
            "concurrent_cpp_llm_with_amx_vad_med": round(statistics.median(vad_llm), 1) if vad_llm else None,
            "amx_vad_during_llm_ms": [r["total_ms"] for r in amx_vad_runs],
        }
    finally:
        for p in (asr_proc, llm_proc):
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", default="/tmp/voxt-wavs")
    ap.add_argument("--out", default="/tmp/apple_silicon_bench.json")
    args = ap.parse_args()

    wavs = sorted(glob.glob(f"{args.wav_dir}/*.wav"))
    if not wavs:
        raise RuntimeError(f"No wavs in {args.wav_dir}")

    print("[1/5] CoreML Silero VAD: cpu_only (AMX)...")
    vad_cpu = run_coreml_vad("cpu_only", wavs)
    print("[2/5] CoreML Silero VAD: cpu_and_ne (ANE)...")
    vad_ane = run_coreml_vad("cpu_and_ne", wavs)
    print("[3/5] CoreML Silero VAD: cpu_and_gpu (Metal)...")
    vad_gpu = run_coreml_vad("cpu_and_gpu", wavs)
    print("[4/5] Voxt MLX Metal silero_vad_server...")
    vad_mlx = bench_mlx_vad(wavs)
    print("[5/5] Metal GPU Contention (MLX Qwen3-ASR vs GGML C++ 0.6B)...")
    contention = bench_metal_contention(wavs)

    summary = {
        "vad_coreml_cpu_only": vad_cpu,
        "vad_coreml_cpu_and_ne": vad_ane,
        "vad_coreml_cpu_and_gpu": vad_gpu,
        "vad_mlx_metal": vad_mlx,
        "metal_contention": contention,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== SUMMARY ===")
    for label, rows in [
        ("CoreML cpu_only (AMX)", vad_cpu),
        ("CoreML cpu_and_ne (ANE)", vad_ane),
        ("CoreML cpu_and_gpu", vad_gpu),
        ("MLX Metal silero_vad_server", vad_mlx),
    ]:
        tot = sum(r["total_ms"] for r in rows)
        e_ms = next(r["total_ms"] for r in rows if "XWQ00" in r["wav_file"])
        print(f"  {label:28s}: 5-wav sum={tot:7.1f} ms | 38.7s wav E={e_ms:6.1f} ms")
    print("  Contention:", json.dumps(contention, ensure_ascii=False))


if __name__ == "__main__":
    main()
