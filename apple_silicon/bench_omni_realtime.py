"""Zero-dependency benchmark for sglang-omni qwen3_asr_server:
1. Offline Final benchmark: Plain Qwen3-ASR-0.6B-4bit vs DFlash (adaptive/always, bits=0/4/8)
   + exact transcript parity + concurrent Metal GPU contention with voxt_polish_cpp.
2. Live WebSocket /v1/realtime + Final benchmark:
   - final_prompt (30 dictionary terms) ON vs OFF
   - Measures Final latency, KV reuse (via server stderr/stats), transcript parity & echo.
"""
import argparse
import base64
import difflib
import glob
import hashlib
import json
import os
import socket
import statistics
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path

HOME = Path.home()
OMNI_BIN = os.environ.get("VOXT_OMNI_BIN", "/Applications/Voxt.app/Contents/Resources/omni/bin")
OMNI_MODELS = HOME / "Library/Application Support/SGLang Omni/models"
ASR_SNAP = os.environ.get(
    "VOXT_ASR_SNAP",
    str(OMNI_MODELS / "models--mlx-community--Qwen3-ASR-0.6B-4bit/snapshots/313d850181767edf09f00a9c289becca70e58cd0"),
)
DFLASH_SNAP = os.environ.get(
    "VOXT_DFLASH_SNAP",
    str(OMNI_MODELS / "models--wilsonzheng0327--Qwen3-ASR-0.6B-dflash/snapshots/75da55f3170c28ce70379b96921c1cf9a331d008"),
)
REPO_ROOT = Path(__file__).resolve().parent.parent
CPP_BIN = os.environ.get("VOXT_CPP_BIN", str(REPO_ROOT / "cpp/build/voxt_polish_cpp"))
GGUF_MODEL = os.environ.get(
    "VOXT_GGUF_MODEL",
    str(REPO_ROOT / "models/qwen3-0.6b-run3-best/gguf_bench/qwen3-0.6b-polish-q8_0.gguf"),
)
SYS_PROMPT = os.environ.get("VOXT_SYS_PROMPT", str(REPO_ROOT / "prompts/sft_system.txt"))
DICT_JSON = os.environ.get("VOXT_DICT_JSON", str(REPO_ROOT / "examples/sample_dictionary.json"))

SAMPLE_DICT_CONTEXT = (
    "Voxt, Utter, SGLang, sglang-omni, llama.cpp, MLX, Metal, CoreML, Silero VAD, "
    "Qwen3-ASR, DFlash, GGUF, Q8_0, Q4_0, bf16, KV Cache, Prefix Cache, "
    "ngram-simple, Obsidian, Triton, CUDA, FlashAttention, PyTorch, GitHub, "
    "Apple Silicon, M1 Pro, AMX, ANE, GRPO, DPO"
)


def wait_health(url: str, timeout: float = 40.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.15)
    raise RuntimeError(f"Server did not become healthy: {url}")


def post_transcribe(port: int, wav_path: str, extra_fields: dict[str, str] | None = None):
    boundary = "----VoxtOmniBenchBoundary9X2K"
    fields = {
        "model": "asr.fast",
        "language": "zh",
        "stream": "true",
        "response_format": "json",
        "stop_at_end_of_text": "true",
        "stop_on_token_loop": "true",
        "include_generation_metadata": "true",
        "audio_layout": "voxt_swift",
    }
    if extra_fields:
        fields.update(extra_fields)
    parts = []
    for k, v in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        )
    wav_bytes = Path(wav_path).read_bytes()
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{Path(wav_path).name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode()
        + wav_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8", errors="replace")
    ms = (time.perf_counter() - t0) * 1000.0
    # Parse SSE stream (data: {...}) or plain JSON fallback
    done_obj = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    ev = json.loads(payload)
                    if ev.get("type") == "transcript.text.done" or "text" in ev:
                        done_obj = ev
                except json.JSONDecodeError:
                    pass
    if not done_obj and raw.strip().startswith("{"):
        done_obj = json.loads(raw)
    return ms, done_obj


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


class MinimalWS:
    """Minimal RFC 6455 loopback WebSocket client using stdlib socket."""

    def __init__(self, host: str, port: int, path_query: str):
        self.sock = socket.create_connection((host, port), timeout=30.0)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {path_query} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("WS handshake closed")
            buf += chunk
        header, rest = buf.split(b"\r\n\r\n", 1)
        if b"101" not in header.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"WS upgrade failed: {header!r}")
        self.buf = bytearray(rest)

    def _read_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self.buf)))
            if not chunk:
                raise EOFError("WS closed")
            self.buf.extend(chunk)
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def send_text(self, text: str):
        payload = text.encode("utf-8")
        mask = os.urandom(4)
        header = bytearray([0x81])  # FIN + text
        ln = len(payload)
        if ln < 126:
            header.append(0x80 | ln)
        elif ln < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", ln))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", ln))
        header.extend(mask)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def recv_json(self, timeout: float = 30.0) -> dict | None:
        self.sock.settimeout(timeout)
        while True:
            b1, b2 = self._read_exact(2)
            opcode = b1 & 0x0F
            masked = (b2 & 0x80) != 0
            ln = b2 & 0x7F
            if ln == 126:
                ln = struct.unpack("!H", self._read_exact(2))[0]
            elif ln == 127:
                ln = struct.unpack("!Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else None
            payload = self._read_exact(ln)
            if mask:
                payload = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
            if opcode == 0x8:  # close
                return None
            if opcode == 0x9:  # ping -> pong
                m = os.urandom(4)
                h = bytearray([0x8A, 0x80 | len(payload)]) + m
                self.sock.sendall(bytes(h) + bytes(b ^ m[i & 3] for i, b in enumerate(payload)))
                continue
            if opcode == 0x1:
                return json.loads(payload.decode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def start_asr_server(port: int, extra_args: list[str] | None = None):
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = OMNI_BIN
    cmd = [
        f"{OMNI_BIN}/qwen3_asr_server",
        "--model-directory",
        ASR_SNAP,
        "--model-kind",
        "qwen3_asr",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--cache-limit-mb",
        "512",
    ]
    if extra_args:
        cmd.extend(extra_args)
    log_path = f"/tmp/omni_bench_{port}.log"
    log_f = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env)
    wait_health(f"http://127.0.0.1:{port}/health", timeout=40.0)
    return proc, log_f, log_path


def stop_proc(proc, log_f=None):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    if log_f:
        log_f.close()


def read_wav_pcm16(wav_path: str) -> bytes:
    with wave.open(wav_path, "rb") as wf:
        return wf.readframes(wf.getnframes())


def run_realtime_final_replay(port: int, wav_path: str, final_prompt: str | None, pace_ratio: float = 0.08):
    """Stream WAV chunks over /v1/realtime (accelerated pacing while still letting live refreshes fire),
    then issue HTTP Final with session_id to measure KV reuse and Final latency.
    """
    pcm = read_wav_pcm16(wav_path)
    q = urllib.parse.urlencode({"intent": "transcription", "model": "asr.fast"})
    ws = MinimalWS("127.0.0.1", port, f"/v1/realtime?{q}")
    session_id = ""
    partials = []
    stop_reader = threading.Event()
    got_first_partial = threading.Event()

    def reader():
        while not stop_reader.is_set():
            try:
                ev = ws.recv_json(timeout=0.5)
                if ev is None:
                    break
                if ev.get("type") == "transcription_session.updated":
                    pass
                elif ev.get("type") in ("transcription.delta", "conversation.item.input_audio_transcription.delta", "transcription.partial", "transcript"):
                    partials.append(ev)
                    got_first_partial.set()
                else:
                    if ev.get("text"):
                        partials.append(ev)
                        got_first_partial.set()
            except socket.timeout:
                continue
            except Exception:
                break

    try:
        sess = {"input_audio_format": "pcm16", "turn_detection": None, "language": "zh"}
        if final_prompt is not None:
            sess["final_prompt"] = final_prompt
        ws.send_text(json.dumps({"type": "session.update", "session": sess}))
        session_id = ""
        for _ in range(10):
            ev = ws.recv_json(timeout=10.0)
            if ev is None:
                break
            if ev.get("type") == "transcription_session.updated":
                session_id = ev.get("session_id", "")
                break

        t_reader = threading.Thread(target=reader, daemon=True)
        t_reader.start()

        # Stream in 1.0s chunks, giving the server enough time between chunks to complete live refreshes & background final prefill
        chunk_bytes = 16000 * 2  # 1.0s
        for off in range(0, len(pcm), chunk_bytes):
            chunk = pcm[off : off + chunk_bytes]
            ws.send_text(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )
            time.sleep(max(0.18, pace_ratio))
        # Brief settle so last live refresh / background prefill completes before Stop
        time.sleep(0.55)

        last_partial = partials[-1].get("text", "") if partials else ""
        extra = {"realtime_session": session_id}
        if last_partial:
            extra["guessed_text"] = last_partial
        if final_prompt:
            extra["prompt"] = final_prompt
        final_ms, resp = post_transcribe(port, wav_path, extra)
        stop_reader.set()
        ws.close()
        t_reader.join(timeout=2.0)
        meta = resp.get("generation_metadata") or {}
        return {
            "wav": Path(wav_path).name,
            "session_id": session_id,
            "final_prompt_on": bool(final_prompt),
            "partials_received": len(partials),
            "guessed_len": len(last_partial),
            "final_ms": round(final_ms, 1),
            "text": resp.get("text", ""),
            "reused_audio_tokens": meta.get("reused_audio_token_count", 0),
            "reused_prompt_tokens": meta.get("reused_prompt_token_count", 0),
            "output_tokens": meta.get("output_token_count"),
            "draft_accepted": meta.get("draft_accepted_token_count"),
            "generation_metadata": meta,
        }
    finally:
        stop_reader.set()
        ws.close()


def bench_asr_mode(label: str, wavs: list[str], extra_args: list[str], port: int = 18105, runs: int = 3):
    proc, log_f, log_path = start_asr_server(port, extra_args)
    try:
        # Warmup
        post_transcribe(port, wavs[0])
        rows = []
        for w in wavs:
            times = []
            text = ""
            meta = {}
            for _ in range(runs):
                ms, resp = post_transcribe(port, w)
                times.append(round(ms, 1))
                text = resp.get("text", "")
                meta = resp.get("generation_metadata") or {}
            rows.append(
                {
                    "wav": Path(w).name,
                    "times_ms": times,
                    "med_ms": round(statistics.median(times), 1),
                    "text": text,
                    "meta": meta,
                }
            )
        return {"label": label, "extra_args": extra_args, "rows": rows}
    finally:
        stop_proc(proc, log_f)


def bench_contention_with_asr(label: str, wavs: list[str], extra_args: list[str], asr_port: int = 18105, llm_port: int = 18106):
    asr_proc, log_f, _ = start_asr_server(asr_port, extra_args)
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
        wait_health(f"http://127.0.0.1:{llm_port}/health", timeout=30.0)
        long_wav = max(wavs, key=lambda p: Path(p).stat().st_size)
        utts = [
            "嗯那个我再试一下我的这个语音输入，看有没有什么问题，我还是觉得第一个 token 的速度有点慢。",
            "之前开了一个 section，你可以在聊天记录里面搜索一下，是关于怎么样加速这个 workflow 的。",
            "就是说我们在 RTX 4090 上把 Qwen3 零点六 B 训完之后，在 Apple Silicon 上做三级 GGUF 门禁校验。",
        ]
        post_transcribe(asr_port, wavs[0])
        post_llm_json(llm_port, utts[0])

        iso_asr = [round(post_transcribe(asr_port, long_wav)[0], 1) for _ in range(3)]
        iso_llm = [round(post_llm_json(llm_port, utts[i % len(utts)])[0], 1) for i in range(9)]

        stop_ev = threading.Event()
        conc_llm = []

        def hammer():
            i = 0
            while not stop_ev.is_set():
                conc_llm.append(round(post_llm_json(llm_port, utts[i % len(utts)])[0], 1))
                i += 1

        t = threading.Thread(target=hammer, daemon=True)
        t.start()
        conc_asr = [round(post_transcribe(asr_port, long_wav)[0], 1) for _ in range(3)]
        stop_ev.set()
        t.join(timeout=10)
        return {
            "label": label,
            "iso_asr_med": round(statistics.median(iso_asr), 1),
            "conc_asr_med": round(statistics.median(conc_asr), 1),
            "asr_slowdown_pct": round(100.0 * (statistics.median(conc_asr) - statistics.median(iso_asr)) / statistics.median(iso_asr), 1),
            "iso_llm_med": round(statistics.median(iso_llm), 1),
            "conc_llm_med": round(statistics.median(conc_llm), 1),
            "llm_slowdown_pct": round(100.0 * (statistics.median(conc_llm) - statistics.median(iso_llm)) / statistics.median(iso_llm), 1),
        }
    finally:
        stop_proc(asr_proc, log_f)
        stop_proc(llm_proc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", default="/tmp/voxt-wavs")
    ap.add_argument("--mode", choices=["realtime_kv", "dflash", "all"], default="all")
    ap.add_argument("--out", default="/tmp/omni_optimizations_bench.json")
    args = ap.parse_args()
    wavs = sorted(glob.glob(f"{args.wav_dir}/*.wav"))
    out = {}

    if args.mode in ("realtime_kv", "all"):
        print("=== [A] Realtime /v1/realtime + Final KV Reuse (Dictionary Context ON vs OFF) ===")
        proc, log_f, log_path = start_asr_server(18105, [])
        try:
            post_transcribe(18105, wavs[0])  # warmup
            kv_rows = []
            for w in wavs:
                off_res = run_realtime_final_replay(18105, w, final_prompt="")
                on_res = run_realtime_final_replay(18105, w, final_prompt=SAMPLE_DICT_CONTEXT)
                kv_rows.append({"off": off_res, "on": on_res})
                print(
                    f"  {Path(w).name}: final_prompt=OFF {off_res['final_ms']:6.1f} ms | "
                    f"ON {on_res['final_ms']:6.1f} ms (delta={on_res['final_ms'] - off_res['final_ms']:+.1f} ms)"
                )
            out["realtime_kv"] = kv_rows
        finally:
            stop_proc(proc, log_f)

    if args.mode in ("dflash", "all"):
        if not Path(DFLASH_SNAP, "model.safetensors").exists():
            print(f"SKIP DFlash: {DFLASH_SNAP}/model.safetensors not present yet")
        else:
            print("=== [B] Qwen3-ASR-0.6B Plain vs DFlash Speculative Decoding ===")
            plain = bench_asr_mode("plain", wavs, [])
            base_texts = {r["wav"]: r["text"] for r in plain["rows"]}
            configs = [
                ("dflash_bf16_adaptive", ["--draft-model-path", DFLASH_SNAP, "--draft-policy", "adaptive"]),
                ("dflash_bf16_fixed", ["--draft-model-path", DFLASH_SNAP, "--draft-policy", "fixed"]),
                ("dflash_8bit_fixed", ["--draft-model-path", DFLASH_SNAP, "--draft-bits", "8", "--draft-policy", "fixed"]),
                ("dflash_4bit_fixed", ["--draft-model-path", DFLASH_SNAP, "--draft-bits", "4", "--draft-policy", "fixed"]),
                ("dflash_4bit_adaptive", ["--draft-model-path", DFLASH_SNAP, "--draft-bits", "4", "--draft-policy", "adaptive"]),
            ]
            dflash_results = {"plain": plain}
            for label, eargs in configs:
                res = bench_asr_mode(label, wavs, eargs)
                for r in res["rows"]:
                    ref = base_texts[r["wav"]]
                    r["exact_match"] = r["text"] == ref
                    r["char_sim"] = round(difflib.SequenceMatcher(None, ref, r["text"]).ratio(), 4)
                dflash_results[label] = res
                tot_plain = sum(r["med_ms"] for r in plain["rows"])
                tot_cur = sum(r["med_ms"] for r in res["rows"])
                exact = sum(1 for r in res["rows"] if r["exact_match"])
                print(
                    f"  {label:22s}: 5-wav sum={tot_cur:6.1f} ms (vs {tot_plain:6.1f} ms, "
                    f"speedup={tot_plain/tot_cur:.2f}x) | exact={exact}/{len(wavs)}"
                )
            out["dflash"] = dflash_results

            print("=== [C] Contention: Plain ASR vs DFlash ASR concurrent with voxt_polish_cpp ===")
            out["contention_plain"] = bench_contention_with_asr("plain", wavs, [])
            out["contention_dflash"] = bench_contention_with_asr(
                "dflash_4bit_fixed", wavs, ["--draft-model-path", DFLASH_SNAP, "--draft-bits", "4", "--draft-policy", "fixed"]
            )
            print("  Plain contention:", json.dumps(out["contention_plain"]))
            print("  DFlash contention:", json.dumps(out["contention_dflash"]))

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
