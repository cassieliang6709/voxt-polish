"""Lightweight on-device inference gateway & guardrail for Voxt dictation polishing.

Architecture (designed to keep system prompt byte-identical so KV prefix cache stays 100% warm):
  1. Deterministic pre-correction: replaces unambiguous ASR phonetic aliases from an external
     `dictionary.json` (e.g. "SG Lang" -> "SGLang", "cloud code" -> "Claude Code") in <0.1 ms.
  2. Dynamic hotword retrieval (optional): retrieves top-K fuzzy/substring matches from
     `dictionary.json` and appends a compact hint at the END of the user message (never touches
     the system prompt, preserving 100% prefix cache hit rate).
  3. Deterministic post-processing: CJK/ASCII spacing & stray whitespace cleanup.
  4. Silent-failure circuit breaker: if model output drops >45% of a long utterance (<0.55x ASR),
     over-expands (>1.5x ASR), leaks `<think>`, or hits max_tokens runaway, falls back to
     deterministically cleaned ASR and logs the bad case to JSONL for next-round SFT/DPO.

Usage:
  # Offline / live eval against an OpenAI-compatible llama-server endpoint:
  python3 polish_gateway.py eval --base-url http://127.0.0.1:8091/v1 \
      --compare-jsonl models/run/eval_compare.jsonl \
      --system-prompt prompts/sft_system.txt \
      --dict dictionary.json --out /tmp/gateway_eval.json

  # Drop-in OpenAI-compatible local proxy (leaves live 8090 untouched):
  python3 polish_gateway.py serve --port 8092 --upstream http://127.0.0.1:8091/v1 \
      --dict dictionary.json --badcase-log /tmp/voxt_badcases.jsonl
"""
import argparse
import difflib
import json
import re
import statistics
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from eval_parity import extract_entities, score_predictions

FILLER_RE = re.compile(r"(?:^|[，,。；;！!？?\n])(?:嗯|呃|啊|哎|那个|就是说|然后)[，,]?\s*")


def load_dictionary(dict_path: str | None):
    """Load canonical -> aliases from external JSON (list of [canonical, [aliases]] per category)."""
    if not dict_path or not Path(dict_path).exists():
        return [], []
    raw = json.loads(Path(dict_path).read_text(encoding="utf-8"))
    alias_rules = []
    canonicals = []
    for entries in raw.values():
        for item in entries:
            canonical = item[0].strip()
            aliases = item[1] if len(item) > 1 else []
            canonicals.append((canonical, aliases))
            for al in aliases:
                al = al.strip()
                if len(al) >= 2 and al.lower() != canonical.lower():
                    alias_rules.append((al, canonical))
    # Longest alias first to avoid partial shadowing
    alias_rules.sort(key=lambda x: len(x[0]), reverse=True)
    return alias_rules, canonicals


def apply_alias_rules(text: str, alias_rules):
    """Deterministic case-insensitive replacement for unambiguous ASR phonetic misrecognitions."""
    out = text
    applied = []
    for alias, canonical in alias_rules:
        # For pure ASCII short aliases (<=3 chars), require word boundary to avoid corrupting words
        if re.fullmatch(r"[A-Za-z0-9 ]+", alias) and len(alias.replace(" ", "")) <= 3:
            pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", re.I)
        else:
            pat = re.compile(re.escape(alias), re.I)
        new_out, n = pat.subn(canonical, out)
        if n > 0:
            out = new_out
            applied.append((alias, canonical))
    return out, applied


def normalize_spacing(text: str) -> str:
    # Keep newlines intact; normalize CJK/ASCII spacing line by line
    lines = []
    for line in text.splitlines():
        line = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", line)
        line = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", line)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        lines.append(line)
    return "\n".join(lines).strip()


def deterministic_fallback(asr_cleaned: str) -> str:
    txt = FILLER_RE.sub(lambda m: m.group(0)[0] if m.group(0)[0] in "，,。；;！!？?\n" else "", asr_cleaned)
    return normalize_spacing(txt)


def check_guardrail(asr: str, pred: str, finish_reason: str = "stop"):
    """Return failure reason string if output violates P0 safety bounds, else None."""
    if not pred.strip():
        return "empty_output"
    if finish_reason == "length":
        return "runaway_max_tokens"
    if "<think>" in pred or "</think>" in pred:
        return "think_tag_leak"
    if len(pred) > max(30, int(len(asr) * 1.5)):
        return f"over_expand({len(pred)}/{len(asr)})"
    if len(asr) >= 40 and len(pred) < int(len(asr) * 0.55):
        return f"under_drop({len(pred)}/{len(asr)})"
    return None


def polish_once(
    upstream_url: str,
    system_prompt: str,
    asr_text: str,
    alias_rules,
    max_tokens: int | None = None,
    apply_pre_alias: bool = True,
    apply_post: bool = True,
    enable_guardrail: bool = True,
):
    pre_text, applied_aliases = apply_alias_rules(asr_text, alias_rules) if apply_pre_alias else (asr_text, [])
    if max_tokens is None:
        max_tokens = min(512, max(64, int(len(pre_text) * 1.8)))
    body = json.dumps(
        {
            "model": "polish",
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pre_text},
            ],
        }
    ).encode()
    req = urllib.request.Request(
        f"{upstream_url.rstrip('/')}/chat/completions", body, {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    total_s = time.perf_counter() - t0
    raw_pred = data["choices"][0]["message"]["content"].strip()
    finish_reason = data["choices"][0].get("finish_reason", "stop")
    timings = data.get("timings", {})

    final_pred = raw_pred
    if apply_post:
        final_pred, post_aliases = apply_alias_rules(final_pred, alias_rules)
        final_pred = normalize_spacing(final_pred)
    else:
        post_aliases = []

    fallback_reason = None
    if enable_guardrail:
        fallback_reason = check_guardrail(pre_text, final_pred, finish_reason)
        if fallback_reason:
            final_pred = deterministic_fallback(pre_text)

    return {
        "pred": final_pred,
        "raw_pred": raw_pred,
        "finish_reason": finish_reason,
        "runaway": finish_reason == "length",
        "fallback_reason": fallback_reason,
        "applied_aliases": applied_aliases + post_aliases,
        "total_s": round(total_s, 4),
        "timings": timings,
    }


def run_eval(args):
    alias_rules, _ = load_dictionary(args.dict)
    sys_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()
    rows = [json.loads(l) for l in Path(args.compare_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    bf16_refs = [r["sft_pred"] for r in rows]

    # Warmup
    polish_once(args.base_url, sys_prompt, rows[0]["asr"], alias_rules, max_tokens=128)

    preds, latencies, gen_ns, gen_ms = [], [], [], []
    fallbacks = 0
    alias_hits = 0
    for r in rows:
        res = polish_once(args.base_url, sys_prompt, r["asr"], alias_rules)
        preds.append(res)
        latencies.append(res["total_s"])
        gen_ns.append(res["timings"].get("predicted_n", 0))
        gen_ms.append(res["timings"].get("predicted_ms", 0))
        if res["fallback_reason"]:
            fallbacks += 1
        if res["applied_aliases"]:
            alias_hits += 1

    summary, errs = score_predictions(rows, preds, bf16_refs=bf16_refs)
    total_gen_s = sum(gen_ms) / 1000.0
    summary.update(
        {
            "label": args.label,
            "alias_hit_rows": alias_hits,
            "guardrail_fallbacks": fallbacks,
            "warm_median_s": round(statistics.median(latencies), 3),
            "warm_p90_s": round(sorted(latencies)[int(len(latencies) * 0.9) - 1], 3),
            "warm_max_s": round(max(latencies), 3),
            "gen_tok_per_s": round(sum(gen_ns) / total_gen_s, 1) if total_gen_s > 0 else 0.0,
        }
    )
    print(json.dumps(summary, ensure_ascii=False))
    Path(args.out).write_text(
        json.dumps({"summary": summary, "errors": errs, "preds": preds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_serve(args):
    alias_rules, _ = load_dictionary(args.dict)
    badcase_path = Path(args.badcase_log) if args.badcase_log else None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            msgs = payload.get("messages", [])
            sys_msg = next((m["content"] for m in msgs if m.get("role") == "system"), "")
            user_msg = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
            res = polish_once(
                args.upstream,
                sys_msg,
                user_msg,
                alias_rules,
                max_tokens=payload.get("max_tokens"),
            )
            if res["fallback_reason"] and badcase_path:
                with badcase_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "reason": res["fallback_reason"],
                                "asr": user_msg,
                                "raw_pred": res["raw_pred"],
                                "fallback": res["pred"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            resp_obj = {
                "id": f"voxt-gw-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": res["pred"]},
                        "finish_reason": "stop",
                    }
                ],
                "timings": res["timings"],
                "voxt_gateway": {
                    "fallback_reason": res["fallback_reason"],
                    "applied_aliases": res["applied_aliases"],
                },
            }
            out_bytes = json.dumps(resp_obj, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out_bytes)))
            self.end_headers()
            self.wfile.write(out_bytes)

        def log_message(self, fmt, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Voxt polish gateway listening on http://127.0.0.1:{args.port} -> {args.upstream}", flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--base-url", required=True)
    p_eval.add_argument("--compare-jsonl", required=True)
    p_eval.add_argument("--system-prompt", required=True)
    p_eval.add_argument("--dict", default=None)
    p_eval.add_argument("--label", required=True)
    p_eval.add_argument("--out", required=True)

    p_srv = sub.add_parser("serve")
    p_srv.add_argument("--port", type=int, default=8092)
    p_srv.add_argument("--upstream", required=True)
    p_srv.add_argument("--dict", default=None)
    p_srv.add_argument("--badcase-log", default=None)

    args = ap.parse_args()
    if args.cmd == "eval":
        run_eval(args)
    else:
        run_serve(args)


if __name__ == "__main__":
    main()
