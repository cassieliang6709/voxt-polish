"""Tiered evaluation & behavioral train-inference parity gate.

Evaluates any OpenAI-compatible local llama-server endpoint (F16 / Q8_0 / Q4_0, ngram ON/OFF)
against the exact same chronological 40-pair holdout set and 4090 bf16 reference predictions.

Metrics:
- P0 Entity Retention: recall of English technical terms (>=2 chars) & numbers from reference
- P0 Zero-defect checks: runaways, over_expands (>1.5x ASR len), think_leaks
- P1 Quality & Behavioral Parity: mean/median char similarity vs cleaned ground truth AND vs 4090 bf16 ref
- Latency: warm median / p90 total latency and generation tok/s
"""
import argparse
import difflib
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

EN_NUM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-+#]{1,}|\d+(?:\.\d+)?")


def extract_entities(text: str):
    return set(w.lower() for w in EN_NUM_RE.findall(text))


def entity_recall(ref: str, pred: str):
    ref_e = extract_entities(ref)
    if not ref_e:
        return 1.0, []
    pred_e = extract_entities(pred)
    missing = sorted(ref_e - pred_e)
    return (len(ref_e) - len(missing)) / len(ref_e), missing


def call_server(base_url: str, system_prompt: str, text: str, max_tokens: int):
    body = json.dumps(
        {
            "model": "polish",
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", body, {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    total_s = time.perf_counter() - t0
    msg = data["choices"][0]["message"]["content"].strip()
    finish_reason = data["choices"][0].get("finish_reason", "")
    timings = data.get("timings", {})
    return msg, finish_reason, total_s, timings


def score_predictions(rows, preds, bf16_refs=None):
    sims_gt, sims_bf16, ent_recalls = [], [], []
    entity_error_cases, preserve_error_cases = [], []
    pres_ok = pres_tot = 0
    runaways = over_expands = under_drops = think_leaks = 0
    for i, (r, p) in enumerate(zip(rows, preds)):
        pred = p["pred"]
        ref = r["ref"]
        asr = r["asr"]
        sim_gt = difflib.SequenceMatcher(None, pred, ref).ratio()
        sims_gt.append(sim_gt)
        rec, missing = entity_recall(ref, pred)
        ent_recalls.append(rec)
        if missing:
            entity_error_cases.append({"i": i, "missing": missing, "ref": ref[:120], "pred": pred[:120]})
        keep_e = extract_entities(asr) & extract_entities(ref)
        pred_e = extract_entities(pred)
        miss_keep = sorted(keep_e - pred_e)
        pres_tot += len(keep_e)
        pres_ok += len(keep_e) - len(miss_keep)
        if miss_keep:
            preserve_error_cases.append({"i": i, "missing_from_asr": miss_keep})
        if p.get("runaway"):
            runaways += 1
        if len(pred) > max(30, int(len(asr) * 1.5)):
            over_expands += 1
        if len(asr) >= 40 and len(pred) < int(len(asr) * 0.55):
            under_drops += 1
        if "<think>" in pred or "</think>" in pred:
            think_leaks += 1
        if bf16_refs is not None:
            sims_bf16.append(difflib.SequenceMatcher(None, pred, bf16_refs[i]).ratio())
    out = {
        "n": len(rows),
        "mean_char_sim_gt": round(sum(sims_gt) / len(sims_gt), 4),
        "median_char_sim_gt": round(statistics.median(sims_gt), 4),
        "p0_preserve_recall": round(pres_ok / pres_tot, 4) if pres_tot else 1.0,
        "p0_preserve_error_rows": len(preserve_error_cases),
        "p0_entity_recall_vs_ref": round(sum(ent_recalls) / len(ent_recalls), 4),
        "runaways": runaways,
        "over_expands": over_expands,
        "under_drops": under_drops,
        "think_leaks": think_leaks,
    }
    if bf16_refs is not None:
        out["parity_mean_sim_vs_bf16"] = round(sum(sims_bf16) / len(sims_bf16), 4)
        out["parity_exact_match_vs_bf16"] = sum(1 for a, b in zip(preds, bf16_refs) if a["pred"] == b)
    return out, {"preserve_errors": preserve_error_cases, "ref_entity_errors": entity_error_cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8091/v1")
    ap.add_argument("--compare-jsonl", required=True, help="4090 eval_compare.jsonl (contains asr, ref, sft_pred)")
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()
    rows = [json.loads(l) for l in Path(args.compare_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    bf16_refs = [r["sft_pred"] for r in rows]

    # Warmup call (fills system prompt KV cache)
    call_server(args.base_url, sys_prompt, rows[0]["asr"], 128)

    preds, latencies, gen_ns, gen_ms_list = [], [], [], []
    for r in rows:
        max_new = min(512, max(64, int(len(r["asr"]) * 1.8)))
        pred, finish, total_s, timings = call_server(args.base_url, sys_prompt, r["asr"], max_new)
        preds.append({"pred": pred, "runaway": finish == "length", "total_s": round(total_s, 4)})
        latencies.append(total_s)
        gen_ns.append(timings.get("predicted_n", 0))
        gen_ms_list.append(timings.get("predicted_ms", 0))

    summary, ent_errors = score_predictions(rows, preds, bf16_refs=bf16_refs)
    total_gen_s = sum(gen_ms_list) / 1000.0
    summary.update(
        {
            "label": args.label,
            "warm_median_s": round(statistics.median(latencies), 3),
            "warm_p90_s": round(sorted(latencies)[int(len(latencies) * 0.9) - 1], 3),
            "warm_max_s": round(max(latencies), 3),
            "gen_tok_per_s": round(sum(gen_ns) / total_gen_s, 1) if total_gen_s > 0 else 0.0,
        }
    )
    print(json.dumps(summary, ensure_ascii=False))
    Path(args.out).write_text(
        json.dumps({"summary": summary, "entity_errors": ent_errors, "preds": preds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
