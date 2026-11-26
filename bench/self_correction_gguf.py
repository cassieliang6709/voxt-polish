"""Score a GGUF polish model on examples/self_correction_eval.jsonl through an
OpenAI-compatible server (same request as eval_parity.py: system prompt + raw
ASR text, temperature 0). A case passes when its output contains one of
must_have and none of must_not_have."""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--system-prompt", required=True)
ap.add_argument("--eval", default="examples/self_correction_eval.jsonl")
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
system = open(a.system_prompt, encoding="utf-8").read().strip()
rows, passed = [], 0
for line in open(a.eval, encoding="utf-8"):
    case = json.loads(line)
    body = json.dumps({"model": "polish", "temperature": 0, "max_tokens": 256,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": case["asr"]}]}).encode()
    req = urllib.request.Request(a.base_url + "/chat/completions", body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"].strip()
    ok = any(m in out for m in case["must_have"]) and not any(m in out for m in case["must_not_have"])
    passed += ok
    rows.append({"id": case["id"], "pass": ok, "pred": out})
    print(f'{case["id"]} {"PASS" if ok else "FAIL"} | {out}')
print(f"{a.label} SELF_CORRECTION_SCORE: {passed}/{len(rows)}")
json.dump({"label": a.label, "passed": passed, "n": len(rows), "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)
