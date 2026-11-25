"""Benchmark a polish server the way Voxt uses it.

Voxt sends a long fixed system prompt (~800 tokens) plus one short dictation.
After the first request the system prompt is cached, so what matters is:
  - cold: first request, full prefill
  - warm: later requests, only the new dictation is prefilled, then decode
Usage: python3 voxt_like.py <base_url> <label> [rounds]
"""

import json
import pathlib
import statistics
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).parent
BASE = (HERE / "polish_prompt.txt").read_text(encoding="utf-8")
# Pad to roughly Voxt's prompt length (~800 tokens).
SYSTEM = BASE + "\n补充说明：" + BASE + "\n再次强调：" + BASE

DICTATIONS = [
    "嗯那个我今天想就是说把那个 SGLang 的 radix cache 呃不对是 radix attention 的那个 PR 再看一下然后明天跟那个 mentor 约个时间聊一下",
    "我觉得这个 landing page 的那个 hero section 嗯太空了然后那个 CTA 按钮颜色也不太对要改成那个品牌色",
    "明天上午十点不对是十一点要跟面试官 mock interview 然后下午去健身房",
    "这个 bug 是因为那个 useEffect 的依赖数组里少了 userId 所以切换用户的时候数据没刷新",
    "我再试一下我的这个语音输入我看有没有什么问题我还是觉得它第一个 token 的速度有点慢",
    "还有就是我发现它有的时候比如说失败了以后它会错误我希望它错误了以后能够自动保存到我的剪贴板里",
]


def call(base_url, text):
    body = json.dumps(
        {
            "model": "polish",
            "temperature": 0,
            "max_tokens": 160,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": text},
            ],
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", body, {"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    total = time.perf_counter() - start
    t = data.get("timings", {})
    return {
        "total": total,
        "prompt_n": t.get("prompt_n", 0),
        "prompt_ms": t.get("prompt_ms", 0),
        "gen_n": t.get("predicted_n", 0),
        "gen_ms": t.get("predicted_ms", 0),
        "text": data["choices"][0]["message"]["content"].strip(),
    }


def main():
    base_url, label = sys.argv[1], sys.argv[2]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    cold = call(base_url, DICTATIONS[0])
    warm = [call(base_url, d) for _ in range(rounds) for d in DICTATIONS]
    gen_tps = sum(r["gen_n"] for r in warm) / (sum(r["gen_ms"] for r in warm) / 1000)
    summary = {
        "label": label,
        "cold_total": round(cold["total"], 2),
        "cold_prefill_tps": round(cold["prompt_n"] / (cold["prompt_ms"] / 1000), 1),
        "warm_median": round(statistics.median(r["total"] for r in warm), 3),
        "warm_p90": round(sorted(r["total"] for r in warm)[int(len(warm) * 0.9) - 1], 3),
        "warm_prefill_ms": round(statistics.median(r["prompt_ms"] for r in warm), 1),
        "warm_gen_ms": round(statistics.median(r["gen_ms"] for r in warm), 1),
        "gen_tok_per_s": round(gen_tps, 1),
    }
    print(json.dumps(summary, ensure_ascii=False))
    out = HERE / "results.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**summary, "outputs": [r["text"] for r in warm[: len(DICTATIONS)]]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
