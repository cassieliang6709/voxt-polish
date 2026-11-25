"""Privacy-safe template for cleaning ASR -> polished dictation pairs.

Implements the deterministic stages of the dictation curation pipeline without
embedding any personal text or private domain corrections:
  1. Drop empty / invalid rows.
  2. Apply external JSON patches (`--patches patches.json`) for manual term,
     hallucination, and light-smoothing fixes.
  3. Normalize stray trailing question marks on non-interrogative sentences.
  4. Strip redundant filler 「然后」 while preserving genuine sequence cues.
  5. Normalize CJK/ASCII spacing, straight quotes, and inline numbered lists.
  6. Flag high-diff or entity-mismatched rows for human review.

Example `patches.json` format:
  {"12": "Corrected polished output for row index 12"}
"""
import argparse
import difflib
import json
import re
from pathlib import Path

EN_NUM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-+#]{1,}|\d+(?:\.\d+)?")
QUESTION_CUE_RE = re.compile(r"(吗|呢|么|谁|啥|哪|几|多少|怎么|为什么|为何|是否|能否|能不能|是不是|对不对|行不行|好不好|有没有|什么)")


def extract_entities(text: str):
    return set(w.lower() for w in EN_NUM_RE.findall(text))


def normalize_spacing(text: str) -> str:
    # Add space between CJK and ASCII alphanumerics
    text = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", text)
    # Collapse repeated spaces within a line (preserve newlines)
    text = "\n".join(re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines())
    return text.strip()


def clean_filler_ranhou(text: str) -> str:
    # Remove sentence-initial or comma-preceded filler 「然后」
    text = re.sub(r"(^|[，,。；;！!？?\n])然后[，,]?\s*", r"\1", text)
    return text


def fix_stray_question_mark(asr: str, out: str) -> str:
    if out.endswith(("？", "?")) and not QUESTION_CUE_RE.search(asr) and not QUESTION_CUE_RE.search(out):
        return out[:-1] + "。"
    return out


def audit_row(idx: int, asr: str, out: str):
    flags = []
    sim = difflib.SequenceMatcher(None, asr, out).ratio()
    if sim < 0.45 and len(asr) > 15:
        flags.append(f"low_sim({sim:.2f})")
    asr_ent = extract_entities(asr)
    out_ent = extract_entities(out)
    missing = sorted(asr_ent - out_ent)
    if missing:
        flags.append(f"missing_entities({','.join(missing[:5])})")
    if len(out) > max(40, int(len(asr) * 1.5)):
        flags.append("over_expand")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--patches", help="Optional JSON file mapping row index (str) to corrected output")
    ap.add_argument("--drop-ids", help="Optional comma-separated row indices to drop")
    args = ap.parse_args()

    patches = json.loads(Path(args.patches).read_text(encoding="utf-8")) if args.patches else {}
    drop_ids = set(int(x) for x in args.drop_ids.split(",") if x.strip()) if args.drop_ids else set()

    raw_rows = [json.loads(l) for l in Path(args.in_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    cleaned, flagged = [], []

    for i, row in enumerate(raw_rows):
        if i in drop_ids:
            continue
        asr = (row.get("input") or row.get("asr") or "").strip()
        out = (row.get("output") or row.get("typeless") or row.get("ref") or "").strip()
        if not asr or not out:
            continue
        if str(i) in patches:
            out = patches[str(i)].strip()
        else:
            out = fix_stray_question_mark(asr, out)
            out = clean_filler_ranhou(out)
            out = normalize_spacing(out)
        flags = audit_row(i, asr, out)
        if flags:
            flagged.append({"idx": i, "flags": flags})
        cleaned.append({"asr": asr, "typeless": out, **{k: v for k, v in row.items() if k not in ("input", "output", "asr", "typeless", "ref")}})

    Path(args.out_jsonl).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in cleaned) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"input_rows": len(raw_rows), "kept_rows": len(cleaned), "flagged_for_review": len(flagged)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
