"""Full-parameter bf16 SFT of Qwen3-0.6B on real Typeless dictation pairs + chronological holdout eval.

Runs on a single RTX 4090 with zero external training frameworks (pure PyTorch + transformers).
"""
import argparse
import difflib
import json
import math
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EN_NUM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-+#]{1,}|\d+(?:\.\d+)?")
RANHOU_RE = re.compile(r"(?:^|[，,。；;！!？?\n])然后")


def extract_entities(text: str):
    return set(w.lower() for w in EN_NUM_RE.findall(text))

DEFAULT_SYSTEM_PROMPT = (
    "你是语音输入文字整理器。把用户的语音转写原文整理为通顺书面语："
    "删口头禅、重复和改口前的内容，补全标点，保留原意和英文术语。"
    "只输出整理后的文字，不要回答问题、不要解释。"
)
_PROMPT_FILE = Path(__file__).parent / "prompts" / "sft_system.txt"
SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8").strip() if _PROMPT_FILE.exists() else DEFAULT_SYSTEM_PROMPT


def load_and_split(pairs_path: str, val_size: int = 40):
    raw = [json.loads(line) for line in Path(pairs_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    cleaned = []
    dropped = 0
    for r in raw:
        asr = (r.get("asr") or "").strip()
        typ = (r.get("typeless") or "").strip()
        if not asr or not typ:
            dropped += 1
            continue
        ratio = len(typ) / max(1, len(asr))
        if ratio > 2.0 or ratio < 0.3:
            dropped += 1
            continue
        cleaned.append({"id": r.get("id", ""), "created_at": r.get("created_at", ""), "asr": asr, "typeless": typ})
    # Chronological split: earliest -> train, latest val_size -> holdout
    cleaned.sort(key=lambda x: x["created_at"])
    train_rows = cleaned[:-val_size]
    val_rows = cleaned[-val_size:]
    return train_rows, val_rows, dropped


def build_example(tokenizer, asr: str, typeless: str, max_len: int = 1024):
    prompt_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": asr},
    ]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": typeless}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_len:
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {"input_ids": full_ids, "labels": labels}


@torch.no_grad()
def evaluate(model, tokenizer, val_rows, label: str):
    model.eval()
    preds = []
    sims = []
    runaways = 0
    over_expands = 0
    under_drops = 0
    think_leaks = 0
    filler_ranhou = 0
    pres_ok = 0
    pres_tot = 0
    preserve_error_rows = 0
    t0 = time.perf_counter()
    for r in val_rows:
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["asr"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        max_new = min(512, max(64, int(len(r["asr"]) * 1.8)))
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        hit_cap = len(gen_ids) >= max_new and (gen_ids[-1].item() != tokenizer.eos_token_id)
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if hit_cap:
            runaways += 1
        if "<think>" in pred or "</think>" in pred:
            think_leaks += 1
        if len(pred) > max(30, int(len(r["asr"]) * 1.5)):
            over_expands += 1
        if len(r["asr"]) >= 40 and len(pred) < int(len(r["asr"]) * 0.55):
            under_drops += 1
        filler_ranhou += len(RANHOU_RE.findall(pred))
        keep_e = extract_entities(r["asr"]) & extract_entities(r["typeless"])
        pred_e = extract_entities(pred)
        miss_e = sorted(keep_e - pred_e)
        pres_tot += len(keep_e)
        pres_ok += len(keep_e) - len(miss_e)
        if miss_e:
            preserve_error_rows += 1
        sim = difflib.SequenceMatcher(None, pred, r["typeless"]).ratio()
        sims.append(sim)
        preds.append(
            {
                "id": r["id"],
                "asr": r["asr"],
                "ref": r["typeless"],
                "pred": pred,
                "sim": round(sim, 4),
                "runaway": hit_cap,
                "missing_asr_entities": miss_e,
            }
        )
    elapsed = time.perf_counter() - t0
    summary = {
        "label": label,
        "n": len(val_rows),
        "mean_char_sim": round(sum(sims) / len(sims), 4),
        "median_char_sim": round(sorted(sims)[len(sims) // 2], 4),
        "p0_preserve_recall": round(pres_ok / pres_tot, 4) if pres_tot else 1.0,
        "p0_preserve_error_rows": preserve_error_rows,
        "filler_ranhou_hits": filler_ranhou,
        "runaways": runaways,
        "over_expands": over_expands,
        "under_drops": under_drops,
        "think_leaks": think_leaks,
        "eval_s": round(elapsed, 2),
    }
    print(f"EVAL {label}: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary, preds


def selection_score(s: dict):
    """Rank checkpoints for early stopping: zero hard defects -> P0 entity preserve -> low filler -> char sim."""
    hard_defects = s["runaways"] + s["over_expands"] + s["under_drops"] + s["think_leaks"]
    # Penalize filler ranhou above 6 hits moderately
    filler_penalty = max(0, s["filler_ranhou_hits"] - 6) * 0.005
    return (-hard_defects, s["p0_preserve_recall"] - filler_penalty, -s["filler_ranhou_hits"], s["mean_char_sim"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--anchor-pairs", help="Optional synthetic anchor JSONL appended ONLY to train split")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-size", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-each-epoch", action="store_true", help="Evaluate after every epoch and save best checkpoint")
    ap.add_argument("--skip-base-eval", action="store_true", help="Skip zero-shot baseline evaluation")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows, dropped = load_and_split(args.pairs, args.val_size)
    anchor_n = 0
    if args.anchor_pairs:
        anc_raw = [json.loads(l) for l in Path(args.anchor_pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in anc_raw:
            asr = (r.get("asr") or "").strip()
            typ = (r.get("typeless") or "").strip()
            if asr and typ:
                train_rows.append({"id": r.get("id", ""), "created_at": r.get("created_at", ""), "asr": asr, "typeless": typ})
                anchor_n += 1
    print(f"Loaded train={len(train_rows)} (incl anchor={anchor_n}), val={len(val_rows)}, dropped={dropped}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")

    # 1. Baseline zero-shot eval on holdout
    if args.skip_base_eval:
        base_summary = {"skipped": True}
        base_preds = [{"id": r["id"], "asr": r["asr"], "ref": r["typeless"], "pred": "", "sim": 0.0} for r in val_rows]
    else:
        base_summary, base_preds = evaluate(model, tokenizer, val_rows, "base_zero_shot")

    # 2. Build tokenized training set
    train_examples = []
    for r in train_rows:
        ex = build_example(tokenizer, r["asr"], r["typeless"])
        if ex is not None:
            train_examples.append(ex)
    print(f"Tokenized train examples={len(train_examples)}", flush=True)

    # 3. Full-parameter bf16 SFT
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(train_examples) / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t_train = time.perf_counter()
    global_step = 0
    epoch_history = []
    best_score = None
    best_epoch = None
    best_summary = None
    best_preds = None
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        g = torch.Generator().manual_seed(args.seed + ep)
        perm = torch.randperm(len(train_examples), generator=g).tolist()
        ep_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, idx in enumerate(perm):
            ex = train_examples[idx]
            input_ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device="cuda")
            labels = torch.tensor([ex["labels"]], dtype=torch.long, device="cuda")
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / args.grad_accum
            loss.backward()
            ep_loss += out.loss.item()
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(perm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        avg_loss = round(ep_loss / len(perm), 4)
        print(f"Epoch {ep + 1}/{args.epochs} avg_loss={avg_loss:.4f} steps={global_step}", flush=True)

        if args.eval_each_epoch:
            model.gradient_checkpointing_disable()
            model.config.use_cache = True
            ep_summary, ep_preds = evaluate(model, tokenizer, val_rows, f"epoch_{ep + 1}")
            ep_summary["avg_loss"] = avg_loss
            ep_summary["epoch"] = ep + 1
            epoch_history.append(ep_summary)
            (out_dir / f"epoch_{ep + 1}_preds.json").write_text(
                json.dumps({"summary": ep_summary, "preds": ep_preds}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            sc = selection_score(ep_summary)
            if best_score is None or sc > best_score:
                best_score = sc
                best_epoch = ep + 1
                best_summary = ep_summary
                best_preds = ep_preds
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"NEW BEST checkpoint at epoch {best_epoch}: score={sc}", flush=True)

    train_s = round(time.perf_counter() - t_train, 2)
    print(f"Training done in {train_s}s, peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    if not args.eval_each_epoch:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        best_summary, best_preds = evaluate(model, tokenizer, val_rows, "sft_qwen3_0.6b")
        best_epoch = args.epochs
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    summary = {
        "train_n": len(train_examples),
        "anchor_n": anchor_n,
        "val_n": len(val_rows),
        "dropped": dropped,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "lr": args.lr,
        "train_s": train_s,
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "base": base_summary,
        "sft": best_summary,
        "epoch_history": epoch_history,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "eval_compare.jsonl").open("w", encoding="utf-8") as f:
        for b, s in zip(base_preds, best_preds):
            f.write(
                json.dumps(
                    {
                        "id": b["id"],
                        "asr": b["asr"],
                        "ref": b["ref"],
                        "base_pred": b["pred"],
                        "base_sim": b["sim"],
                        "sft_pred": s["pred"],
                        "sft_sim": s["sim"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print("ALL DONE:", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
