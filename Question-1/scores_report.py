"""Recompute every Q1 evaluation score from the saved test predictions and write them out in full.

Why this exists: evaluate.py stores the numbers in several JSON files, and the figures only show BLEU/chrF
curves. This script recomputes BLEU (with all four n-gram precisions, brevity penalty, lengths), chrF, chrF++
and TER from outputs/results/<model>_seed42/test_predictions.jsonl, checks them against the values saved by
evaluate.py, and writes one readable place for all of them:

    outputs/results/all_evaluation_scores.json
    outputs/results/all_evaluation_scores.md
    outputs/results/all_evaluation_scores.csv

Usage (from Question-1/):  python scores_report.py
"""

import csv
import json
import math
import sys
from pathlib import Path

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

ROOT = Path(__file__).resolve().parent
RES = ROOT / "outputs" / "results"
MODELS = {"baseline": "A: baseline (no attention)", "attention": "B: attention (Bahdanau)"}
BUCKETS = [("1-5", 1, 5), ("6-10", 6, 10), ("11-15", 11, 15), ("16+", 16, 10**9), ("all", 0, 10**9)]


def score(hyps: list[str], refs: list[str]) -> dict:
    b = BLEU().corpus_score(hyps, [refs])
    return {
        "n_sentences": len(hyps),
        "BLEU": b.score,
        "BLEU_p1": b.precisions[0], "BLEU_p2": b.precisions[1],
        "BLEU_p3": b.precisions[2], "BLEU_p4": b.precisions[3],
        "brevity_penalty": b.bp, "length_ratio": b.ratio,
        "hyp_tokens": b.sys_len, "ref_tokens": b.ref_len,
        "chrF": CHRF().corpus_score(hyps, [refs]).score,
        "chrF++": CHRF(word_order=2).corpus_score(hyps, [refs]).score,
        "TER": TER().corpus_score(hyps, [refs]).score,
    }


def main() -> int:
    out: dict = {"sacrebleu_version": sacrebleu.__version__, "models": {}}
    ok = True
    for key in MODELS:
        d = RES / f"{key}_seed42"
        rows = [json.loads(l) for l in open(d / "test_predictions.jsonl", encoding="utf-8")]
        saved = json.load(open(d / "test_bleu.json"))
        hist = json.load(open(d / "history.json"))
        info = json.load(open(d / "run_info.json"))
        best = min(hist, key=lambda h: h["val_loss"])
        per = {}
        for name, lo, hi in BUCKETS:
            sel = [r for r in rows if lo <= r["src_len"] <= hi]
            per[name] = score([r["hyp_text"] for r in sel], [r["ref_text"] for r in sel])
        # consistency with what evaluate.py saved
        diff = abs(per["all"]["BLEU"] - saved["bleu"]) + abs(per["all"]["chrF"] - saved["chrf"])
        if diff > 1e-6 or per["all"]["n_sentences"] != 2393:
            ok = False
        out["models"][key] = {
            "name": MODELS[key],
            "parameters": info["parameters"]["total"],
            "test": per["all"],
            "buckets": {k: v for k, v in per.items() if k != "all"},
            "test_loss": saved["test_loss"], "test_perplexity": math.exp(saved["test_loss"]),
            "test_token_acc": saved["test_token_acc"],
            "best_epoch": best["epoch"], "best_val_loss": best["val_loss"], "best_val_acc": best["val_acc"],
            "best_epoch_train_loss": best["train_loss"], "best_epoch_train_acc": best["train_acc"],
            "epochs_run": len(hist), "seconds_per_epoch": sum(h["time_s"] for h in hist) / len(hist),
            "matches_saved_test_bleu_json": diff <= 1e-6,
        }
    (RES / "all_evaluation_scores.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    cols = ["n_sentences", "BLEU", "BLEU_p1", "BLEU_p2", "BLEU_p3", "BLEU_p4", "brevity_penalty",
            "length_ratio", "hyp_tokens", "ref_tokens", "chrF", "chrF++", "TER"]
    with open(RES / "all_evaluation_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset"] + cols)
        for key, m in out["models"].items():
            for sub, v in [("all", m["test"])] + list(m["buckets"].items()):
                w.writerow([key, sub] + [round(v[c], 4) if isinstance(v[c], float) else v[c] for c in cols])

    md = ["# Q1 evaluation scores (test set, 2,393 sentences, seed 42)\n",
          f"Recomputed from `test_predictions.jsonl` with sacrebleu {sacrebleu.__version__}; "
          "identical to `test_bleu.json` written by evaluate.py.\n",
          "## Overall\n",
          "| metric | baseline (A) | attention (B) |", "|---|---|---|"]
    A, B = out["models"]["baseline"], out["models"]["attention"]
    for c in cols:
        fa, fb = A["test"][c], B["test"][c]
        fmt = (lambda x: f"{x:,}") if isinstance(fa, int) else (lambda x: f"{x:.4f}")
        md.append(f"| {c} | {fmt(fa)} | {fmt(fb)} |")
    for c in ["test_loss", "test_perplexity", "test_token_acc", "best_epoch", "best_val_loss", "best_val_acc",
              "best_epoch_train_loss", "best_epoch_train_acc", "parameters", "seconds_per_epoch"]:
        va, vb = A[c], B[c]
        f = (lambda x: f"{x:,}") if isinstance(va, int) else (lambda x: f"{x:.4f}")
        md.append(f"| {c} | {f(va)} | {f(vb)} |")
    for key, m in out["models"].items():
        md += [f"\n## By English sentence length: {m['name']}\n",
               "| length | n | BLEU | p1 | p2 | p3 | p4 | BP | len ratio | chrF | chrF++ | TER |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for b, v in m["buckets"].items():
            md.append(f"| {b} | {v['n_sentences']} | {v['BLEU']:.2f} | {v['BLEU_p1']:.1f} | {v['BLEU_p2']:.1f} | "
                      f"{v['BLEU_p3']:.1f} | {v['BLEU_p4']:.1f} | {v['brevity_penalty']:.3f} | "
                      f"{v['length_ratio']:.3f} | {v['chrF']:.2f} | {v['chrF++']:.2f} | {v['TER']:.1f} |")
    (RES / "all_evaluation_scores.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("\nconsistent with saved test_bleu.json:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
