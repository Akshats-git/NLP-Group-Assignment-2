"""Part 5 (human evaluation, step 1): build the blinded rating sheet.

    python Question-4/build_human_eval.py --configs greedy "top-k=40"

Takes sample 0 of every plot-twist prompt x model x listed decoding configuration (a fixed rule, no cherry-picking),
shuffles them, replaces the labels with neutral ids and writes:

    outputs/human_eval/rate.html     what raters open: a self-contained form that exports their ratings as CSV
    outputs/human_eval/samples.json  the same stories without labels (for anyone who prefers a spreadsheet)
    outputs/human_eval/samples.csv
    outputs/human_eval/key.json      PRIVATE: which model/decoding wrote each id. Git-ignored until rating closes.

Give raters only rate.html (or samples.csv). Do not show them key.json, the generations file, or the report.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import load_jsonl, save_json
from plot_twist import config as cfg
from plot_twist.data import detokenize, sentence_case
from plot_twist.human_eval import CRITERIA, CRITERIA_DESCRIPTIONS, select_samples
from plot_twist.generation import prompt_text_for_display

TEMPLATE = Path(__file__).parent / "plot_twist" / "rate_template.html"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default="generations_seed42.jsonl")
    ap.add_argument("--configs", nargs="+", default=["greedy", "top-k=40"], help="decoding labels to include")
    ap.add_argument("--seed", type=int, default=2024, help="shuffle seed for the sheet order")
    args = ap.parse_args()

    p = cfg.paths()
    out_dir = p["results"].parent / "human_eval"
    rows = load_jsonl(p["results"] / args.generations)
    public, key = select_samples(rows, cfg.CELLS, args.configs, args.seed)
    for s in public:
        s["prompt"] = prompt_text_for_display(s["prompt"])
    if len(public) < 10:
        raise SystemExit(f"only {len(public)} samples; the brief asks for at least 10")

    save_json(public, out_dir / "samples.json")
    save_json({"seed": args.seed, "configs": args.configs, "key": key}, out_dir / "key.json")
    with (out_dir / "samples.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "opening", "continuation", *CRITERIA])
        for s in public:
            w.writerow([s["id"], s["prompt"], s["text"], *[""] * len(CRITERIA)])

    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("/*SAMPLES*/[]", json.dumps(public, ensure_ascii=False))
    page = page.replace("/*CRITERIA*/[]", json.dumps([{"key": c, "text": CRITERIA_DESCRIPTIONS[c]} for c in CRITERIA]))
    (out_dir / "rate.html").write_text(page, encoding="utf-8")
    print(f"{len(public)} blinded samples -> {out_dir}")


if __name__ == "__main__":
    main()
