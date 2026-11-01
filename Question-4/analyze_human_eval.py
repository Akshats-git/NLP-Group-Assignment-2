"""Part 5 (human evaluation, step 2): unblind and analyse the collected ratings.

    python Question-4/analyze_human_eval.py                       # all CSVs in outputs/human_eval/ratings/
    python Question-4/analyze_human_eval.py --exclude akshat      # e.g. drop the author, who is not blind

Needs outputs/human_eval/key.json (from build_human_eval.py) and one ratings CSV per rater with the columns
rater, sample_id, coherence, grammar, relevance, creativity. Writes outputs/results/human_eval.json and
outputs/results/human_eval_tables.md, and the figure outputs/figures/human_ratings.png.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import load_json, markdown_table, save_json
from common.plotting import plot_grouped_bars
from plot_twist import config as cfg
from plot_twist.human_eval import (CRITERIA, agreement, group_means, load_ratings, paired_difference, rater_summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings-dir", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], help="rater names to leave out (case-insensitive)")
    args = ap.parse_args()

    p = cfg.paths()
    hdir = p["results"].parent / "human_eval"
    files = sorted(Path(args.ratings_dir or hdir / "ratings").glob("*.csv"))
    if not files:
        raise SystemExit(f"no rating CSVs found in {args.ratings_dir or hdir / 'ratings'}")
    key = load_json(hdir / "key.json")["key"]
    ratings = [r for r in load_ratings(files) if r["rater"].lower() not in {e.lower() for e in args.exclude}]
    unknown = {r["sample_id"] for r in ratings} - set(key)
    if unknown:
        raise SystemExit(f"ratings mention unknown sample ids: {sorted(unknown)}")

    raters = sorted({r["rater"] for r in ratings})
    by_model = group_means(ratings, key, ("model",))
    by_model_dec = group_means(ratings, key, ("model", "decoding"))
    paired = paired_difference(ratings, key, "lstm", "rnn")
    agree = agreement(ratings)
    per_rater = rater_summary(ratings)

    perplexity = {}
    for cell in cfg.CELLS:
        f = p["results"] / f"{cell}_seed42" / "metrics.json"
        if f.exists():
            perplexity[cell] = {"val_ppl": load_json(f)["val_ppl"], "test_ppl": load_json(f)["test_ppl"]}

    save_json({"raters": raters, "excluded": args.exclude, "n_ratings": len(ratings), "by_model": by_model,
               "by_model_decoding": by_model_dec, "paired_lstm_minus_rnn": paired, "agreement": agree,
               "per_rater": per_rater, "perplexity": perplexity}, p["results"] / "human_eval.json")

    names = cfg.MODEL_NAMES
    md = [f"Raters ({len(raters)}): {', '.join(raters)}. Ratings: {len(ratings)}. Stories: {len({r['sample_id'] for r in ratings})}.", "",
          "## Mean rating by model (1-5, mean over stories of the mean over raters)", "",
          markdown_table([{"model": names[m], "stories": v["n_stories"], **{c: f"{v[c]:.2f} ± {v[c + '_sd']:.2f}" for c in CRITERIA},
                           "overall": f"{v['overall']:.2f}", **({"val ppl": f"{perplexity[m]['val_ppl']:.2f}"} if m in perplexity else {})}
                          for m, v in by_model.items()]), "",
          "## By model and decoding configuration", "",
          markdown_table([{"model": names[k.split('/')[0]], "decoding": k.split('/')[1], "stories": v["n_stories"],
                           **{c: f"{v[c]:.2f}" for c in CRITERIA}, "overall": f"{v['overall']:.2f}"} for k, v in by_model_dec.items()]), "",
          f"## Paired comparison, LSTM minus RNN ({paired['n_pairs']} matched pairs: same opening, same decoding)", "",
          markdown_table([{"criterion": c, "mean diff": f"{paired[c]['mean_diff']:+.2f}",
                           "95% bootstrap CI": f"[{paired[c]['ci95'][0]:+.2f}, {paired[c]['ci95'][1]:+.2f}]" if paired[c]["ci95"] else "n/a",
                           "LSTM wins / ties / RNN wins": f"{paired[c]['wins']} / {paired[c]['ties']} / {paired[c]['losses']}"}
                          for c in (*CRITERIA, "overall")]), "",
          "## Inter-rater agreement", "",
          markdown_table([{"criterion": c, "Krippendorff alpha (ordinal)": f"{v['krippendorff_alpha_ordinal']:.2f}",
                           "mean pairwise weighted kappa": f"{v['mean_pairwise_quadratic_kappa']:.2f}",
                           "exact agreement": f"{v['exact_agreement']:.0%}", "within 1 point": f"{v['within_one_agreement']:.0%}",
                           "mean abs difference": f"{v['mean_abs_difference']:.2f}"} for c, v in agree.items()]), "",
          "## Each rater's mean rating", "",
          markdown_table([{"rater": r, "n": v["n"], **{c: f"{v[c]:.2f}" for c in CRITERIA}} for r, v in per_rater.items()]), ""]
    (p["results"] / "human_eval_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    models = list(by_model)
    plot_grouped_bars({names[m]: {c: by_model[m][c] for c in CRITERIA} for m in models},
                      ylabel="mean rating (1-5)", out_path=p["figures"] / "human_ratings.png",
                      title="Blinded human ratings", value_format="{:.2f}")


if __name__ == "__main__":
    main()
