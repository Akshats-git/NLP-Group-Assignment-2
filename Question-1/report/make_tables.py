"""Generate report/tables.tex for Q1_report.tex straight from outputs/results/*.json, so that every number in the
report's tables is read from the result files and not typed by hand.

Run from anywhere:  python Question-1/report/make_tables.py   (run ../scores_report.py first)
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "outputs" / "results"
load = lambda p: json.load(open(RES / p, encoding="utf-8"))

DEV = r"[ऀ-ॿ‌‍]+"
HI_RUN = re.compile(rf"{DEV}(?:\s+{DEV})*")
ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
       "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}", "<": r"\textless{}",
       ">": r"\textgreater{}"}


def tex(s: str) -> str:
    """Escape LaTeX specials and wrap each Devanagari run in \\hi{...}."""
    out, last = [], 0
    for m in HI_RUN.finditer(s):
        out.append("".join(ESC.get(c, c) for c in s[last:m.start()]))
        out.append(r"\hi{" + m.group(0) + "}")
        last = m.end()
    out.append("".join(ESC.get(c, c) for c in s[last:]))
    return "".join(out)


S = load("all_evaluation_scores.json")["models"]
A, B = S["baseline"], S["attention"]
t = []


def table(name, head, rows, spec):
    t.append(f"% ---- {name}")
    t.append(rf"\newcommand{{\tab{name}}}{{")
    t.append(rf"\begin{{tabular}}{{{spec}}}\toprule")
    t.append(head + r" \\ \midrule")
    for r in rows:
        t.append(r + r" \\" if not r.endswith(r"\midrule") else r)
    t.append(r"\bottomrule\end{tabular}}")


f = lambda x, d=2: f"{x:.{d}f}"
bold = lambda x: r"\textbf{" + x + "}"

# 1. headline translation metrics
rows = []
for lab, key, d, better in [("BLEU (corpus, sacreBLEU 13a)", "BLEU", 2, "max"), ("chrF", "chrF", 2, "max"),
                            ("chrF++", "chrF++", 2, "max"), ("TER (lower is better)", "TER", 2, "min"),
                            ("BLEU 1-gram precision", "BLEU_p1", 2, "max"), ("BLEU 2-gram precision", "BLEU_p2", 2, "max"),
                            ("BLEU 3-gram precision", "BLEU_p3", 2, "max"), ("BLEU 4-gram precision", "BLEU_p4", 2, "max"),
                            ("Brevity penalty", "brevity_penalty", 3, None), ("Length ratio (hyp/ref)", "length_ratio", 3, None)]:
    a, b = A["test"][key], B["test"][key]
    fa, fb = f(a, d), f(b, d)
    if better:
        b_wins = b > a if better == "max" else b < a
        if b_wins:
            fb = bold(fb)
        else:
            fa = bold(fa)
    rows.append(f"{lab} & {fa} & {fb}")
rows.append(r"Hypothesis tokens / reference tokens & " + f"{A['test']['hyp_tokens']:,} / {A['test']['ref_tokens']:,} & "
            f"{B['test']['hyp_tokens']:,} / {B['test']['ref_tokens']:,}")
table("Metrics", r"Test metric (2,393 sentences) & A: baseline & B: attention", rows, "lrr")

# 2. loss / accuracy / size
rows = []
for lab, key, d in [("Best epoch (lowest val. loss)", "best_epoch", 0), ("Train loss at best epoch", "best_epoch_train_loss", 3),
                    ("Validation loss", "best_val_loss", 3), ("Test loss", "test_loss", 3),
                    ("Test perplexity", "test_perplexity", 1), ("Train token accuracy at best epoch", "best_epoch_train_acc", 3),
                    ("Validation token accuracy", "best_val_acc", 3), ("Test token accuracy", "test_token_acc", 3),
                    ("Parameters", "parameters", 0), ("Seconds per epoch", "seconds_per_epoch", 0)]:
    a, b = A[key], B[key]
    sa = f"{a:,.0f}" if d == 0 else f(a, d)
    sb = f"{b:,.0f}" if d == 0 else f(b, d)
    rows.append(f"{lab} & {sa} & {sb}")
table("Loss", r"Training / validation / test & A: baseline & B: attention", rows, "lrr")

# 3. per-epoch history
hb, ha = load("baseline_seed42/history.json"), load("attention_seed42/history.json")
rows = []
for x, y in zip(hb, ha):
    mark = r"\textbf{%s}" if x["epoch"] == 4 else "%s"
    cells = [x["epoch"], x["train_loss"], x["val_loss"], x["train_acc"], x["val_acc"],
             y["train_loss"], y["val_loss"], y["train_acc"], y["val_acc"]]
    cells = [str(cells[0])] + [f"{c:.3f}" if i < 5 else f"{c:.3f}" for i, c in enumerate(cells[1:])]
    if x["epoch"] == 4:
        cells = [r"\textbf{" + c + "}" for c in cells]
    rows.append(" & ".join(cells))
table("History", r"Epoch & \multicolumn{4}{c}{A: baseline} & \multicolumn{4}{c}{B: attention} \\ & "
      r"train loss & val loss & train acc & val acc & train loss & val loss & train acc & val acc",
      rows, "r|rrrr|rrrr")

# 4. per length bucket (all metrics)
lim = load("baseline_limitation.json")
for key, M in (("A", A), ("B", B)):
    rows = []
    for b, v in M["buckets"].items():
        rows.append(f"{b} & {v['n_sentences']:,} & {v['BLEU']:.2f} & {v['BLEU_p1']:.1f} & {v['BLEU_p2']:.1f} & "
                    f"{v['BLEU_p3']:.1f} & {v['BLEU_p4']:.1f} & {v['brevity_penalty']:.3f} & {v['length_ratio']:.3f} & "
                    f"{v['chrF']:.2f} & {v['TER']:.1f}")
    rows.append(r"\midrule all & " + f"{M['test']['n_sentences']:,} & {M['test']['BLEU']:.2f} & {M['test']['BLEU_p1']:.1f} & "
                f"{M['test']['BLEU_p2']:.1f} & {M['test']['BLEU_p3']:.1f} & {M['test']['BLEU_p4']:.1f} & "
                f"{M['test']['brevity_penalty']:.3f} & {M['test']['length_ratio']:.3f} & {M['test']['chrF']:.2f} & {M['test']['TER']:.1f}")
    table("Bucket" + key, r"EN length & n & BLEU & p1 & p2 & p3 & p4 & BP & ratio & chrF & TER", rows, "lrrrrrrrrrr")

# 5. side-by-side bucket comparison + repetition
rows = []
for b in ["1-5", "6-10", "11-15", "16+"]:
    a, c = A["buckets"][b], B["buckets"][b]
    rows.append(f"{b} & {a['n_sentences']:,} & {a['BLEU']:.2f} & {c['BLEU']:.2f} & {c['BLEU'] / a['BLEU']:.1f}$\\times$ & "
                f"{a['chrF']:.2f} & {c['chrF']:.2f} & {lim['baseline'][b]['repetition_rate'] * 100:.1f}\\% & "
                f"{lim['attention'][b]['repetition_rate'] * 100:.1f}\\%")
table("Compare", r"EN length & n & BLEU A & BLEU B & B/A & chrF A & chrF B & repeats A & repeats B", rows, "lrrrrrrrr")

# 6. position recall
rows = []
for lab, k in (("A: baseline", "baseline"), ("B: attention", "attention")):
    q = lim[k]["long (16+)"]["position_recall_quartiles"]
    rows.append(lab + " & " + " & ".join(f"{x:.3f}" for x in q))
table("Recall", r"Unigram recall, 16+ tokens & Q1 (first) & Q2 & Q3 & Q4 (last)", rows, "lrrrr")

# 7. attention statistics
ab = load("attention_behaviour.json")
rows = [f"{l} & {v}" for l, v in [
    ("Test sentences analysed", f"{ab['n_sentences']:,}"),
    ("Mean peak attention weight", f"{ab['mean_max_weight']:.3f}"),
    ("Mean attention entropy (nats)", f"{ab['mean_entropy_nats']:.3f}"),
    ("Steps whose peak stays or moves forward", f"{ab['share_steps_monotone_forward'] * 100:.1f}\\%"),
    ("Mean deviation from the diagonal (relative)", f"{ab['mean_deviation_from_diagonal']:.3f}"),
    ("Steps with multi-word attention", f"{ab['share_steps_multi_word_attention'] * 100:.1f}\\%"),
    ("Attention mass on <sos>/<eos>".replace("<", r"\textless{}").replace(">", r"\textgreater{}"),
     f"{ab['mean_mass_on_sos_eos'] * 100:.2f}\\%"),
    ("Generated tokens that occur in the reference", f"{ab['share_tokens_in_reference'] * 100:.1f}\\%"),
    ("Mean peak weight, token in reference", f"{ab['mean_max_weight_when_token_in_reference']:.4f}"),
    ("Mean peak weight, token not in reference", f"{ab['mean_max_weight_when_token_not_in_reference']:.4f}"),
    ("Wrong tokens generated with peak weight $>0.7$", f"{ab['share_wrong_tokens_with_max_weight_above_0.7'] * 100:.2f}\\%")]]
table("Attn", r"Attention statistic & Value", rows, "lr")

# 8. data statistics
ds = load("attention_seed42/data_stats.json")
rows = [f"{l} & {v}" for l, v in [
    ("Raw training pairs in IITB corpus", f"{ds['total_raw_train']:,}"),
    ("Training pairs used (random draw, seed 42)", f"{ds['train_pairs']:,}"),
    ("Development pairs / test pairs (official, cleaned)", f"{ds['dev_pairs']:,} / {ds['test_pairs']:,}"),
    ("English vocabulary size", f"{ds['en_vocab_size']:,}"),
    ("Hindi vocabulary size", f"{ds['hi_vocab_size']:,}"),
    ("Maximum English / Hindi sentence length (tokens)", f"{ds['max_en_len_train']} / {ds['max_hi_len_train']}"),
    ("Mean length, training (English / Hindi)", f"{ds['avg_en_len_train']:.1f} / {ds['avg_hi_len_train']:.1f}"),
    ("OOV rate, dev (English / Hindi)", f"{ds['en_oov_rate_dev'] * 100:.1f}\\% / {ds['hi_oov_rate_dev'] * 100:.1f}\\%"),
    ("OOV rate, test (English / Hindi)", f"{ds['en_oov_rate_test'] * 100:.1f}\\% / {ds['hi_oov_rate_test'] * 100:.1f}\\%")]]
table("Data", r"Data statistic & Value", rows, "lr")

# 9. error cases as blocks
cases = load("error_cases.json")
notes = load("error_case_notes.json")
order = sorted(cases, key=lambda c: c["case"])
t.append(r"\newcommand{\errorcase}[1]{%")
t.append(r"\ifcase#1\or")
for c in order:
    n = notes[str(c["test_idx"])]
    t.append(r"\begin{minipage}{\linewidth}\small")
    t.append(rf"\textbf{{Case {c['case']}}} (test sentence \#{c['test_idx']}; sentence chrF: baseline {c['base_sentence_chrf']:.1f}, "
             rf"attention {c['att_sentence_chrf']:.1f}; error type: \emph{{{tex(n['error_type'])}}})\\[2pt]")
    t.append(r"\begin{tabular}{@{}l p{0.86\linewidth}@{}}")
    t.append(rf"EN & {tex(c['source'])} \\")
    t.append(rf"Ref & {tex(c['reference'])} \\")
    t.append(rf"A & {tex(c['baseline'])} \\")
    t.append(rf"B & {tex(c['attention'])} \\")
    t.append(r"\end{tabular}\\[2pt]")
    t.append(rf"\textit{{Reading the heatmap:}} {tex(n['comment'])} \textit{{Useful as explanation?}} {tex(n['heatmap_useful'])}.")
    t.append(r"\end{minipage}")
    t.append(r"\or" if c is not order[-1] else r"\fi}")

(HERE / "tables.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
print("wrote", HERE / "tables.tex", len(t), "lines")
