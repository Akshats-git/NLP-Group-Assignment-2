"""Post-hoc analysis of the two translation systems (run after evaluate.py).

    python analysis.py

Produces, in outputs/results/ and outputs/figures/:
    baseline_limitation.json   short vs long, hypothesis length ratio, per-position recall
    attention_behaviour.json   alignment shape, entropy, mass on specials, weight vs correctness
    error_cases.json / .md     test sentences where attention is wrong or partly wrong
    error_case_XX.png          attention heatmap for each case
    position_recall.png        reference-token recall by position, long sentences
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    load_jsonl, load_json, save_json, results_dir, figures_dir, bucket_indices,
    bucket_labels, markdown_table,
)
from common.plotting import plot_attention, plot_grouped_bars
from data import DEFAULT_MAX_TRAIN, load_splits, build_vocabs
from evaluate import attn_path

import sacrebleu

PUNCT = set(". , ; : ! ? ( ) [ ] \" ' ‘ ’ “ ” । |".split())


# ── helpers ──────────────────────────────────────────────────────────────────

def sent_scores(hyp: str, ref: str) -> tuple[float, float]:
    """Sentence BLEU and chrF (both 0-100); smoothed, so usable for ranking."""
    return (sacrebleu.sentence_bleu(hyp, [ref]).score, sacrebleu.sentence_chrf(hyp, [ref]).score)


def clipped_matches(hyp_tokens: list[str], ref_tokens: list[str]) -> list[bool]:
    """For each reference token, whether the hypothesis contains it (multiset-clipped)."""
    pool = Counter(hyp_tokens)
    flags = []
    for tok in ref_tokens:
        if pool[tok] > 0:
            pool[tok] -= 1
            flags.append(True)
        else:
            flags.append(False)
    return flags


def position_recall(rows: list[dict], n_bins: int = 4) -> list[float]:
    """Share of reference tokens found in the hypothesis, by relative position in the reference."""
    hits, totals = [0] * n_bins, [0] * n_bins
    for r in rows:
        ref = r["ref_text"].split()
        flags = clipped_matches(r["hyp_tokens"], ref)
        for j, ok in enumerate(flags):
            b = min(n_bins - 1, int(n_bins * j / len(ref)))
            totals[b] += 1
            hits[b] += ok
    return [h / t if t else float("nan") for h, t in zip(hits, totals)]


def has_repetition(tokens: list[str], n: int = 3) -> bool:
    """True if some token is repeated n or more times in a row, or a bigram repeats n times."""
    for i in range(len(tokens) - n + 1):
        if len(set(tokens[i:i + n])) == 1:
            return True
    for i in range(len(tokens) - 2 * n + 1):
        b = tokens[i:i + 2]
        if all(tokens[i + 2 * k: i + 2 * k + 2] == b for k in range(n)):
            return True
    return False


# ── baseline limitation ──────────────────────────────────────────────────────

def baseline_limitation(base: list[dict], att: list[dict]) -> dict:
    """Why a fixed-size state struggles on long inputs: numbers behind the argument."""
    out = {}
    buckets = bucket_indices([r["src_len"] for r in base])
    groups = {**{k: v for k, v in buckets.items()},
              "short (1-10)": buckets["1-5"] + buckets["6-10"],
              "long (16+)": buckets["16+"]}
    for name, rows in (("baseline", base), ("attention", att)):
        per = {}
        for label, idxs in groups.items():
            sub = [rows[i] for i in idxs]
            if not sub:
                continue
            hyp_len = sum(len(r["hyp_tokens"]) for r in sub)
            ref_len = sum(len(r["ref_text"].split()) for r in sub)
            chrf = np.mean([sent_scores(r["hyp_text"], r["ref_text"])[1] for r in sub])
            per[label] = {
                "count": len(sub),
                "hyp_ref_length_ratio": hyp_len / ref_len,
                "mean_sentence_chrf": float(chrf),
                "repetition_rate": float(np.mean([has_repetition(r["hyp_tokens"]) for r in sub])),
                "unk_rate": float(np.mean([("<unk>" in r["hyp_tokens"]) for r in sub])),
                "position_recall_quartiles": position_recall(sub),
            }
        out[name] = per
    return out


# ── attention behaviour ──────────────────────────────────────────────────────

def attention_behaviour(att: list[dict], weights: list[torch.Tensor]) -> dict:
    """Statistics of the attention weights over the whole test set."""
    ent, special_mass, fwd, argmax_conf = [], [], [], []
    multi, conf_correct, conf_wrong = [], [], []
    diag_dev = []
    for r, w in zip(att, weights):
        w = w.numpy()
        n_words = len(r["hyp_tokens"])
        if n_words == 0:
            continue
        content = w[:n_words]                           # one row per generated word
        S = w.shape[1]
        ent.extend((-(content * np.log(content + 1e-12)).sum(1)).tolist())
        special_mass.extend((content[:, 0] + content[:, S - 1]).tolist())
        am = content.argmax(1)
        if n_words > 1:
            fwd.append(float(np.mean(am[1:] >= am[:-1] - 1)))   # monotone (allow 1 step back)
        top = content.max(1)
        argmax_conf.extend(top.tolist())
        multi.extend(((top < 0.5) & ((content > 0.2).sum(1) >= 2)).tolist())
        # distance from the straight line joining first and last source word
        src_pos = (am / max(S - 1, 1))
        tgt_pos = np.arange(n_words) / max(n_words - 1, 1)
        diag_dev.append(float(np.abs(src_pos - tgt_pos).mean()))
        ref_set = set(r["ref_text"].split())
        for tok, c in zip(r["hyp_tokens"], top):
            (conf_correct if tok in ref_set else conf_wrong).append(float(c))
    return {
        "mean_entropy_nats": float(np.mean(ent)),
        "mean_mass_on_sos_eos": float(np.mean(special_mass)),
        "mean_max_weight": float(np.mean(argmax_conf)),
        "share_steps_multi_word_attention": float(np.mean(multi)),
        "share_steps_monotone_forward": float(np.mean(fwd)),
        "mean_deviation_from_diagonal": float(np.mean(diag_dev)),
        "mean_max_weight_when_token_in_reference": float(np.mean(conf_correct)),
        "mean_max_weight_when_token_not_in_reference": float(np.mean(conf_wrong)),
        "share_tokens_in_reference": len(conf_correct) / max(len(conf_correct) + len(conf_wrong), 1),
        "share_wrong_tokens_with_max_weight_above_0.7":
            float(np.mean(np.array(conf_wrong) > 0.7)),
        "n_sentences": len(att),
    }


# ── error cases ──────────────────────────────────────────────────────────────

def auto_flags(r_att: dict, w: np.ndarray, src_tokens: list[str], vocab_en, vocab_hi) -> list[str]:
    """Heuristic first-pass labels; the final classification is done by hand in the report."""
    ref = r_att["ref_text"].split()
    hyp = r_att["hyp_tokens"]
    flags = []
    if any(t not in vocab_en.stoi for t in src_tokens):
        flags.append("oov_source")
    if "<unk>" in hyp or sum(t not in vocab_hi.stoi for t in ref) / max(len(ref), 1) > 0.15:
        flags.append("oov_target")
    matched = clipped_matches(hyp, ref)
    if np.mean(matched) < 0.5:
        flags.append("missing_or_wrong_words")
    if len(hyp) > 1.3 * len(ref) + 1:
        flags.append("extra_words")
    if len(hyp) < 0.6 * len(ref) - 1:
        flags.append("truncated")
    if sorted(hyp) == sorted(ref) and hyp != ref or (np.mean(matched) > 0.7 and sacrebleu.sentence_bleu(r_att["hyp_text"], [r_att["ref_text"]]).score < 40):
        flags.append("word_order")
    if has_repetition(hyp):
        flags.append("repetition")
    if len(src_tokens) >= 16:
        flags.append("long_input")
    content = w[: len(hyp)]
    if len(hyp) and (content[:, 0] + content[:, -1]).mean() > 0.35:
        flags.append("attention_on_sos_eos")
    return flags


def select_error_cases(att, base, weights, vocab_en, vocab_hi, n_cases: int = 8):
    """Pick informative failures of the attention model, spread over error types and lengths."""
    scored = []
    for i, (a, b, w) in enumerate(zip(att, base, weights)):
        n_src = a["src_len"]
        if n_src < 6 or len(a["hyp_tokens"]) < 3:
            continue
        a_bleu, a_chrf = sent_scores(a["hyp_text"], a["ref_text"])
        b_bleu, b_chrf = sent_scores(b["hyp_text"], b["ref_text"])
        if not (22 <= a_chrf <= 55):                       # partly wrong, not garbage
            continue
        flags = auto_flags(a, w.numpy(), a["src_text"].split(), vocab_en, vocab_hi)
        scored.append({"idx": i, "flags": flags, "att_chrf": a_chrf, "base_chrf": b_chrf,
                       "att_bleu": a_bleu, "base_bleu": b_bleu, "src_len": n_src})
    chosen, used_flags = [], Counter()
    # greedy: prefer cases with a rarely used flag, alternate short/medium/long inputs
    pool = sorted(scored, key=lambda c: c["idx"])
    while pool and len(chosen) < n_cases:
        def rarity(c):
            return -sum(1 / (1 + used_flags[f]) for f in c["flags"] if f not in ("long_input",))
        best = min(pool[::7] or pool, key=lambda c: (rarity(c), c["att_chrf"]))
        chosen.append(best)
        used_flags.update(best["flags"])
        pool = [c for c in pool if c["idx"] != best["idx"]]
    return chosen, len(scored)


def write_error_cases(chosen, att, base, weights, notes: dict):
    fig_dir = figures_dir(1)
    rows, md = [], []
    for k, c in enumerate(chosen, 1):
        a, b, w = att[c["idx"]], base[c["idx"]], weights[c["idx"]]
        src_tokens = ["<sos>"] + a["src_text"].split() + ["<eos>"]
        n = len(a["hyp_tokens"])
        fig = f"error_case_{k:02d}.png"
        plot_attention(w[:n], src_tokens, a["hyp_tokens"], out_path=fig_dir / fig,
                       title=f"Error case {k} (test #{c['idx']}): auto flags {', '.join(c['flags']) or 'none'}")
        note = notes.get(str(c["idx"]), {})
        rows.append({"case": k, "test_idx": c["idx"], "source": a["src_text"], "reference": a["ref_text"],
                     "baseline": b["hyp_text"], "attention": a["hyp_text"],
                     "att_sentence_chrf": c["att_chrf"], "base_sentence_chrf": c["base_chrf"],
                     "auto_flags": c["flags"], "figure": fig, **note})
        md.append({"case": k, "English": a["src_text"], "Reference": a["ref_text"],
                   "Baseline": b["hyp_text"], "Attention": a["hyp_text"],
                   "chrF (B / A)": f"{c['base_chrf']:.0f} / {c['att_chrf']:.0f}",
                   "Auto flags": ", ".join(c["flags"]),
                   "Error type (manual)": note.get("error_type", "")})
    save_json(rows, results_dir(1) / "error_cases.json")
    (results_dir(1) / "error_cases.md").write_text(markdown_table(md), encoding="utf-8")
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    p.add_argument("--n-cases", type=int, default=8)
    args = p.parse_args()

    res = results_dir(1)
    base = load_jsonl(res / f"baseline_seed{args.seed}" / "test_predictions.jsonl")
    att = load_jsonl(res / f"attention_seed{args.seed}" / "test_predictions.jsonl")
    weights = torch.load(attn_path(args.seed), weights_only=True)
    assert len(base) == len(att) == len(weights)

    train, _, _, _ = load_splits(max_train=args.max_train, seed=args.seed)
    vocab_en, vocab_hi, _ = build_vocabs(train)

    print("baseline limitation ...")
    lim = baseline_limitation(base, att)
    save_json(lim, res / "baseline_limitation.json")
    for m in lim:
        for label, v in lim[m].items():
            print(f"  {m:9s} {label:13s} n={v['count']:4d} len-ratio {v['hyp_ref_length_ratio']:.2f} "
                  f"chrF {v['mean_sentence_chrf']:.1f} rep {v['repetition_rate']:.3f} "
                  f"recall-by-quartile {[round(x, 2) for x in v['position_recall_quartiles']]}")
    plot_grouped_bars(
        {m.title(): {f"Q{i + 1}": lim[m]["long (16+)"]["position_recall_quartiles"][i] * 100
                     for i in range(4)} for m in lim},
        ylabel="reference tokens found in output (%)",
        xlabel="position in reference (Q1 = first quarter), source 16+ tokens",
        title="Which part of a long sentence is translated?",
        out_path=figures_dir(1) / "position_recall.png")

    print("attention behaviour ...")
    beh = attention_behaviour(att, weights)
    save_json(beh, res / "attention_behaviour.json")
    for k, v in beh.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("error cases ...")
    notes_path = res / "error_case_notes.json"            # manual labels keyed by test index
    notes = load_json(notes_path) if notes_path.exists() else {}
    chosen, n_eligible = select_error_cases(att, base, weights, vocab_en, vocab_hi, args.n_cases)
    rows = write_error_cases(chosen, att, base, weights, notes)
    print(f"  {n_eligible} eligible sentences, {len(rows)} written")
    for r in rows:
        print(f"\n[{r['case']}] test #{r['test_idx']}  flags={r['auto_flags']}  chrF B/A {r['base_sentence_chrf']:.0f}/{r['att_sentence_chrf']:.0f}")
        print(f"  EN : {r['source']}\n  REF: {r['reference']}\n  BAS: {r['baseline']}\n  ATT: {r['attention']}")


if __name__ == "__main__":
    main()
