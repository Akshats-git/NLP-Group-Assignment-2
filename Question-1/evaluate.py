"""Evaluate trained NMT models on the test split: BLEU, chrF, token accuracy, length buckets.

Usage:
    python evaluate.py                  # evaluate both models on the test set
    python evaluate.py --model baseline # evaluate one model

Writes, per model, into outputs/results/<model>_seed<seed>/:
    test_bleu.json, bucket_bleu.json, bucket_stats.json, test_predictions.jsonl
and the attention model's test-set attention weights into data/ (gitignored).
Error analysis and the baseline-limitation study live in analysis.py.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    set_seed, get_device, to_device, Vocab,
    generate, padding_mask, pad_collate,
    save_json, save_jsonl, results_dir, checkpoints_dir, data_dir,
    bucket_indices, bucket_labels,
)
from data import DEFAULT_MAX_TRAIN, DEFAULT_MIN_LEN, load_splits, build_vocabs, NMTDataset
from models import build_model
from train import evaluate as teacher_forced_eval

try:
    import sacrebleu
except ImportError:
    sacrebleu = None
    print("WARNING: sacrebleu not installed; BLEU scores will be unavailable.")


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_bleu(hypotheses: list[str], references: list[str]) -> dict:
    """Corpus-level BLEU and chrF via sacrebleu. Inputs are space-tokenised strings."""
    if sacrebleu is None:
        return {"bleu": -1, "warning": "sacrebleu not installed"}
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "bleu_signature": str(bleu),
        "bp": bleu.bp,
        "precisions": bleu.precisions,
        "sys_len": bleu.sys_len,
        "ref_len": bleu.ref_len,
    }


# ── translation ──────────────────────────────────────────────────────────────

def translate_batch(model, src_ids: torch.Tensor, vocab_hi: Vocab,
                    device, is_attention: bool,
                    max_new_tokens: int = 60,
                    src_mask: torch.Tensor | None = None):
    """Greedy-decode a batch of source sentences.

    Returns:
        translations: list[list[str]]  — decoded Hindi tokens per sentence (no <eos>)
        all_weights:  [batch, steps, src_len] attention weights, or None for the baseline
    """
    model.eval()
    src_ids = src_ids.to(device)
    enc_out, hidden = model.encode(src_ids)

    if is_attention and src_mask is not None:
        src_mask = src_mask.to(device)

    batch_size = src_ids.size(0)
    sos_tokens = torch.full((batch_size, 1), vocab_hi.sos_id, dtype=torch.long, device=device)

    # <unk> is the most likely single target token (about 1 in 9 reference words is
    # out-of-vocabulary), so unconstrained greedy decoding collapses into "<unk> <unk> ...".
    # <pad>, <unk> and <sos> are never valid output words, so both models decode with them banned.
    banned = [vocab_hi.pad_id, vocab_hi.unk_id, vocab_hi.sos_id]

    def ban(logits):
        logits = logits.clone()
        logits[:, banned] = torch.finfo(logits.dtype).min
        return logits

    if is_attention:
        def step_fn(tokens, state):
            h = hidden if state is None else state
            logits, h, alpha = model.decode_step(tokens[:, -1], h, enc_out, src_mask)
            return ban(logits), h, alpha
    else:
        def step_fn(tokens, state):
            h = hidden if state is None else state
            logits, h = model.decode_step(tokens[:, -1], h, enc_out)
            return ban(logits), h

    out = generate(step_fn, sos_tokens, max_new_tokens=max_new_tokens,
                   eos_id=vocab_hi.eos_id, pad_id=vocab_hi.pad_id)

    translations = [vocab_hi.decode(out.new_tokens[i]) for i in range(batch_size)]
    all_weights = torch.stack(out.extras, dim=1) if (is_attention and out.extras) else None
    return translations, all_weights


def translate_dataset(model, dataset, vocab_en, vocab_hi, device,
                      is_attention: bool, batch_size: int = 64,
                      max_new_tokens: int = 60):
    """Translate a whole dataset. Returns (result dicts, per-sentence attention tensors)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=pad_collate(vocab_en.pad_id))
    results, all_attn = [], []

    for batch in tqdm(loader, desc="translating"):
        src = to_device(batch, device)["src"]
        src_mask = padding_mask(src, vocab_en.pad_id) if is_attention else None
        translations, weights = translate_batch(
            model, src, vocab_hi, device, is_attention,
            max_new_tokens=max_new_tokens, src_mask=src_mask,
        )
        for i, trans in enumerate(translations):
            results.append({
                "src_text": batch["src_text"][i],
                "ref_text": batch["tgt_text"][i],
                "hyp_tokens": trans,
                "hyp_text": " ".join(trans),
                "src_len": len(batch["src_text"][i].split()),
            })
            if weights is not None:
                src_len = int(batch["src_lengths"][i])          # includes <sos>/<eos>
                # one row per generated word plus the <eos> step
                all_attn.append(weights[i, : len(trans) + 1, :src_len].float().cpu())
    return results, all_attn


# ── length-bucket analysis ───────────────────────────────────────────────────

def length_bucket_metrics(results: list[dict]) -> tuple[dict, dict]:
    """Corpus BLEU per source-length bucket (1-5, 6-10, 11-15, 16+) plus bucket stats."""
    buckets = bucket_indices([r["src_len"] for r in results])
    bleu, stats = {}, {}
    for label, idxs in buckets.items():
        if not idxs:
            bleu[label], stats[label] = 0.0, {"count": 0}
            continue
        hyps = [results[i]["hyp_text"] for i in idxs]
        refs = [results[i]["ref_text"] for i in idxs]
        m = compute_bleu(hyps, refs)
        bleu[label] = m["bleu"]
        stats[label] = {"count": len(idxs), "bleu": m["bleu"], "chrf": m["chrf"],
                        "bp": m["bp"], "hyp_ref_length_ratio": m["sys_len"] / max(m["ref_len"], 1)}
    return bleu, stats


# ── loading ──────────────────────────────────────────────────────────────────

def load_trained_model(model_type: str, vocab_en: Vocab, vocab_hi: Vocab,
                       device) -> torch.nn.Module:
    """Load a trained model from its best checkpoint."""
    model = build_model(model_type, len(vocab_en), len(vocab_hi), pad_idx=vocab_en.pad_id)
    ckpt_path = checkpoints_dir(1) / f"{model_type}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"  loaded {model_type} from {ckpt_path}")
    return model


def attn_path(seed: int) -> Path:
    return data_dir(1) / f"attention_test_weights_seed{seed}.pt"


def evaluate_model(model_type: str, model, test_ds, vocab_en, vocab_hi, device, seed: int):
    """Full test evaluation of one model; writes result files."""
    is_attention = model_type == "attention"
    pad_id = vocab_en.pad_id
    print(f"\nEvaluating {model_type.upper()} on {len(test_ds)} test examples ...")

    results, attn_weights = translate_dataset(
        model, test_ds, vocab_en, vocab_hi, device, is_attention=is_attention,
    )
    bleu = compute_bleu([r["hyp_text"] for r in results], [r["ref_text"] for r in results])

    # teacher-forced loss / token accuracy on the test set (same metric as training)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=pad_collate(pad_id))
    test_loss, test_acc = teacher_forced_eval(
        model, loader, nn.CrossEntropyLoss(ignore_index=pad_id), device, pad_id, is_attention)
    bleu["test_loss"], bleu["test_token_acc"] = test_loss, test_acc
    print(f"  BLEU {bleu['bleu']:.2f}  chrF {bleu['chrf']:.2f}  "
          f"test loss {test_loss:.3f}  token acc {test_acc:.3f}")

    bucket_bleu, bucket_stats = length_bucket_metrics(results)
    print(f"  bucket BLEU: { {k: round(v, 2) for k, v in bucket_bleu.items()} }")

    run_dir = results_dir(1) / f"{model_type}_seed{seed}"
    save_json(bleu, run_dir / "test_bleu.json")
    save_json(bucket_bleu, run_dir / "bucket_bleu.json")
    save_json(bucket_stats, run_dir / "bucket_stats.json")
    save_jsonl(results, run_dir / "test_predictions.jsonl")
    if attn_weights:
        torch.save(attn_weights, attn_path(seed))
    print(f"  results saved to {run_dir}")
    return {"bleu": bleu, "bucket_bleu": bucket_bleu, "bucket_stats": bucket_stats}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q1 NMT models")
    parser.add_argument("--model", choices=["baseline", "attention", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN)
    parser.add_argument("--min-freq", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    train_pairs, _, test_pairs, _ = load_splits(
        max_train=args.max_train, max_len=args.max_len, seed=args.seed, min_len=args.min_len)
    vocab_en, vocab_hi, _ = build_vocabs(train_pairs, min_freq=args.min_freq)
    test_ds = NMTDataset(test_pairs, vocab_en, vocab_hi)

    models = ["baseline", "attention"] if args.model == "both" else [args.model]
    all_results = {mt: evaluate_model(mt, load_trained_model(mt, vocab_en, vocab_hi, device),
                                      test_ds, vocab_en, vocab_hi, device, args.seed)
                   for mt in models}

    if len(all_results) == 2:
        print("\n" + "=" * 60 + "\nCOMPARISON SUMMARY\n" + "=" * 60)
        for name, r in all_results.items():
            print(f"  {name:10s} BLEU {r['bleu']['bleu']:.2f}  chrF {r['bleu']['chrf']:.2f}  "
                  f"acc {r['bleu']['test_token_acc']:.3f}")
        print("Bucket BLEU:")
        for label in bucket_labels():
            print(f"  {label:>6s}  " + "  ".join(
                f"{n}: {r['bucket_bleu'].get(label, 0):.1f}" for n, r in all_results.items()))
        save_json({n: {"bleu": r["bleu"]["bleu"], "chrf": r["bleu"]["chrf"],
                       "test_token_acc": r["bleu"]["test_token_acc"],
                       "bucket_bleu": r["bucket_bleu"]}
                   for n, r in all_results.items()},
                  results_dir(1) / "model_comparison.json")


if __name__ == "__main__":
    main()
