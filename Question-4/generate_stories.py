"""Parts 3 and 4: continue the 10 prompts with both trained models under every decoding configuration.

    python Question-4/generate_stories.py            # uses the seed-42 checkpoints
    python Question-4/generate_stories.py --seed 43

Part 3 prompts (mystery, discovery, unusual character, problem, unexpected ending) and the five
Part 4 plot-twist openings all go through the same grid, so the decoding comparison and the
plot-twist challenge come from one set of unedited generations:

    model (RNN, LSTM) x decoding (greedy, temp 0.5 / 0.8 / 1.2, top-k 10 / 40) x N samples per prompt

Writes:
    outputs/results/generations_seed<seed>.jsonl   every story with prompt, model, decoding, seed and metrics
    outputs/results/generated_stories_seed<seed>.md  the same stories for reading; model text only, no commentary
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import Vocab, get_device, save_json, save_jsonl
from plot_twist import config as cfg
from plot_twist.generation import DECODING_CONFIGS, derive_seed, generate_from_text, prompt_text_for_display
from plot_twist.metrics import repetition_rate
from plot_twist.models import load_model
from plot_twist.prompts import ALL_PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=cfg.GENERATION["seed"], help="base seed for sampling")
    ap.add_argument("--train-seed", type=int, default=42, help="which trained checkpoints to use")
    args = ap.parse_args()

    p, g = cfg.paths(), cfg.GENERATION
    device = get_device()
    vocab = Vocab.load(p["results"].parent / "vocab.json")
    models = {cell: load_model(p["checkpoints"] / f"{cell}_seed{args.train_seed}.pt", device)[0] for cell in cfg.CELLS}

    rows = []
    for prompt_key, prompt in ALL_PROMPTS.items():
        part = "plot_twist" if prompt_key.startswith("twist_") else "decoding"
        for cell, model in models.items():
            for decoding in DECODING_CONFIGS:
                n = 1 if decoding.strategy == "greedy" else g["n_samples"]
                seed = derive_seed(args.seed, prompt_key, decoding.label)
                for i, row in enumerate(generate_from_text(model, vocab, prompt, decoding, n, seed, g["max_new_tokens"], device)):
                    rows.append({
                        "id": f"{cell}|{prompt_key}|{decoding.label}|{i}", "part": part, "prompt_key": prompt_key,
                        "prompt": prompt, "model": cell, "train_seed": args.train_seed, "decoding": decoding.to_dict(),
                        "decoding_label": decoding.label, "sample_index": i, "sampling_seed": seed,
                        "max_new_tokens": g["max_new_tokens"], "text": row["text"], "n_tokens": row["n_tokens"],
                        "finished": row["finished"], "repetition_rate": repetition_rate(row["tokens"]),
                        "tokens": row["tokens"],
                    })
        print("done", prompt_key)

    suffix = f"seed{args.seed}" + ("" if args.train_seed == 42 else f"_models{args.train_seed}")
    save_jsonl(rows, p["results"] / f"generations_{suffix}.jsonl")
    save_json({"generation": g, "decoding_configs": [c.to_dict() for c in DECODING_CONFIGS], "sampling_seed": args.seed,
              "train_seed": args.train_seed, "prompts": ALL_PROMPTS}, p["results"] / f"generation_settings_{suffix}.json")

    lines = ["# Generated stories (unedited model output)", "",
             f"Checkpoints: seed {args.train_seed}. Sampling seed: {args.seed}. Cap: {g['max_new_tokens']} new tokens. "
             "Text is exactly what the model produced after detokenisation (lowercase tokens re-capitalised for reading); "
             "it was not edited. `[no <eos>]` means the story hit the length cap without ending.", ""]
    for prompt_key, prompt in ALL_PROMPTS.items():
        lines += [f"## {prompt_key}", "", "**Prompt (given to the model):**", "", f"> {prompt_text_for_display(prompt)}", ""]
        for cell in cfg.CELLS:
            for decoding in DECODING_CONFIGS:
                group = [r for r in rows if r["prompt_key"] == prompt_key and r["model"] == cell and r["decoding_label"] == decoding.label]
                for r in group:
                    flag = "" if r["finished"] else " [no <eos>]"
                    lines += [f"**{cfg.MODEL_NAMES[cell]} / {decoding.label} / sample {r['sample_index'] + 1}** "
                              f"({r['n_tokens']} tokens, repetition {r['repetition_rate']:.2f}){flag}", "",
                              "> " + r["text"].replace("\n", "\n> "), ""]
    (p["results"] / f"generated_stories_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(rows)} stories written")


if __name__ == "__main__":
    main()
