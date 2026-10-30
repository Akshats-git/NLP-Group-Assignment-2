"""Optional extension: one small open-source LLM on the same openings, zero-shot, no fine-tuning.

    python Question-4/llm_baseline.py                          # SmolLM2-360M-Instruct
    python Question-4/llm_baseline.py --model Qwen/Qwen2.5-0.5B-Instruct

The LLM does not replace either from-scratch model. It gets the same ten openings, the same greedy and
temperature-0.8 / top-k-40 decoding settings as the RNN/LSTM, and the same cap on new tokens (its own tokens
are sub-words, so 250 of them cover somewhat less text than 250 word-level tokens; noted in the report).

Writes outputs/results/llm_generations.jsonl and llm_settings.json (checkpoint, prompt template, decoding, precision, hardware).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from common import get_device, save_json, save_jsonl, set_seed
from plot_twist import config as cfg
from plot_twist.generation import derive_seed
from plot_twist.metrics import repetition_rate
from plot_twist.data import tokenize
from plot_twist.prompts import ALL_PROMPTS

SYSTEM = "You are a children's story writer."
TEMPLATE = ("Here is the beginning of a short story for young children. Continue the story in simple language and "
            "give it an ending. Write only the continuation, in plain text.\n\nBeginning: {opening}")
SETTINGS = {
    "greedy": dict(do_sample=False),
    "temp=0.8": dict(do_sample=True, temperature=0.8, top_k=0, top_p=1.0),
    "top-k=40": dict(do_sample=True, temperature=1.0, top_k=40, top_p=1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--n-samples", type=int, default=cfg.GENERATION["n_samples"])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    p, g = cfg.paths(), cfg.GENERATION
    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    rows = []
    for prompt_key, opening in ALL_PROMPTS.items():
        chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TEMPLATE.format(opening=opening)}]
        inputs = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
        for label, kw in SETTINGS.items():
            n = 1 if not kw["do_sample"] else args.n_samples
            seed = derive_seed(g["seed"], "llm", prompt_key, label)
            set_seed(seed)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=g["max_new_tokens"], num_return_sequences=n,
                                     pad_token_id=tok.eos_token_id, **kw)
            for i in range(n):
                new = out[i, inputs["input_ids"].shape[1]:]
                finished = bool((new == tok.eos_token_id).any())
                text = tok.decode(new, skip_special_tokens=True).strip()
                rows.append({"id": f"llm|{prompt_key}|{label}|{i}", "part": "plot_twist" if prompt_key.startswith("twist_") else "decoding",
                             "prompt_key": prompt_key, "prompt": opening, "model": "llm", "decoding_label": label,
                             "sample_index": i, "sampling_seed": seed, "text": text, "finished": finished,
                             "n_tokens": int((new != tok.eos_token_id).sum()), "repetition_rate": repetition_rate(tokenize(text))})
        print("done", prompt_key)

    save_jsonl(rows, p["results"] / "llm_generations.jsonl")
    save_json({"checkpoint": args.model, "system_prompt": SYSTEM, "user_prompt_template": TEMPLATE, "decoding": SETTINGS,
               "max_new_tokens": g["max_new_tokens"], "precision": str(dtype), "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "fine_tuned": False,
               "shots": 0}, p["results"] / "llm_settings.json")


if __name__ == "__main__":
    main()
