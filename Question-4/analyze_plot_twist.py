"""Part 4: automatic evidence for the four plot-twist criteria, per generated story.

    python Question-4/analyze_plot_twist.py

The brief asks whether each continuation (1) stays connected to the prompt, (2) introduces a development,
(3) stays consistent with details established earlier, (4) avoids repetition and abrupt nonsense. Only some of
that can be measured, so this script reports measurable proxies and leaves the judgement to the report:

- prompt coverage    share of the prompt's content words (stop-words and punctuation removed) that the continuation
                     reuses. Low = drifted off the prompt, very high with low novelty = just echoing it.
- new content words  how many content-word types in the continuation are not in the prompt. 0 = no development.
- repetition rate    the Part 5 definition (repeated word trigrams), the loop detector.
- max repeat         how often the most repeated trigram occurs.
- names kept         share of the characters named in the opening (Sara, Jack, ... see prompts.PROMPT_NAMES) that the
                     continuation still uses; a name that disappears is a consistency failure.
- ended              produced <eos> within the cap.

Reads outputs/results/generations_seed42.jsonl (plot-twist prompts only) and writes plot_twist_proxies.json / .md.
"""

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import load_jsonl, markdown_table, save_json
from plot_twist import config as cfg
from plot_twist.data import tokenize
from plot_twist.metrics import max_repeat, repetition_rate
from plot_twist.prompts import PROMPT_NAMES

STOP = set("""a an the and or but if so to of in on at by for with from as is was were be been are am it its he she they we i you
his her their our my your him them us me this that these those there then than not no do did does had has have will would
could can said up down out into over under again very one all some any each when while what who which where how also just""".split())
PUNCT = re.compile(r"^[^a-z0-9]+$")


def content_words(tokens):
    return [t for t in tokens if t not in STOP and not PUNCT.match(t)]


def main():
    p = cfg.paths()
    rows = [r for r in load_jsonl(p["results"] / "generations_seed42.jsonl") if r["part"] == "plot_twist"]
    out = []
    for r in rows:
        prompt_tokens, gen = tokenize(r["prompt"]), r["tokens"]
        pw, gw = set(content_words(prompt_tokens)), set(content_words(gen))
        names = PROMPT_NAMES.get(r["prompt_key"], [])
        out.append({
            "id": r["id"], "model": r["model"], "prompt_key": r["prompt_key"], "decoding": r["decoding_label"],
            "sample_index": r["sample_index"], "n_tokens": r["n_tokens"], "ended": r["finished"],
            "prompt_coverage": len(pw & gw) / len(pw) if pw else 0.0, "new_content_words": len(gw - pw),
            "repetition_rate": repetition_rate(gen), "max_repeat": max_repeat(gen),
            "names_kept": (sum(n in gen for n in names) / len(names)) if names else None,  # None: the opening names no character
        })
    save_json(out, p["results"] / "plot_twist_proxies.json")

    groups = defaultdict(list)
    for o in out:
        groups[(o["model"], o["decoding"])].append(o)
    table = []
    for (model, dec), g in sorted(groups.items()):
        names = [x["names_kept"] for x in g if x["names_kept"] is not None]
        table.append({
            "model": cfg.MODEL_NAMES[model], "decoding": dec, "stories": len(g),
            "prompt coverage": statistics.fmean(x["prompt_coverage"] for x in g),
            "new content words": statistics.fmean(x["new_content_words"] for x in g),
            "repetition": statistics.fmean(x["repetition_rate"] for x in g),
            "max trigram repeat": statistics.fmean(x["max_repeat"] for x in g),
            "name kept": statistics.fmean(names) if names else float("nan"),
            "ended": statistics.fmean(x["ended"] for x in g),
        })
    md = ["## Plot-twist proxies (5 openings; all samples, per model and decoding)", "", markdown_table(table, floatfmt=".2f"), ""]
    (p["results"] / "plot_twist_proxies.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
