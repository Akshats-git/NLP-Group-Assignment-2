"""ROUGE, the Lead-3 baseline and simple checks on generated summaries.

ROUGE follows the brief: the rouge_score package, use_stemmer=True, F1 scores, reported
as 0-100. ROUGE-L treats the whole summary as one sequence; ROUGE-Lsum works sentence by
sentence and needs one sentence per line in both the reference and the prediction.
"""

import re

import numpy as np

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")
ROUGE_NAMES = {"rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L", "rougeLsum": "ROUGE-Lsum"}

# Lead-3 on the full test set as given in the brief (rouge_score, stemming on).
LEAD3_BRIEF = {"rouge1": 40.06, "rouge2": 17.48, "rougeL": 25.02, "rougeLsum": 36.36}


def rouge_per_example(predictions, references) -> list[dict]:
    """F1 for every pair, scaled to 0-100. References keep their original newlines."""
    from rouge_score import rouge_scorer

    if len(predictions) != len(references):
        raise ValueError("predictions and references differ in length")
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
    rows = []
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)          # score(target, prediction)
        rows.append({k: 100 * scores[k].fmeasure for k in ROUGE_TYPES})
    return rows


def rouge_summary(rows, n_boot=1000, seed=0) -> dict:
    """Mean F1 per ROUGE type with a 95% bootstrap confidence interval over examples."""
    rng = np.random.default_rng(seed)
    resample = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    out = {}
    for k in ROUGE_TYPES:
        values = np.array([row[k] for row in rows])
        boot_means = values[resample].mean(axis=1)
        out[k] = {"mean": float(values.mean()), "ci_low": float(np.percentile(boot_means, 2.5)),
                  "ci_high": float(np.percentile(boot_means, 97.5))}
    return out


def rouge_means(rows) -> dict:
    return {k: float(np.mean([row[k] for row in rows])) for k in ROUGE_TYPES}


# ---------------------------------------------------------------- Lead-3 baseline

def sentence_splitter():
    """NLTK's Punkt splitter (what the brief's Lead-3 numbers use), or a regex fallback."""
    try:
        import nltk
        for package in ("punkt_tab", "punkt"):
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
        from nltk.tokenize import sent_tokenize
        sent_tokenize("Check it works. Yes.")
        return sent_tokenize, "nltk"
    except Exception:
        pattern = re.compile(r"(?<=[.!?])\s+(?=[\"'A-Z(])")
        return (lambda text: [s for s in pattern.split(text.strip()) if s]), "regex"


def lead3(article: str, split) -> str:
    """The first three sentences of the article, one per line."""
    return "\n".join(split(article)[:3])


# ---------------------------------------------------------------- checks on generated text

def words(text: str) -> list[str]:
    """Lowercase alphanumeric words, the same tokenisation rouge_score uses."""
    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def repeated_trigrams(text: str) -> list[str]:
    """Word trigrams that occur more than once, in order of first repeat."""
    seen, repeats = set(), []
    for gram in ngrams(words(text), 3):
        phrase = " ".join(gram)
        if phrase in seen and phrase not in repeats:
            repeats.append(phrase)
        seen.add(phrase)
    return repeats


def has_repeated_trigram(text: str) -> bool:
    return bool(repeated_trigrams(text))


def unsupported_numbers(summary: str, article: str) -> list[str]:
    """Digit strings in the summary that never appear in the article (a rough factuality flag)."""
    in_article = set(re.findall(r"\d+", article))
    return sorted(set(re.findall(r"\d+", summary)) - in_article)


def novel_bigram_rate(summary: str, article: str) -> float:
    """Share of summary bigrams that are not in the article (how abstractive the output is)."""
    summary_bigrams = ngrams(words(summary), 2)
    if not summary_bigrams:
        return 0.0
    article_bigrams = set(ngrams(words(article), 2))
    return sum(b not in article_bigrams for b in summary_bigrams) / len(summary_bigrams)


def text_checks(summary: str, article: str) -> dict:
    return {
        "n_words": len(words(summary)),
        "n_sentences": len([line for line in summary.split("\n") if line.strip()]),
        "repeated_trigram": has_repeated_trigram(summary),
        "unsupported_numbers": unsupported_numbers(summary, article),
        "novel_bigrams": novel_bigram_rate(summary, article),
        "empty": not summary.strip(),
    }


def aggregate_checks(checks: list[dict]) -> dict:
    return {
        "avg_words": float(np.mean([c["n_words"] for c in checks])),
        "avg_sentences": float(np.mean([c["n_sentences"] for c in checks])),
        "pct_repeated_trigram": float(100 * np.mean([c["repeated_trigram"] for c in checks])),
        "pct_unsupported_number": float(100 * np.mean([bool(c["unsupported_numbers"]) for c in checks])),
        "pct_novel_bigrams": float(100 * np.mean([c["novel_bigrams"] for c in checks])),
        "pct_empty": float(100 * np.mean([c["empty"] for c in checks])),
    }


def describe_example(greedy: str, blocked: str, reference: str, article: str, rouge_lsum: float,
                     median_lsum: float) -> list[str]:
    """Short factual notes on one qualitative example, built from the checks above."""
    notes = [f"Greedy ROUGE-Lsum {rouge_lsum:.1f} (test median {median_lsum:.1f});",
             f"{len(words(greedy))} words vs {len(words(reference))} in the reference."]
    if not greedy.strip():
        notes.append("The model produced an empty summary.")
    repeats = repeated_trigrams(greedy)
    if repeats:
        notes.append(f"Greedy output repeats \"{repeats[0]}\"" + (" and more." if len(repeats) > 1 else "."))
        if not has_repeated_trigram(blocked):
            notes.append("Trigram blocking removes the repetition.")
    numbers = unsupported_numbers(greedy, article)
    if numbers:
        notes.append(f"Numbers not in the article: {', '.join(numbers)}.")
    if not repeats and not numbers and greedy.strip():
        notes.append("No repeated phrase and no number missing from the article; names and facts still "
                     "have to be checked against the article by reading it.")
    return notes


def pick_qualitative(rows: list[dict], seed=0) -> list[tuple[str, int]]:
    """Choose at least five test examples to show, including failure cases.

    rows: one dict per test example with "rougeLsum", "repeated_trigram",
    "unsupported_numbers" and "empty". Returns [(why it was picked, index)].
    """
    order = np.argsort([row["rougeLsum"] for row in rows], kind="stable")
    nonempty = [int(i) for i in order if not rows[i]["empty"]]
    picks = {}

    def add(reason, candidates):
        for i in candidates:
            if i not in picks.values():
                picks[reason] = int(i)
                return

    def middle_first(indices):                 # a typical case of a failure type, not its most extreme one
        mid = len(indices) // 2
        return [indices[j] for j in sorted(range(len(indices)), key=lambda j: abs(j - mid))]

    # Failure types first, so another category cannot take their only example
    add("repeated phrase", middle_first([i for i in nonempty if rows[i]["repeated_trigram"]]))
    add("number not in the article", middle_first([i for i in nonempty if rows[i]["unsupported_numbers"]]))
    add("highest ROUGE-Lsum", nonempty[::-1])
    add("median ROUGE-Lsum", middle_first(nonempty))
    add("lowest ROUGE-Lsum", nonempty)
    rng = np.random.default_rng(seed)
    add("random example", rng.permutation(len(rows)).tolist())
    while len(picks) < 5:                      # only if a failure type never occurs
        add(f"random example {len(picks)}", rng.permutation(len(rows)).tolist())
    shown = ["highest ROUGE-Lsum", "median ROUGE-Lsum", "repeated phrase", "number not in the article",
             "lowest ROUGE-Lsum"]
    return [(r, picks[r]) for r in shown if r in picks] + [(r, i) for r, i in picks.items() if r not in shown]


def attention_by_position(attention_rows: list[np.ndarray], bin_size=50, max_len=400) -> list[float]:
    """Share of cross-attention mass in each block of `bin_size` source positions.

    attention_rows: one [generated tokens, source tokens] array per summary.
    """
    bins = np.zeros(int(np.ceil(max_len / bin_size)))
    for weights in attention_rows:
        mass = weights.sum(axis=0)                                  # attention each source token got
        for start in range(0, weights.shape[1], bin_size):
            bins[start // bin_size] += mass[start:start + bin_size].sum()
    return (100 * bins / bins.sum()).tolist()


def copy_alignment(attention_rows, generated_tokens, source_tokens) -> float:
    """% of generated word tokens found in the article whose most-attended article token is the same token.

    attention_rows[i]: [len(generated_tokens[i]), len(source_tokens[i])] weights (head-averaged).
    Special tokens ([CLS], [NL], ...) and pure punctuation are skipped.
    """
    hits = total = 0
    for weights, generated, source in zip(attention_rows, generated_tokens, source_tokens):
        source_set = set(source)
        for row, token in enumerate(generated):
            if token.startswith("[") or not any(ch.isalnum() for ch in token) or token not in source_set:
                continue
            total += 1
            hits += source[int(weights[row].argmax())] == token
    return 100 * hits / total if total else float("nan")
