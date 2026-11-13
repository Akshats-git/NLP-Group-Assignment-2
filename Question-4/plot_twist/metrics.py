"""Generation-quality metrics for Q4. All of them work on token lists (specials removed), never on detokenised text.

Definitions (these go verbatim into the report):

- Distinct-n (corpus level): unique n-grams / total n-grams, pooled over every story of one
  (model, decoding) group. Measures diversity across the group's outputs. Reported for n = 1, 2.
- Distinct-n (story level): the same ratio inside each story, then averaged. Measures how much a
  single story varies its own wording.
- Repetition rate: per story, the share of word trigrams that already appeared earlier in the same
  story (1 - unique trigrams / total trigrams), averaged over stories. 0 = no repeats, near 1 = stuck in a loop.
  Stories with fewer than 3 tokens count as 0.
- Degenerate share: share of stories whose repetition rate is >= 0.5, i.e. mostly looping text.
- Length: number of generated tokens up to (excluding) <eos>.
  too short = the story ended with <eos> after fewer than `short_below` tokens;
  too long  = no <eos> by the `max_new_tokens` cap (the story never finished).
"""

import statistics
from collections import Counter
from collections.abc import Sequence


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(stories: Sequence[Sequence[str]], n: int) -> float:
    """Corpus-level Distinct-n over a group of stories."""
    grams = [g for story in stories for g in ngrams(story, n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def distinct_n_story(story: Sequence[str], n: int) -> float:
    grams = ngrams(story, n)
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(story: Sequence[str], n: int = 3) -> float:
    grams = ngrams(story, n)
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def max_repeat(story: Sequence[str], n: int = 3) -> int:
    """How many times the most frequent n-gram occurs. A loop shows up as a large number."""
    counts = Counter(ngrams(story, n))
    return max(counts.values()) if counts else 0


def summarize_group(
    stories: Sequence[Sequence[str]],
    finished: Sequence[bool],
    max_new_tokens: int,
    short_below: int = 30,
    degenerate_at: float = 0.5,
) -> dict:
    """All the Part 5 quantitative numbers for one group of generations (e.g. LSTM + top-k 40)."""
    lengths = [len(s) for s in stories]
    rep = [repetition_rate(s) for s in stories]
    n = len(stories)
    return {
        "n_stories": n,
        "distinct_1": distinct_n(stories, 1),
        "distinct_2": distinct_n(stories, 2),
        "distinct_1_story": statistics.fmean(distinct_n_story(s, 1) for s in stories),
        "distinct_2_story": statistics.fmean(distinct_n_story(s, 2) for s in stories),
        "repetition_rate": statistics.fmean(rep),
        "degenerate_share": sum(r >= degenerate_at for r in rep) / n,
        "avg_length": statistics.fmean(lengths),
        "median_length": statistics.median(lengths),
        "too_short_share": sum(f and ln < short_below for ln, f in zip(lengths, finished)) / n,
        "too_long_share": sum(not f for f in finished) / n,
        "finished_share": sum(finished) / n,
    }
