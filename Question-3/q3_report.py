"""Writes REPORT.md for Question 3 from the numbers the notebook produced.

Everything quantitative in the report (settings, sizes, losses, ROUGE, checks, examples)
is filled in from the run, so the report always matches the code that was executed.
Sentences that depend on the outcome (e.g. "still improving") are chosen from the numbers.
"""

import os
from pathlib import Path

from q3_eval import LEAD3_BRIEF, ROUGE_NAMES, ROUGE_TYPES


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _signed(x):
    return f"{x:+.2f}"


def _quote(text):
    return "\n".join("> " + line for line in text.split("\n"))


def _scores(scores, keys=("rouge1", "rouge2", "rougeLsum")):
    return " / ".join(f"{scores[k]:.1f}" for k in keys)


def write_report(path, v: dict) -> Path:
    """v is the dict assembled in section 10 of the notebook."""
    path = Path(path)

    def fig(name):
        return os.path.relpath(v["figures"][name], path.parent).replace(os.sep, "/")

    cfg, data, train, params = v["cfg"], v["data_info"], v["training"], v["params"]
    main_label = v["main_system"]
    main = v["test_rouge"][main_label]
    greedy = v["test_rouge"]["Greedy"]
    blocked = v["test_rouge"]["Greedy + trigram blocking"]
    lead = v["test_rouge"]["Lead-3 (our run)"]
    checks = v["text_checks"]
    history = v["history"]
    copy = cfg.get("copy_mechanism", False)
    out = []
    add = out.append

    add("# Question 3: Abstractive summarisation with a Transformer written from scratch\n")
    if v["smoke"]:
        add("> **SMOKE RUN.** Tiny model and data subsets, only to check that the code runs. "
            "These numbers are not results; rerun the notebook with `SMOKE = False`.\n")

    add("## Summary of the setup\n")
    add(_table(["Item", "Value"], [
        ["Dataset", f"CNN / Daily Mail {data['dataset']} (Hugging Face)"],
        ["Training-set size", f"{data['pairs']['train']:,} article-summary pairs "
                              f"({train['examples_seen']:,} pairs seen = {train['epochs']:.2f} epochs)"],
        ["Validation / test", f"{data['pairs']['validation']:,} / {data['pairs']['test']:,} pairs"],
        ["Tokenizer", f"`{data['tokenizer']}` WordPiece (uncased; only the tokenizer is pre-trained) "
                      f"+ `[NL]` sentence-break token; one shared vocabulary of {data['vocab_size']:,}"],
        ["Truncation", f"article {cfg['src_len']} tokens, summary {cfg['tgt_len']} tokens (both incl. start/end)"],
        ["Model", f"encoder-decoder, d_model {cfg['d_model']}, {cfg['n_heads']} heads, "
                  f"{cfg['n_enc_layers']} encoder + {cfg['n_dec_layers']} decoder layers, d_ff {cfg['d_ff']}, "
                  f"dropout {cfg['dropout']}, copy mechanism {'on' if copy else 'off'}"],
        ["Parameters", f"{params['total']:,} (all trained from random initialisation)"],
        ["Decoding", f"greedy, up to {cfg['max_summary_len']} tokens; also greedy + trigram blocking "
                     f"(main result: **{main_label}**, chosen on validation)"],
        ["Test ROUGE F1, " + main_label, " / ".join(f"{ROUGE_NAMES[k]} {main[k]['mean']:.2f}" for k in ROUGE_TYPES)],
    ]))
    add("")

    # ------------------------------------------------------------------ 1. data
    add("## 1. Data and preprocessing\n")
    add(f"We use CNN/DailyMail version 3.0.0 with the official splits: {data['pairs']['train']:,} training, "
        f"{data['pairs']['validation']:,} validation and {data['pairs']['test']:,} test pairs. The model learns "
        "only from the training split; the validation split is used for early stopping and for choosing the "
        "decoding setting; the test split is used once, for the final ROUGE scores.\n")
    add("Preprocessing, in order:\n")
    add("1. **Lowercasing** of articles and highlights, matching the uncased tokenizer.")
    add("2. **Sentence breaks.** The newline between highlight sentences becomes the special token `[NL]`, so "
        "the model learns where sentences end and its output can be written one sentence per line, as "
        "ROUGE-Lsum requires. The references used for scoring keep their original newlines.")
    add(f"3. **Sub-word tokenisation** with the pre-trained `{data['tokenizer']}` WordPiece tokenizer. Articles "
        f"and summaries share one vocabulary ({data['vocab_size']:,} entries). `[CLS]` is the start token and "
        "`[SEP]` the end token of every sequence; `[PAD]` (id 0) is padding.")
    add(f"4. **Truncation.** Articles keep their first {cfg['src_len']} tokens and summaries their first "
        f"{cfg['tgt_len']}; a truncated sequence still ends with `[SEP]`. News articles put the key facts first, "
        "so little is lost.")
    add("5. **Padding** of each batch to its longest sequence. Decoder input = summary shifted right (starts with "
        "`[CLS]`); target = summary shifted left (ends with `[SEP]`).\n")
    rows = []
    for split in ("train", "validation", "test"):
        for kind in ("article", "summary"):
            s = data["lengths"][split][kind]
            rows.append([split, kind, f"{s['median']:.0f}", f"{s['p90']:.0f}", s["max"], s["limit"],
                         f"{s['pct_truncated']:.1f}%"])
    add("Token lengths (including `[CLS]`/`[SEP]`) before truncation:\n")
    add(_table(["Split", "Text", "Median", "90th pct", "Max", "Limit", "Truncated"], rows))
    add("")

    # ------------------------------------------------------------------ 2. model
    add("## 2. Model: a Transformer written from scratch\n")
    add("All components are written with basic tensor operations and `nn.Linear`, `nn.Embedding`, "
        "`nn.LayerNorm` and `nn.Dropout` (file `q3_model.py`). No `nn.Transformer*`, "
        "`nn.MultiheadAttention`, `F.scaled_dot_product_attention`, Hugging Face model class or pre-trained "
        "weight is used; a check in the notebook scans the code and the model for them.\n")
    add("- **Positional encoding**: fixed sinusoids, PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), "
        "PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)), added to the token embeddings (scaled by sqrt(d_model)).")
    add("- **Multi-head attention**: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V with "
        f"{cfg['n_heads']} heads of size {cfg['d_model'] // cfg['n_heads']}, separate Q, K, V and output projections.")
    add(f"- **Position-wise feed-forward**: FFN(x) = max(0, xW1 + b1)W2 + b2 with inner size {cfg['d_ff']}.")
    add("- **Encoder layer**: self-attention, then feed-forward. **Decoder layer**: masked self-attention, "
        "encoder-decoder (cross) attention, then feed-forward. Every sub-layer is wrapped as "
        "LayerNorm(x + Dropout(Sublayer(x))).")
    add("- **Masks** (True = may attend): a source padding mask in encoder self-attention and in cross-attention; "
        "a causal (look-ahead) mask combined with a target padding mask in decoder self-attention.")
    add("- **Output**: a final linear layer maps decoder states to vocabulary logits. As in the paper, its weight "
        "is shared with the embedding (one matrix for encoder input, decoder input and output)."
        + ("" if copy else "\n"))
    if copy:
        add("- **Copy (pointer) switch** on top of the Transformer, following See et al. (2017). At every summary "
            "position a gate p_gen = sigmoid(W [decoder state; article context] + b) mixes two distributions: "
            "P(w) = p_gen * P_vocab(w) + (1 - p_gen) * (attention on the article positions that hold w). The copy "
            "distribution is the last decoder layer's cross-attention averaged over heads, and the context is that "
            "attention applied to the encoder output. The gate adds 1,025 parameters. Without it the decoder must "
            "produce every name and number from the vocabulary softmax, and in early training it writes fluent but "
            "unrelated text; with it, words can be taken straight from the article. The loss is -log P(target) of "
            "the mixed distribution, computed for the target word only.\n")
    add(_table(["Setting", "Value"], [
        ["d_model / heads / d_ff", f"{cfg['d_model']} / {cfg['n_heads']} / {cfg['d_ff']}"],
        ["Encoder / decoder layers", f"{cfg['n_enc_layers']} / {cfg['n_dec_layers']}"],
        ["Dropout", cfg["dropout"]],
        ["Vocabulary", f"{data['vocab_size']:,} (shared)"],
        ["Parameters", f"{params['total']:,} total, {params['trainable']:,} trainable"],
    ]))
    add("")

    # ------------------------------------------------------------------ 3. training
    add("## 3. Training\n")
    add(f"- **Loss**: token-level cross-entropy with label smoothing {cfg['label_smoothing']}; padding positions "
        "are excluded from the loss and from the token count. The validation loss, the loss curves and early "
        "stopping use plain (unsmoothed) cross-entropy, so train and validation numbers are comparable.")
    add("- **Teacher forcing**: the gold summary shifted right is the decoder input, so the whole summary is "
        "trained in one parallel pass; the causal mask stops each position from seeing later tokens.")
    add(f"- **Optimiser**: Adam, betas (0.9, 0.98), eps 1e-9, with the paper's schedule "
        f"lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5), warmup {cfg['warmup_steps']:,} steps "
        f"(peak lr {v['peak_lr']:.2e}). Gradients clipped to norm {cfg['clip_norm']}. Precision: {v['precision']}.")
    add(f"- **Batches**: {v['batch_size_total']} pairs per step on {v['hardware']}.")
    add(f"- **Early stopping**: validation loss on all {data['pairs']['validation']:,} validation pairs every "
        f"{cfg['eval_every']:,} steps, patience {cfg['patience']} checks, time budget {cfg['train_hours']} h. "
        "The best checkpoint is restored before any test evaluation.")
    add(f"- **Run**: {train['steps']:,} steps, {train['examples_seen']:,} pairs seen ({train['epochs']:.2f} epochs) "
        f"in {train['hours']:.2f} h. Training stopped because of: **{train['stop_reason']}**. Best validation "
        f"loss {train['best_val_loss']:.3f} (perplexity {train['best_val_ppl']:.1f}) at step "
        f"{train['best_step']:,}; that checkpoint is the one evaluated.\n")
    add(f"![Loss curves]({fig('loss_curves')})\n")
    first, best = history[0], min(history, key=lambda r: r["val_loss"])
    add(f"![Validation ROUGE during training]({fig('val_rouge')})\n")
    add(f"The second plot decodes the same {cfg['val_monitor_size']} validation articles greedily at every check. "
        f"Validation ROUGE-1 / 2 / Lsum went from {first['val_rouge1']:.1f} / {first['val_rouge2']:.1f} / "
        f"{first['val_rougeLsum']:.1f} at the first check (step {first['step']:,}) to {best['val_rouge1']:.1f} / "
        f"{best['val_rouge2']:.1f} / {best['val_rougeLsum']:.1f} at the best checkpoint. Watching free-running "
        "summaries, not only the loss, is what catches a broken causal mask (falling loss, useless summaries).")
    if "loss_right_article" in best:
        add(f"\nTo check that the summaries depend on the article, the same {cfg['val_monitor_size']} reference "
            "summaries were also scored against a different article (each batch rotated by one). At the first check "
            f"the loss was {first['loss_right_article']:.2f} with the right article vs "
            f"{first['loss_wrong_article']:.2f} with a wrong one; at the best checkpoint "
            f"{best['loss_right_article']:.2f} vs {best['loss_wrong_article']:.2f} (gap "
            f"{best['loss_wrong_article'] - best['loss_right_article']:.2f} nats per token). The model's predictions "
            "are strongly tied to the article it is given, not generic summary text."
            if best["loss_wrong_article"] - best["loss_right_article"] > 1.0 else
            f"\nThe same reference summaries scored against a different article give a loss of "
            f"{best['loss_wrong_article']:.2f} vs {best['loss_right_article']:.2f} with the right one: only a small "
            "gap, so the model still relies a lot on generic summary text.")
        if best.get("p_gen") is not None:
            add(f"The average copy gate on validation summaries is p_gen = {best['p_gen']:.2f}: the model puts about "
                f"{100 * (1 - best['p_gen']):.0f}% of the weight on copying from the article and "
                f"{100 * best['p_gen']:.0f}% on generating from the vocabulary.")
    if train["best_step"] == history[-1]["step"]:
        add("The best validation loss was at the last check, so the model was still improving when the time "
            "budget ended; a longer run would very likely score higher.\n")
    else:
        add(f"The validation loss stopped improving after step {train['best_step']:,}.\n")
    add("**Checks run before training** (all must pass, otherwise the notebook stops):\n")
    add(_table(["Check", "Result"], [[name, "pass" if ok else "FAIL"] for name, ok in v["sanity"].items()]))
    add("")

    # ------------------------------------------------------------------ 4. decoding
    add("## 4. Decoding\n")
    add("Generation uses no teacher forcing. Each article is encoded once; the decoder starts from `[CLS]`, "
        "and at every step the most likely token (greedy) is appended and fed back in, until `[SEP]` or "
        f"{cfg['max_summary_len']} tokens. `[PAD]`, `[CLS]` and `[UNK]` are never generated. Token ids are "
        "converted back to text and split into lines at `[NL]` before scoring.\n")
    add("We also tried greedy decoding with **trigram blocking**: a token is not allowed if it would repeat a "
        "trigram already in the summary (Paulus et al., 2018), a common fix for the repetition greedy decoding "
        f"causes. The choice between the two was made on {v['decode_choice_size']:,} validation articles, never "
        "on the test set:\n")
    add(_table(["Decoding (validation)"] + [ROUGE_NAMES[k] for k in ROUGE_TYPES],
               [[name] + [f"{s[k]:.2f}" for k in ROUGE_TYPES] for name, s in v["decoding_choice"].items()]))
    add(f"\nMain result: **{main_label}**. Plain greedy, which the brief requires, is reported as well.\n")

    # ------------------------------------------------------------------ 5. results
    add("## 5. Results on the test set\n")
    add(f"F1 scores (x100) on all {v['n_test']:,} test pairs with `rouge_score` (use_stemmer=True); 95% "
        "bootstrap confidence intervals in brackets. **ROUGE-L** treats each summary as one sequence; "
        "**ROUGE-Lsum** splits both summaries into sentences at newlines and is the usual CNN/DM metric.\n")
    rows = [["Lead-3 (brief)"] + [f"{LEAD3_BRIEF[k]:.2f}" for k in ROUGE_TYPES]]
    for name, s in v["test_rouge"].items():
        rows.append([name] + [f"{s[k]['mean']:.2f} [{s[k]['ci_low']:.2f}, {s[k]['ci_high']:.2f}]" for k in ROUGE_TYPES])
    add(_table(["System"] + [ROUGE_NAMES[k] for k in ROUGE_TYPES], rows))
    add("")
    ours = ", ".join(f"{lead[k]['mean']:.2f}" for k in ROUGE_TYPES)
    brief = ", ".join(f"{LEAD3_BRIEF[k]:.2f}" for k in ROUGE_TYPES)
    same = all(abs(lead[k]["mean"] - LEAD3_BRIEF[k]) < 0.05 for k in ROUGE_TYPES)
    add(f"Our own Lead-3 run gives {ours} against the brief's {brief} (sentence splitter: {v['splitter']}); "
        + ("identical, which confirms the scoring pipeline is the standard one.\n" if same else
           "close; the small difference comes from sentence splitting.\n"))
    add(f"![ROUGE comparison]({fig('rouge_comparison')})\n")

    gaps = {k: main[k]["mean"] - lead[k]["mean"] for k in ROUGE_TYPES}
    add("### Comparison with Lead-3\n")
    below = [ROUGE_NAMES[k] for k in ROUGE_TYPES if gaps[k] < 0]
    add(f"Relative to Lead-3, {main_label} scores {_signed(gaps['rouge1'])} ROUGE-1, {_signed(gaps['rouge2'])} "
        f"ROUGE-2, {_signed(gaps['rougeL'])} ROUGE-L and {_signed(gaps['rougeLsum'])} ROUGE-Lsum points"
        + (f", so it is below Lead-3 on {', '.join(below)}. " if below else ", so it is at or above Lead-3 on "
           "every metric. ")
        + "The brief expects a from-scratch model with limited compute to score below Lead-3. "
        "What separates the two systems:\n")
    add("- **Lead bias of news.** Journalists put the key facts in the first sentences and the highlights "
        "paraphrase them. Lead-3 copies those sentences word for word, so it gets facts, names and numbers "
        "exactly right at no cost. Our model has to learn language, content selection and generation from "
        "random weights.")
    add(f"- **Small compute.** {train['epochs']:.2f} epochs in {train['hours']:.1f} h on {v['hardware']}, with "
        f"{params['total'] / 1e6:.0f}M parameters and no pre-training. Published Transformer baselines train "
        "for days on several GPUs, and pre-trained models (BART, PEGASUS) add large-scale pre-training on top.")
    if copy:
        add("- **Copying is learned, not exact.** The copy switch lets the model take names and numbers from the "
            "article, but it still has to learn which ones belong in the summary and when to copy rather than "
            "generate; mistakes show up as wrong or swapped entities (see the checks below).")
    else:
        add(f"- **No copy mechanism.** Rare names and numbers have to come out of a {data['vocab_size']:,}-way "
            "softmax, which is where wrong entities and numbers come from (see the checks below). Pointer-generator "
            "networks copy them from the article instead.")
    add("- **Greedy decoding** commits to one token at a time; beam search usually adds some ROUGE.")
    a_trunc = data["lengths"]["test"]["article"]["pct_truncated"]
    add(f"- **Truncation** to {cfg['src_len']} tokens cuts {a_trunc:.0f}% of test articles, so highlights drawn "
        "from later in the article cannot be found. Lead-3 only reads the start and is not affected.")
    add("- **ROUGE measures word overlap.** A correct paraphrase scores lower than a copied sentence, and an "
        "abstractive model paraphrases (see the novel-bigram rate below).\n")
    lower_everywhere = all(s["rougeL"]["mean"] < s["rougeLsum"]["mean"] for s in v["test_rouge"].values())
    add(f"ROUGE-L and ROUGE-Lsum are different metrics ({main_label}: ROUGE-L {main['rougeL']['mean']:.2f}, "
        f"ROUGE-Lsum {main['rougeLsum']['mean']:.2f}"
        + ("; ROUGE-L is lower for every system). " if lower_everywhere else "). ")
        + "Plain ROUGE-L needs one common subsequence across the whole summary, so content in a different "
        "sentence order is lost; ROUGE-Lsum matches each reference sentence against the generated sentences "
        "separately, which is why it needs the newlines.\n")

    add("### What the generated summaries look like\n")
    names = list(checks)
    add(_table(["Measured on the test set"] + names, [
        ["Average length (words)"] + [f"{checks[n]['avg_words']:.1f}" for n in names],
        ["Average sentences"] + [f"{checks[n]['avg_sentences']:.2f}" for n in names],
        ["Summaries with a repeated word trigram"] + [f"{checks[n]['pct_repeated_trigram']:.1f}%" for n in names],
        ["Summaries with a number not in the article"] + [f"{checks[n]['pct_unsupported_number']:.1f}%" for n in names],
        ["Bigrams not in the article (abstractiveness)"] + [f"{checks[n]['pct_novel_bigrams']:.1f}%" for n in names],
        ["Empty summaries"] + [f"{checks[n]['pct_empty']:.2f}%" for n in names],
    ]))
    add("")
    g, b = checks["Greedy"], checks["Greedy + trigram blocking"]
    add(f"Plain greedy decoding repeats a word trigram in {g['pct_repeated_trigram']:.1f}% of summaries "
        f"(references: {checks['Reference']['pct_repeated_trigram']:.1f}%). Trigram blocking brings this to "
        f"{b['pct_repeated_trigram']:.1f}% (blocking works on sub-word tokens and allows trigrams around a "
        "sentence break, so word-level repeats can remain) and changes test ROUGE-Lsum by "
        f"{_signed(blocked['rougeLsum']['mean'] - greedy['rougeLsum']['mean'])} points. "
        f"{g['pct_unsupported_number']:.1f}% of greedy summaries contain a number that never occurs in the "
        f"article, against {checks['Reference']['pct_unsupported_number']:.1f}% of the human references (which "
        "sometimes compute or reword numbers). The check is rough, it only compares digit strings, but it "
        "points at the main factual weakness: numbers and names are generated, not copied.\n")

    # ------------------------------------------------------------------ 6. qualitative
    add("## 6. Qualitative examples\n")
    add("Test articles chosen automatically to cover good and bad cases (best, median and worst greedy "
        "ROUGE-Lsum, a repeated phrase, a number not in the article, one random). Articles are cut to their "
        "first 120 words; scores are ROUGE-1 / ROUGE-2 / ROUGE-Lsum.\n")
    for k, ex in enumerate(v["qualitative"], 1):
        add(f"### Example {k}: {ex['reason']} (test id `{ex['id']}`)\n")
        add("**Article (start):**\n")
        add(_quote(ex["article_start"]) + "\n")
        add("**Reference:**\n")
        add(_quote(ex["reference"]) + "\n")
        add(f"**Greedy** ({_scores(ex['scores']['Greedy'])}):\n")
        add(_quote(ex["greedy"] or "(empty)") + "\n")
        add(f"**Greedy + trigram blocking** ({_scores(ex['scores']['Greedy + trigram blocking'])}):\n")
        add(_quote(ex["blocked"] or "(empty)") + "\n")
        add("**Automatic notes:** " + " ".join(ex["notes"]) + "\n")

    # ------------------------------------------------------------------ 7. attention
    att = v["attention"]
    add("## 7. Encoder-decoder attention\n")
    add("The heatmaps show the cross-attention of the **last decoder layer**, recorded while the model generated "
        f"the summary greedily (not under teacher forcing), **averaged over the {cfg['n_heads']} heads**. Rows are "
        "generated summary tokens, columns are article tokens (WordPiece). A whole article is unreadable, so each "
        f"plot shows the {att['window']}-token window of the article that receives the most attention."
        + (" With the copy switch on, this head-averaged attention is also the distribution the model copies "
           "article words from.\n" if copy else "\n"))
    for item in att["examples"]:
        add(f"**{item['title']}**, article tokens {item['start']} to {item['start'] + item['width'] - 1}:\n")
        add(f"![Cross-attention]({fig(item['figure'])})\n")
    add(f"The first example with each head shown separately:\n\n![Cross-attention per head]({fig('attention_heads')})\n")
    bins = att["position_bins"]
    first100 = bins[0] + bins[1]
    add(f"Over {att['n_articles']} test summaries, {first100:.0f}% of the cross-attention mass falls on the first 100 "
        f"article tokens, which are {att['first100_share_of_source']:.0f}% of the article tokens the model sees:\n")
    add(f"![Attention by source position]({fig('attention_by_position')})\n")
    if first100 > 1.5 * att["first100_share_of_source"]:
        add("The decoder concentrates on the opening of the article, the same region Lead-3 copies: it has picked "
            "up the lead bias of news from the data on its own.")
    else:
        add("Attention is spread over the article rather than concentrated on its opening.")
    if att["copy_alignment"] == att["copy_alignment"]:   # not NaN
        add(f"For {att['copy_alignment']:.0f}% of generated word tokens that also occur in the article, the "
            "article token with the highest (head-averaged) attention is that same token: "
            + ("cross-attention often works as a soft pointer to the words being reproduced. "
               if att["copy_alignment"] >= 50 else "the strongest attention is usually somewhere else, so "
               "attention is not a simple pointer to the words being reproduced. "))
    add("Attention shows where information was read from; it is not proof that the model understood it, and "
        "high attention on the right word does not guarantee a correct summary.\n")

    # ------------------------------------------------------------------ 8. pitfalls
    add("## 8. How the common pitfalls were avoided\n")
    add(_table(["Pitfall from the brief", "What we did", "Evidence"], [
        ["1. nn.Transformer / nn.MultiheadAttention / pre-trained model",
         "All layers written in `q3_model.py`; only the tokenizer is pre-trained", "check `no_banned_layers`"],
        ["2. Missing or wrong causal mask", "Lower-triangular mask combined with the target padding mask",
         "check `causal_mask`; validation ROUGE of free-running output tracked during training"],
        ["3. Attending to padding", "Source padding mask in encoder and cross-attention; target padding mask",
         "checks `source_padding_mask`, `target_padding_mask`"],
        ["4. Padding in the loss", "Loss only over non-padding target tokens", "check `loss_ignores_padding`"],
        ["5. ROUGE on teacher-forced output", "ROUGE only on free-running greedy summaries",
         "`summarize_arrays` never sees the reference"],
        ["6. Unlabelled ROUGE-L", "ROUGE-L and ROUGE-Lsum both reported and labelled", "section 5"],
        ["7. Tuning on test / testing the last checkpoint", "Early stopping and decoding choice on validation; "
         "best checkpoint restored before testing", "sections 3 and 4"],
    ]))
    add("")

    # ------------------------------------------------------------------ 9-10
    add("## 9. Limitations and what would help next\n")
    add(f"- One run, one seed, {train['hours']:.1f} h of training. More training is the cheapest improvement.")
    add("- Greedy decoding only; beam search with a length penalty is the standard next step.")
    add("- No coverage mechanism, so phrases can repeat (trigram blocking only hides this at decoding time)."
        if copy else "- No copy or coverage mechanism, so rare names and numbers are often wrong and phrases can repeat.")
    add(f"- Articles are truncated to the first {cfg['src_len']} tokens.")
    add("- ROUGE and the number check do not measure factual correctness; the examples above need to be read "
        "against their articles.\n")
    add("## 10. Reproducing the results\n")
    add("See `Question-3/README.md`: open `Question-3/Question_3.ipynb` on Kaggle with GPU and Internet on and run "
        f"all cells. All settings are in the `CFG` cell and the seed is fixed ({cfg['seed']}). "
        f"Environment of this run: {v['environment']}.\n")
    path.write_text("\n".join(out), encoding="utf-8")
    return path
