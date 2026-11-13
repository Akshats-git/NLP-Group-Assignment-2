"""Q2 live demo: 5-class Amazon review rating, BiGRU + attention vs fine-tuned RoBERTa.

    # from Question-2/
    ../.venv/bin/streamlit run demo/app.py
"""

import hashlib
import html
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import inference  # noqa: E402  (sets offline mode and sys.path before anything imports transformers)

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import results  # noqa: E402
from results import BIGRU, FIRST, MODELS, ROBERTA  # noqa: E402

st.set_page_config(page_title="Review Rating Demo", page_icon="⭐", layout="wide")

LIVE, RESULTS, BROWSE, HOW = "🔴 Live demo", "📊 Test-set results", "🔎 Browse test predictions", "🧠 How it works"
COLOR = {name: colour for name, (_, colour) in MODELS.items()}
GOLD = "#eda100"
FOR_RGB, AGAINST_RGB, ATTN_RGB = "27,175,122", "227,73,72", "42,120,214"

st.markdown("""
<style>
.toks {line-height: 2.15; font-size: 0.97rem; margin: 0.2rem 0 0.4rem 0;}
.tok {padding: 2px 4px; border-radius: 5px; margin: 0 1px; white-space: pre-wrap;}
.unk {text-decoration: underline wavy rgba(227,73,72,.95); text-underline-offset: 4px;}
.stars {font-size: 2.1rem; letter-spacing: 2px; line-height: 1.1; color: #eda100;}
.stars .off {opacity: .25;}
.legend {font-size: .82rem; opacity: .8;}
.chip {display: inline-block; padding: 1px 6px; margin: 2px; border-radius: 4px; font-family: monospace;
       font-size: .82rem; border: 1px solid rgba(128,128,128,.35);}
</style>
""", unsafe_allow_html=True)


# ======================================================================= cached resources
@st.cache_resource(show_spinner=False)
def get_demo() -> inference.Demo:
    demo = inference.Demo()
    demo.warm_up([r for r in ("bigru_seed42", "roberta_mean_seed42", "roberta_first_seed42")
                  if r in inference.available_runs()])
    return demo


@st.cache_data(max_entries=512, show_spinner=False)
def predict(model: str, text: str, seed: int) -> inference.Prediction:
    demo = get_demo()
    return demo.predict_bigru(text, seed) if model == "bigru" else demo.predict_roberta(text, model, seed)


@st.cache_data(max_entries=128, show_spinner=False)
def explain(text: str, pooling: str, seed: int, target: int, steps: int) -> inference.Explanation:
    return get_demo().explain_roberta(text, pooling, seed, target, steps)


@st.cache_data(show_spinner=False)
def test_table() -> pd.DataFrame:
    from q2.data import load_reviews
    test = load_reviews(("test",))["test"]
    texts = pd.DataFrame({"id": test["id"], "text": test["text"]})
    return results.saved_predictions().merge(texts, on="id")


cached_examples = st.cache_data(results.report_examples)
cached_metrics = st.cache_data(results.metrics_table)
cached_headline = st.cache_data(results.headline)


# ======================================================================= small views
def stars_html(n: int) -> str:
    return f'<div class="stars">{"★" * n}<span class="off">{"★" * (5 - n)}</span></div>'


def tokens_html(words, weights, mode: str, unknown=None) -> str:
    """mode "attention": one colour, darker = more weight. mode "ig": green pushes towards the target, red away."""
    w = np.asarray(weights, dtype=float)
    scale = float(np.abs(w).max()) or 1.0
    spans = []
    for i, (word, x) in enumerate(zip(words, w)):
        alpha = 0.65 * min(abs(x) / scale, 1.0)
        rgb = ATTN_RGB if mode == "attention" else (FOR_RGB if x >= 0 else AGAINST_RGB)
        tip = f"attention {x:.1%}" if mode == "attention" else f"{x:+.3f}"
        cls = "tok"
        if unknown is not None and unknown[i]:
            cls += " unk"
            tip += " · not in the BiGRU vocabulary, read as <unk>"
        spans.append(f'<span class="{cls}" style="background: rgba({rgb},{alpha:.2f})" '
                     f'title="{html.escape(tip)}">{html.escape(word)}</span>')
    return f'<div class="toks">{" ".join(spans)}</div>'


def prob_chart(probs: np.ndarray, colour: str) -> alt.Chart:
    df = pd.DataFrame({"Stars": [f"{k}★" for k in range(1, 6)], "Probability": probs,
                       "predicted": [k == int(np.argmax(probs)) for k in range(5)]})
    base = alt.Chart(df).encode(x=alt.X("Stars:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)))
    bars = base.mark_bar(color=colour, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1.12]), axis=None),
        opacity=alt.condition("datum.predicted", alt.value(1.0), alt.value(0.35)),
        tooltip=["Stars", alt.Tooltip("Probability:Q", format=".1%")])
    labels = base.mark_text(dy=-8, fontSize=12, fontWeight="bold", color=colour).encode(y="Probability:Q", text=alt.Text("Probability:Q", format=".0%"))
    return (bars + labels).properties(height=165)


def model_card(name: str, subtitle: str, pred: inference.Prediction, true_stars: int | None) -> None:
    with st.container(border=True):
        st.markdown(f"**{name}**  \n<span class='legend'>{subtitle}</span>", unsafe_allow_html=True)
        left, right = st.columns([3, 2], vertical_alignment="center")
        left.markdown(stars_html(pred.stars), unsafe_allow_html=True)
        verdict = ""
        if true_stars is not None:
            verdict = " · ✅ correct" if pred.stars == true_stars else f" · ❌ off by {abs(pred.stars - true_stars)}"
        right.markdown(f"**{pred.stars}★**{verdict}  \n{pred.confidence:.0%} confident  \n"
                       f"expected {pred.expected:.2f}★", unsafe_allow_html=True)
        st.altair_chart(prob_chart(pred.probs, COLOR[name]), width="stretch")
        note = f"{pred.seconds * 1000:.0f} ms"
        if pred.truncated:
            note += f" · truncated to {pred.max_len} of {pred.n_tokens} tokens"
        st.caption(note)


def top_words(words, scores, k=3, sign=1) -> str:
    order = [i for i in np.argsort(-sign * np.asarray(scores)) if sign * scores[i] > 0][:k]
    return ", ".join(f"“{words[i]}” ({scores[i]:+.2f})" for i in order) or "none"


# ======================================================================= state + callbacks
EXAMPLE_KEYS = (["custom"] + [f"ex:{x['id']}" for x in cached_examples()] + [f"probe:{i}" for i in range(len(results.PROBES))])


def example_label(key: str) -> str:
    if key == "custom":
        return "✍️  Your own review"
    if key.startswith("ex:"):
        x = next(x for x in cached_examples() if x["id"] == key[3:])
        return f"Test example {x['letter']} · {x['name']} · true {x['stars']}★"
    name, _, _ = results.PROBES[int(key.split(":")[1])]
    return f"Probe · {name}"


def load_example() -> None:
    key = st.session_state.example
    if key == "custom":
        st.session_state.loaded = None
        return
    if key.startswith("ex:"):
        x = next(x for x in cached_examples() if x["id"] == key[3:])
        title, body, meta = "", x["text"], {"stars": x["stars"], "why": x["why"], "letter": x["letter"],
                                            "saved": {BIGRU: x["bigru_stars"], ROBERTA: x["roberta_stars"]}}
    else:
        name, title, body = results.PROBES[int(key.split(":")[1])]
        meta = {"stars": None, "why": None, "probe": name}
    st.session_state.title, st.session_state.body = title, body
    st.session_state.loaded = {"text": inference.Demo.text(title, body), **meta}


def send_to_live(review_id: str, text: str, stars: int) -> None:
    st.session_state.example = "custom"
    st.session_state.title, st.session_state.body = "", text
    st.session_state.loaded = {"text": text, "stars": stars, "why": None, "browse": review_id}
    st.session_state.tab = LIVE


for key, value in {"example": "custom", "title": "", "body": "", "loaded": None}.items():
    st.session_state.setdefault(key, value)

# ======================================================================= sidebar
runs = inference.available_runs()
with st.sidebar:
    st.header("Settings")
    seeds = [s for s in inference.SEEDS if inference.run_name("bigru", s) in runs
             and inference.run_name("mean", s) in runs] or list(inference.SEEDS)
    seed = st.radio("Training seed", seeds, horizontal=True,
                    help="Every model was trained twice, with seeds 42 and 43. Both are loaded from their best "
                         "validation checkpoint.")
    show_ig = st.toggle("Explain RoBERTa (integrated gradients)", value=True)
    ig_steps = st.select_slider("Integrated-gradient steps", options=[32, 64, 128, 256], value=256,
                                help="256 is what the report uses and keeps the completeness gap small (about 6 s per "
                                     "review on a Mac, cached afterwards). Fewer steps are faster but less exact.")
    show_ablation = st.toggle("Show pooling ablation", value=True,
                              help="The same RoBERTa recipe trained with first-token (<s>) pooling instead of mean pooling.")
    st.divider()
    st.caption(f"Checkpoints found: {len(runs)} of 6")
    st.caption("Runs fully offline: weights from `outputs/checkpoints/`, roberta-base from the local cache.")

# ======================================================================= header
h = cached_headline()
st.title("⭐ How many stars? BiGRU vs fine-tuned RoBERTa")
st.caption("DSL 504 · Group Assignment 2 · Question 2 · 5-class rating of English Amazon reviews "
           "(SetFit/amazon_reviews_multi_en: 200k train, 5k validation, 5k test)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("RoBERTa test macro-F1", f"{100 * h['macro_f1'][0]:.1f}%", f"{100 * (h['macro_f1'][0] - h['macro_f1'][1]):+.1f} pts vs BiGRU", border=True)
m2.metric("RoBERTa test accuracy", f"{100 * h['accuracy'][0]:.1f}%", f"{100 * (h['accuracy'][0] - h['accuracy'][1]):+.1f} pts vs BiGRU", border=True)
m3.metric("Mean absolute error", f"{h['mae'][0]:.3f}★", f"{round(h['mae'][0], 3) - round(h['mae'][1], 3):+.3f}★ vs BiGRU", delta_color="inverse", border=True)
m4.metric("1★ ↔ 5★ swaps (of 5,000)", f"{h['extreme_confusions'][0]:.1f}", f"{h['extreme_confusions'][0] - h['extreme_confusions'][1]:+.1f} vs BiGRU",
          delta_color="inverse", border=True, help="Mean of the two seeds.")

tab_live, tab_results, tab_browse, tab_how = st.tabs([LIVE, RESULTS, BROWSE, HOW], key="tab", on_change="rerun")

# ======================================================================= live demo
def render_live() -> None:
    needed = [inference.run_name(m, seed) for m in ("bigru", "mean")]
    if not all(r in runs for r in needed):
        st.error("The trained checkpoints are missing. From `Question-2/`, run "
                 "`../.venv/bin/python demo/find_checkpoint.py <folder with best.pt files> --install`.")
        return

    with st.spinner("Loading the trained models (about 15 s on first start)…"):
        demo = get_demo()

    st.selectbox("Pick an example, or write your own review below", EXAMPLE_KEYS, key="example",
                 format_func=example_label, on_change=load_example)
    st.text_input("Title (optional)", key="title", placeholder="e.g. Not what I expected")
    st.text_area("Review", key="body", height=140, placeholder="Type any product review…")

    text = inference.Demo.text(st.session_state.title, st.session_state.body)
    loaded = st.session_state.loaded if st.session_state.loaded and st.session_state.loaded["text"] == text else None
    true_stars = loaded["stars"] if loaded else None
    if loaded and loaded.get("why"):
        st.info(f"**Why it's hard (report example {loaded['letter']}):** {loaded['why']}  \n"
                f"True rating **{loaded['stars']}★**. In the report, the BiGRU said "
                f"{', '.join(f'{s}★' for s in loaded['saved'][BIGRU])} and RoBERTa "
                f"{', '.join(f'{s}★' for s in loaded['saved'][ROBERTA])} (seeds 42, 43).")
    elif loaded and loaded.get("browse"):
        st.info(f"Test review `{loaded['browse']}` · true rating **{loaded['stars']}★**")

    if not text.strip():
        st.markdown("👆 Pick an example or type a review. Both models predict as you type "
                    "(press **Ctrl/⌘ + Enter** in the text box).")
    else:
        bigru_pred = predict("bigru", text, seed)
        roberta_pred = predict("mean", text, seed)

        if bigru_pred.stars == roberta_pred.stars:
            st.success(f"Both models say **{roberta_pred.stars}★**.")
        else:
            st.warning(f"The models disagree: BiGRU **{bigru_pred.stars}★** · RoBERTa **{roberta_pred.stars}★**.")

        col_b, col_r = st.columns(2, gap="medium")
        with col_b:
            model_card(BIGRU, "GloVe 300-d → 2-layer BiGRU → additive attention · 12.0 M parameters",
                       bigru_pred, true_stars)
            st.markdown("**What the BiGRU attends to** (attention weights α, darker = more weight)")
            st.markdown(tokens_html(bigru_pred.tokens, bigru_pred.attention, "attention", bigru_pred.unknown),
                        unsafe_allow_html=True)
            top = np.argsort(-bigru_pred.attention)[:3]
            n_unk = sum(bigru_pred.unknown)
            st.caption("Top: " + ", ".join(f"“{bigru_pred.tokens[i]}” {bigru_pred.attention[i]:.0%}" for i in top)
                       + (f" · {n_unk} word(s) unknown to its vocabulary (wavy underline)" if n_unk else "")
                       + ". Attention shows what was pooled, not what changed the decision.")

        with col_r:
            model_card(ROBERTA, "roberta-base, fully fine-tuned → masked mean pooling · 124.1 M parameters",
                       roberta_pred, true_stars)
            if show_ig:
                digest = hashlib.md5(f"{text}|{seed}".encode()).hexdigest()[:10]
                target = st.radio("Explain RoBERTa's score for", [1, 2, 3, 4, 5], index=roberta_pred.stars - 1,
                                  format_func=lambda s: f"{s}★", horizontal=True, key=f"target_{digest}",
                                  help="Integrated gradients for that star's logit. The predicted rating is selected "
                                       "first; pick another to ask 'why not 1★?'.")
                with st.spinner(f"Integrated gradients, {ig_steps} steps…"):
                    exp = explain(text, "mean", seed, target, ig_steps)
                st.markdown(tokens_html(exp.words, exp.scores, "ig"), unsafe_allow_html=True)
                st.markdown(f"<span class='legend'><span class='tok' style='background: rgba({FOR_RGB},.55)'>green</span> "
                            f"pushes towards {target}★ · <span class='tok' style='background: rgba({AGAINST_RGB},.55)'>"
                            f"red</span> pushes away</span>", unsafe_allow_html=True)
                st.caption(f"For {target}★: {top_words(exp.words, exp.scores)}. Against: "
                           f"{top_words(exp.words, exp.scores, sign=-1)}. Scores add up to the logit change "
                           f"{exp.logit_change:+.2f} (gap {exp.gap:+.3f}, {exp.steps} steps, {exp.seconds:.1f} s).")

        if show_ablation and inference.run_name("first", seed) in runs:
            first_pred = predict("first", text, seed)
            with st.container(border=True):
                st.markdown("**Pooling ablation**: same RoBERTa recipe, only the pooling differs. "
                            "On test the two are within seed noise (66.1% vs 66.4% macro-F1).")
                a, b = st.columns(2)
                a.markdown(f"Mean pooling (main model): **{roberta_pred.stars}★** ({roberta_pred.confidence:.0%})")
                b.markdown(f"First-token `<s>` pooling: **{first_pred.stars}★** ({first_pred.confidence:.0%})")

        with st.expander("What each model actually reads"):
            st.markdown(f"**BiGRU**: cleaned, lowercased spaCy tokens; {len(bigru_pred.tokens)} of "
                        f"{bigru_pred.n_tokens} kept (max {bigru_pred.max_len}).")
            st.markdown(tokens_html(bigru_pred.tokens, np.zeros(len(bigru_pred.tokens)), "attention", bigru_pred.unknown),
                        unsafe_allow_html=True)
            ids, n = demo.roberta_ids(text)
            pieces = demo.tokenizer.convert_ids_to_tokens(ids[0].tolist())
            st.markdown(f"**RoBERTa**: cased byte-level BPE, {n} pieces (max {roberta_pred.max_len}); "
                        "`Ġ` marks a piece that starts a new word, so no word is ever unknown.")
            st.markdown("".join(f"<span class='chip'>{html.escape(p)}</span>" for p in pieces), unsafe_allow_html=True)

with tab_live:
    render_live()

# ======================================================================= test-set results
with tab_results:
    st.subheader("Test set: 5,000 reviews, 1,000 per star, mean ± std over seeds 42 and 43")
    st.dataframe(cached_metrics(), hide_index=True, width="stretch")
    st.markdown("- RoBERTa is ahead on **every class**, most on the ambiguous **2★ and 3★**.\n"
                "- The gain is mostly about **severity**: far fewer errors of 2+ stars and about half the 1★↔5★ swaps, "
                "while 4★ vs 5★ confusions stay at the baseline's level.\n"
                "- First-token pooling matches mean pooling within seed noise.")
    scale = alt.Scale(domain=list(MODELS), range=list(COLOR.values()))

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Per-class F1**")
        f1 = results.per_class_f1()
        st.altair_chart(alt.Chart(f1).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Stars:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
            xOffset=alt.XOffset("Model:N", sort=list(MODELS)),
            y=alt.Y("F1:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", title=None)),
            color=alt.Color("Model:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "Stars", alt.Tooltip("F1:Q", format=".1%")]).properties(height=280), width="stretch")
    with c2:
        st.markdown("**How far off are the errors?** (share of all test reviews)")
        dist = results.error_distance()
        st.altair_chart(alt.Chart(dist).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Off by:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
            xOffset=alt.XOffset("Model:N", sort=list(MODELS)),
            y=alt.Y("Share:Q", axis=alt.Axis(format=".0%", title=None)),
            color=alt.Color("Model:N", scale=scale, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "Off by", alt.Tooltip("Share:Q", format=".2%")]).properties(height=280), width="stretch")

    st.markdown("**Confusion matrices** (rows = true rating, normalised per row, both seeds pooled)")
    cols = st.columns(2, gap="medium")
    for col, model in zip(cols, (BIGRU, ROBERTA)):
        cm = results.confusion(model)
        base = alt.Chart(cm).encode(x=alt.X("Predicted:N", sort=None, axis=alt.Axis(labelAngle=0, orient="top")),
                                    y=alt.Y("True:N", sort=None))
        heat = base.mark_rect().encode(color=alt.Color("Share:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]),
                                                       legend=None),
                                       tooltip=["True", "Predicted", alt.Tooltip("Share:Q", format=".1%"), "Count"])
        txt = base.mark_text(fontSize=13).encode(text=alt.Text("Share:Q", format=".0%"),
                                                 color=alt.condition("datum.Share > 0.5", alt.value("white"),
                                                                     alt.value("#1a1a1a")))
        col.markdown(f"*{model}*")
        col.altair_chart((heat + txt).properties(height=300), width="stretch")

    st.markdown("**Training curves** (macro-F1; train dashed, validation solid; one line per seed)")
    curves = results.curves()
    y_scale = alt.Scale(domain=[float(curves["Macro-F1"].min()) - 0.01, float(curves["Macro-F1"].max()) + 0.01])
    for col, models in zip(st.columns(2, gap="medium"), ([BIGRU], [ROBERTA, FIRST])):
        col.altair_chart(alt.Chart(curves[curves["Model"].isin(models)]).mark_line(point=True).encode(
            x=alt.X("Epoch:Q", axis=alt.Axis(title="epoch", tickMinStep=0.5)),
            y=alt.Y("Macro-F1:Q", scale=y_scale, axis=alt.Axis(format=".0%", title=None)),
            color=alt.Color("Model:N", scale=alt.Scale(domain=models, range=[COLOR[m] for m in models]),
                            legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.StrokeDash("Split:N", scale=alt.Scale(domain=["val", "train"], range=[[1, 0], [5, 4]]),
                                      legend=alt.Legend(orient="bottom", title=None)),
            detail="Run:N",
            tooltip=["Model", "Seed", "Split", "Epoch", alt.Tooltip("Macro-F1:Q", format=".2%")]).properties(height=300),
            width="stretch")
    st.caption("The BiGRU embedding is frozen in epoch 1 and fine-tuned afterwards, which is when it starts to overfit; "
               "early stopping keeps its best validation epoch. RoBERTa is evaluated twice per epoch for 2 epochs.")

# ======================================================================= browse saved test predictions
with tab_browse:
    st.caption("These are the predictions saved once by the test-set evaluation (`test_predictions.jsonl`). "
               "Nothing is re-predicted here, so the test set is still used for one measurement only.")
    table = test_table()
    b_run, r_run = inference.run_name("bigru", seed), inference.run_name("mean", seed)
    b_ok, r_ok = table[b_run + "_pred"] == table["label"], table[r_run + "_pred"] == table["label"]
    cases = {
        "All reviews": np.ones(len(table), bool),
        "RoBERTa right, BiGRU wrong": r_ok & ~b_ok,
        "BiGRU right, RoBERTa wrong": b_ok & ~r_ok,
        "Both wrong": ~b_ok & ~r_ok,
        "RoBERTa off by 2+ stars": (table[r_run + "_pred"] - table["label"]).abs() >= 2,
    }
    f1c, f2c = st.columns([3, 2])
    case = f1c.selectbox(f"Show (seed {seed})", list(cases))
    stars_sel = f2c.pills("True rating", [1, 2, 3, 4, 5], selection_mode="multi", default=[1, 2, 3, 4, 5],
                          format_func=lambda s: f"{s}★")
    mask = cases[case] & table["label"].isin([s - 1 for s in stars_sel])
    pool = table[mask]
    st.markdown(f"**{len(pool):,}** of {len(table):,} test reviews match.")

    if "browse_id" not in st.session_state or st.session_state.browse_id not in set(pool["id"]):
        st.session_state.browse_id = pool["id"].sample(1).iloc[0] if len(pool) else None
    if st.button("🎲 Another random review", disabled=not len(pool)):
        st.session_state.browse_id = pool["id"].sample(1).iloc[0]

    if st.session_state.browse_id is not None:
        row = table[table["id"] == st.session_state.browse_id].iloc[0]
        true = int(row["label"]) + 1
        with st.container(border=True):
            st.markdown(f"`{row['id']}` · true rating **{true}★**")
            st.markdown(stars_html(true), unsafe_allow_html=True)
            st.write(row["text"])
        rows = []
        for model, (model_runs, _) in MODELS.items():
            for run in model_runs:
                pred = int(row[run + "_pred"]) + 1
                rows.append({"Model": model, "Seed": int(run[-2:]), "Prediction": f"{pred}★",
                             "Correct": "✅" if pred == true else f"❌ off by {abs(pred - true)}",
                             f"P({true}★)": row[run + "_probs"][true - 1], "Probabilities 1★→5★": row[run + "_probs"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            f"P({true}★)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Probabilities 1★→5★": st.column_config.BarChartColumn(y_min=0, y_max=1)})
        st.button("Try this review live →", type="primary", on_click=send_to_live,
                  args=(row["id"], row["text"], true))

# ======================================================================= how it works
with tab_how:
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#9a9a9a", fillcolor="#f4f4f2"];
  edge [color="#8a8a8a", arrowsize=0.7];
  review [label="Review text\\n\\"<title>. <body>\\""];
  out [label="P(1★ … 5★)"];
  subgraph cluster_b { label="BiGRU + attention · 12.0 M params"; fontname="Helvetica"; fontcolor="#2a78d6"; color="#2a78d6";
    b1 [label="strip HTML/URLs,\\nlowercase, spaCy tokens", fillcolor="#e3eefb"];
    b2 [label="GloVe 6B 300-d\\nvocab 32,878, max 200", fillcolor="#e3eefb"];
    b3 [label="2-layer BiGRU\\n256 per direction", fillcolor="#e3eefb"];
    b4 [label="additive attention\\npooling α", fillcolor="#e3eefb"];
    b5 [label="dropout → linear\\n512 → 5", fillcolor="#e3eefb"];
    b1 -> b2 -> b3 -> b4 -> b5; }
  subgraph cluster_r { label="RoBERTa · 124.1 M params, all fine-tuned"; fontname="Helvetica"; fontcolor="#eb6834"; color="#eb6834";
    r1 [label="byte-level BPE (cased)\\n<s> … </s>, max 256", fillcolor="#fde8de"];
    r2 [label="roberta-base\\n12 layers, 768-d", fillcolor="#fde8de"];
    r3 [label="masked mean pooling\\n(ablation: <s> only)", fillcolor="#fde8de"];
    r4 [label="dropout → linear\\n768 → 5", fillcolor="#fde8de"];
    r1 -> r2 -> r3 -> r4; }
  review -> b1; review -> r1; b5 -> out; r4 -> out;
}""", width="stretch")
    a, b = st.columns(2, gap="medium")
    a.markdown("""
**BiGRU + additive attention (baseline)**
- Embedding initialised from GloVe, frozen in epoch 1, fine-tuned after.
- Packed sequences, so neither direction reads padding.
- Attention: e_t = vᵀ tanh(W h_t + b), α = masked softmax, c = Σ α_t h_t.
- Adam, lr 1e-3, batch 128, fp16, up to 15 epochs, early stopping on validation macro-F1.
""")
    b.markdown("""
**RoBERTa (main model)**
- `roberta-base`, every layer fine-tuned.
- Masked mean of the last hidden layer over all real tokens (sum in fp32).
- AdamW, lr 2e-5, weight decay 0.01, linear warm-up 6%, batch 32, fp16, 2 epochs, 2× T4.
- Checkpoint with the best validation macro-F1 (evaluated twice per epoch).
""")
    st.markdown("""
**Protocol.** Splits and labels come from `SetFit/amazon_reviews_multi_en`; titles are re-attached from
`mteb/amazon_reviews_multi` by review id (100% matched). Validation is used for early stopping, checkpoint choice and
the ablation; the test split was predicted **once** per run and every test number comes from those saved predictions.
Each model was trained with seeds 42 and 43.

**Explanations in the live demo.** For the BiGRU we show its attention weights α. For RoBERTa we use
*integrated gradients* (Sundararajan et al., 2017) on the input embeddings, from a baseline where every token except
`<s>` and `</s>` is `<pad>`; the word scores add up to the change in the chosen star's logit.
""")
