"""Every setting of the Q4 experiments in one place. The scripts read these; run_info.json records them."""

from common import checkpoints_dir, data_dir, figures_dir, results_dir

QUESTION = 4

DATA = dict(
    n_train=20_000, n_val=1_000, n_test=1_000,  # the brief's suggested subset
    max_tokens=400,                              # longer stories are dropped, never truncated
    pool_factor=3,                               # train sample is drawn from the first 3 * n_train kept stories
    subset_seed=42,
    min_freq=5,                                  # words seen fewer times in train become <unk>
)

# Both models share all of these. The only thing that differs is the recurrent cell.
MODEL = dict(emb_dim=256, hid_dim=512, num_layers=1, dropout=0.3)

TRAIN = dict(
    batch_size=64, rows_per_pass=16,  # 64 stories per optimiser step, processed 16 at a time to fit in 4 GB
    lr=2e-3, grad_clip=1.0, max_epochs=25,
    patience=3,                 # early stopping on validation loss
    lr_min_factor=0.05,         # cosine decay from lr to 5% of lr over max_epochs, the same for both models
    seeds=(42, 43),             # 42 is the headline run (shipped checkpoints); 43 measures run-to-run spread
)

# Generation. One cap everywhere so lengths are comparable across experiments.
GENERATION = dict(
    max_new_tokens=250,
    short_below=30,
    n_samples=3,          # continuations per (prompt, model, sampling config); greedy is deterministic so 1
    seed=42,
    test_prefix_tokens=12,  # quantitative eval: each held-out test story is cut after its first 12 tokens
)

CELLS = ("rnn", "lstm")
MODEL_NAMES = {"rnn": "Vanilla RNN", "lstm": "LSTM"}


def paths():
    return dict(
        data=data_dir(QUESTION), results=results_dir(QUESTION), figures=figures_dir(QUESTION),
        checkpoints=checkpoints_dir(QUESTION),
    )
