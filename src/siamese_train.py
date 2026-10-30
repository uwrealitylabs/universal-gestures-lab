"""
Pre-train the Siamese (triplet-loss) embedding on `src/siamese/siamese training data` and test
few-shot learning on gestures the model never saw during training.

Each fold hides 3 gestures (test) and holds out 2 more (validation). The model trains on the
remaining gestures plus the negatives. A gesture is taught from K clips: its prototype is the
mean embedding of those clips, as in Unity's PrototypeGestureRecog (Prototypical Networks,
Snell et al. 2017). The "no gesture" cutoff and the best epoch are chosen on the validation
gestures, never on the test gestures (class-disjoint validation, Ravi & Larochelle 2017).
All splits are by whole clip, never by frame.

Data layout, one folder per person who recorded:
    <person>/positive/<gesture>/<left|right>/<timestamp>.json
    <person>/negative/<everyday|resting|transitions>/<left|right>/<timestamp>.json
Gestures are taught and scored per person and hand: the K teaching clips and the scored clips
come from the same person, as when someone teaches Unity their own gesture. A person with no
negative recordings is scored against everyone else's test negatives.

Run from the repo root:
    python -m src.siamese_train                 # few-shot evaluation, seed 0
    python -m src.siamese_train --seeds 0 1 2   # repeat with 3 seeds to measure run-to-run noise
    python -m src.siamese_train --augment       # + noise augmentation on gesture frames
    python -m src.siamese_train --near-miss     # + synthetic near-miss negatives
    python -m src.siamese_train --final         # also train on all gestures and export ONNX
    python -m src.siamese_train --test-person edy   # never train on edy: scores on new hands
"""

import argparse
import copy
import json
import os
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners
from sklearn.metrics import f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIAMESE_DIR = os.path.join(SCRIPT_DIR, "siamese")
DATA_DIR = os.path.join(SIAMESE_DIR, "siamese training data")
OUTPUT_DIR = os.path.join(SIAMESE_DIR, "trained_model")

# Feature order written by Unity's UGDataExtractorScript (finger-only, no wrist pose)
FEATURE_NAMES = [
    "thumbCurl", "thumbAbd", "idxCurl", "idxAbd", "idxFlex", "idxOpp",
    "midCurl", "midAbd", "midFlex", "midOpp", "ringCurl", "ringAbd", "ringFlex", "ringOpp",
    "pinkyCurl", "pinkyFlex", "pinkyOpp",
]
FINGER_GROUPS = {
    "thumb": [0, 1], "index": [2, 3, 4, 5], "middle": [6, 7, 8, 9],
    "ring": [10, 11, 12, 13], "pinky": [14, 15, 16],
}
HANDS = ("left", "right")
NONE = -1  # "no gesture" label / prediction

# Model and training
input_dim = len(FEATURE_NAMES)
hidden_dim = 128
embedding_dim = 16
margin = 0.2                # FaceNet (Schroff et al. 2015) margin for L2-normalised embeddings
learning_rate = 1e-3
num_epochs = 30
batches_per_epoch = 200
clips_per_gesture = 8       # each batch takes one frame from 8 different clips of every gesture
negatives_per_batch = 40
zeros_per_batch = 4         # all-zero "hand not tracked" frames, absent from the recordings
near_misses_per_batch = 40
augment_scale = 0.5         # noise std as a fraction of each feature's between-clip spread

# Splits and evaluation
hidden_per_fold = 3
val_gestures_per_fold = 2
neg_val_fraction = 0.2
neg_test_fraction = 0.2
shots = (1, 3)
val_shots = 1
val_episodes = 5
episodes = 10
on_gesture_percentile = 90  # same rule as negative-cleaning step 4

# Ways to pick the "no gesture" cutoff on the validation episodes (all are scored):
#   best_f1                the cutoff with the best validation macro-F1
#   f1_plateau_middle      middle of the continuous range of cutoffs within
#                          plateau_tolerance of the best validation F1
#   false_trigger_1pct/5pct  largest cutoff that fires on at most 1% / 5% of validation
#                          negative frames (threshold at a fixed false-accept rate, as in
#                          face verification). The rate holds for the validation gestures
#                          only; new gestures can trigger more often.
CUTOFF_RULES = ("best_f1", "f1_plateau_middle", "false_trigger_1pct", "false_trigger_5pct")
plateau_tolerance = 0.005
epoch_min_delta = 0.001  # move to a later epoch only if validation F1 improves by more than this

# Random streams, so changing one option does not change the others' random draws.
# They start at 1: numpy treats [seed, 0, 0] the same as seed.
RNG_BATCHES, RNG_NEAR_MISS, RNG_VALIDATION, RNG_TEST, RNG_NOISE, RNG_NEAR_MISS_SAMPLING = range(1, 7)


@dataclass
class Clip:
    path: str           # relative to DATA_DIR; used as the clip id. File names are timestamps.
    person: str         # who recorded it (top-level folder)
    kind: str           # "positive" or "negative"
    category: str       # gesture name, or negative category
    hand: str
    frames: np.ndarray  # (num_frames, 17), raw features


# ---------------------------------------------------------------- data

def subfolders(path):
    return sorted(name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))


def load_clips(data_dir=DATA_DIR):
    clips = []
    for person in subfolders(data_dir):
        for kind, label in (("positive", 1), ("negative", 0)):
            kind_dir = os.path.join(data_dir, person, kind)
            if not os.path.isdir(kind_dir):
                continue  # e.g. a person who recorded no negatives
            for category in subfolders(kind_dir):
                for hand in HANDS:
                    hand_dir = os.path.join(kind_dir, category, hand)
                    if not os.path.isdir(hand_dir):
                        continue
                    for name in sorted(os.listdir(hand_dir)):
                        if not name.endswith(".json"):
                            continue
                        path = os.path.join(person, kind, category, hand, name)
                        with open(os.path.join(data_dir, path)) as f:
                            frames = json.load(f)
                        X = np.array([frame["handData"] for frame in frames], dtype=np.float32)
                        assert X.ndim == 2 and X.shape[1] == input_dim and len(X) > 0, path
                        assert all(frame["confidence"] == label for frame in frames), f"label mismatch in {path}"
                        clips.append(Clip(path, person, kind, category, hand, X))
    return clips


def people_in(part_clips):
    return sorted({c.person for c in part_clips})


def make_folds(gestures, seed):
    """Balanced folds: with the gestures in a random order, fold i hides gestures i..i+2 and
    holds out i+3..i+4 for validation (wrapping around). Every gesture is hidden exactly
    3 times and used for validation exactly 2 times."""
    order = [str(g) for g in np.random.default_rng(seed).permutation(gestures)]
    per_fold = hidden_per_fold + val_gestures_per_fold
    folds = []
    for i in range(len(order)):
        picked = [order[(i + j) % len(order)] for j in range(per_fold)]
        folds.append((picked[:hidden_per_fold], picked[hidden_per_fold:]))
    return folds


def split_clips(clips, hidden, val_gestures, hold_out_negatives=True, test_person=None):
    """Assign whole clips to train / val / test. Hidden gestures go to test, validation
    gestures to val, every other gesture to train. Negatives are split by time within each
    person, category and hand: oldest 60% train, next 20% val, newest 20% test. Each negative
    folder is one continuous recording, so a random split would put test clips between
    near-identical training clips. None of the test person's clips are trained on or used for
    validation: their hidden-gesture clips and all their negatives go to test, the rest is unused."""
    split = {part: [] for part in ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")}
    groups = {}
    for c in clips:
        groups.setdefault((c.person, c.kind, c.category, c.hand), []).append(c)
    for (person, kind, category, hand), group in sorted(groups.items()):
        group = sorted(group, key=lambda c: c.path)
        if kind == "positive":
            if category in hidden:
                split["test_pos"] += group
            elif person != test_person:
                split["val_pos" if category in val_gestures else "train_pos"] += group
        elif person == test_person:
            split["test_neg"] += group
        else:
            n_test = round(neg_test_fraction * len(group)) if hold_out_negatives else 0
            n_val = round(neg_val_fraction * len(group)) if hold_out_negatives else 0
            n_train = len(group) - n_val - n_test
            split["train_neg"] += group[:n_train]
            split["val_neg"] += group[n_train:n_train + n_val]
            split["test_neg"] += group[n_train + n_val:]
    return split


def check_held_out(split, person):
    """Stop if any clip of the held-out person is used for training or validation."""
    for part in ("train_pos", "val_pos", "train_neg", "val_neg"):
        assert not any(c.person == person for c in split[part]), f"{person} has clips in {part}"


def check_split(split, hidden, val_gestures, test_person=None):
    """Stop if any clip is in two parts, if a hidden or validation gesture leaked into another
    part, if the test person leaked into training or validation, or if the negatives are not
    split by time."""
    part_of = {}
    for part, part_clips in split.items():
        for c in part_clips:
            assert c.path not in part_of, f"{c.path} is in both {part_of[c.path]} and {part}"
            part_of[c.path] = part
    assert not set(hidden) & set(val_gestures), "a gesture is both hidden and validation"
    assert all(c.category in hidden for c in split["test_pos"]), "non-hidden gesture in test_pos"
    assert all(c.category in val_gestures for c in split["val_pos"]), "non-validation gesture in val_pos"
    assert not any(c.category in hidden or c.category in val_gestures for c in split["train_pos"]), \
        "hidden or validation gesture in train_pos"
    # Gestures with too few clips for the larger K are left out of those episodes (evaluate_hidden)
    for gestures, part, k in ((hidden, "test_pos", min(shots)), (val_gestures, "val_pos", val_shots)):
        for person in people_in(split[part]):
            for g in gestures:
                for hand in HANDS:
                    n = sum(c.person == person and c.category == g and c.hand == hand for c in split[part])
                    assert n > k, f"{person} {g} {hand}: {n} clips is too few for {k}-shot"
    if test_person is not None:
        check_held_out(split, test_person)
    rank = {"train_neg": 0, "val_neg": 1, "test_neg": 2}
    by_group = {}
    for part in rank:
        for c in split[part]:
            by_group.setdefault((c.person, c.category, c.hand), []).append((c.path, rank[part]))
    for items in by_group.values():
        ranks = [r for _, r in sorted(items)]
        assert ranks == sorted(ranks), "negatives are not split by time"


def fit_normalization(split):
    """Per-feature mean and std of the training frames only."""
    X = np.concatenate([c.frames for c in split["train_pos"] + split["train_neg"]])
    return X.mean(0), X.std(0) + 1e-6


# ---------------------------------------------------------------- model

class EmbeddingNet(nn.Module):
    """Raw features -> L2-normalised embedding. The z-scoring is part of the model so Unity
    can feed the raw UGDataExtractorScript features."""
    def __init__(self, mean, std):
        super(EmbeddingNet, self).__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        e = self.net((x - self.mean) / self.std)
        # L2 normalisation (FaceNet), written with basic ops to keep the ONNX graph simple
        return e / torch.sqrt((e * e).sum(dim=1, keepdim=True) + 1e-12)


class RawFeatures(nn.Module):
    """Baseline with no training: the 'embedding' is just the z-scored features."""
    def __init__(self, mean, std):
        super(RawFeatures, self).__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) / self.std


@torch.no_grad()
def embed(model, X, batch_size=8192):
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + batch_size])).numpy()
                           for i in range(0, len(X), batch_size)])


# ---------------------------------------------------------------- episodes, cutoff and F1

def nearest_prototype(queries, prototypes):
    """Index of the nearest prototype for each query, and the Euclidean distance to it."""
    d = np.sqrt(((queries[:, None, :] - prototypes[None, :, :]) ** 2).sum(-1))
    nearest = d.argmin(1)
    return nearest, d[np.arange(len(d)), nearest]


def predict(nearest, dist, cutoff):
    """Nearest gesture if it is within the cutoff, otherwise no gesture."""
    return np.where(dist <= cutoff, nearest, NONE)


def macro_f1_curve(true, nearest, dist, labels, cutoffs):
    """Macro-F1 over the gesture `labels` at every cutoff in `cutoffs` (sorted), computed
    with cumulative counts so all cutoffs are scored in one pass. NONE is never averaged:
    negatives only count as false positives when a gesture fires on them."""
    per_label = []
    for g in labels:
        fires = nearest == g
        tp = np.searchsorted(np.sort(dist[fires & (true == g)]), cutoffs, side="right")
        fp = np.searchsorted(np.sort(dist[fires & (true != g)]), cutoffs, side="right")
        fn = (true == g).sum() - tp
        denom = 2 * tp + fp + fn
        per_label.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(per_label, axis=0)


def candidate_cutoffs(dist):
    """Midpoints between consecutive distances, plus one below and one above all of them."""
    d = np.unique(dist)
    return np.concatenate([[d[0] / 2], (d[:-1] + d[1:]) / 2, [d[-1] * 2]])


def choose_cutoff(true, nearest, dist, labels):
    """Cutoff with the best macro-F1, and that F1."""
    cutoffs = candidate_cutoffs(dist)
    curve = macro_f1_curve(true, nearest, dist, labels, cutoffs)
    best = int(np.argmax(curve))
    return float(cutoffs[best]), float(curve[best])


def false_trigger_cutoff(negative_dist, rate):
    """Largest cutoff at which at most `rate` of the negative frames fire a gesture. The cutoff
    sits below the (k+1)-th smallest distance, so frames tied with it do not fire either."""
    d = np.sort(negative_dist)
    k = int(np.floor(rate * len(d)))
    below = d[d < d[k]]
    return float(d[k] / 2) if len(below) == 0 else float((below[-1] + d[k]) / 2)


def cutoff_rules(true, nearest, dist, labels):
    """The cutoff each rule in CUTOFF_RULES picks, and the best macro-F1."""
    cutoffs = candidate_cutoffs(dist)
    curve = macro_f1_curve(true, nearest, dist, labels, cutoffs)
    best = int(np.argmax(curve))
    # continuous run of near-best cutoffs around the best one
    near_best = curve >= curve[best] - plateau_tolerance
    lo, hi = best, best
    while lo > 0 and near_best[lo - 1]:
        lo -= 1
    while hi < len(cutoffs) - 1 and near_best[hi + 1]:
        hi += 1
    negative_dist = dist[true == NONE]
    rules = {
        "best_f1": float(cutoffs[best]),
        "f1_plateau_middle": float((cutoffs[lo] + cutoffs[hi]) / 2),
        "false_trigger_1pct": false_trigger_cutoff(negative_dist, 0.01),
        "false_trigger_5pct": false_trigger_cutoff(negative_dist, 0.05),
    }
    assert list(rules) == list(CUTOFF_RULES)
    return rules, float(curve[best])


def clips_by_gesture(split_part, gestures, person, hand):
    return [[c for c in split_part if c.person == person and c.category == g and c.hand == hand]
            for g in gestures]


def episode_negatives(emb, negative_clips, person, hand):
    """The person's own negative clips for this hand. A person who recorded none gets everyone
    else's, so false triggers still count, but they are measured on other people's hands."""
    own = [c for c in negative_clips if c.person == person and c.hand == hand]
    chosen = own or [c for c in negative_clips if c.hand == hand]
    return np.concatenate([emb[c.path] for c in chosen])


def draw_support(rng, by_gesture, k):
    return [set(rng.choice(len(clips), k, replace=False).tolist()) for clips in by_gesture]


def build_episode(emb, by_gesture, negatives, support):
    """Prototype per gesture = mean embedding of all frames of its support clips. Queries =
    every other clip of those gestures (labelled by gesture index) plus the negatives (NONE).
    Validation and test episodes are both built here, so they match."""
    prototypes = np.stack([np.concatenate([emb[clips[i].path] for i in sup]).mean(0)
                           for clips, sup in zip(by_gesture, support)])
    q_emb, q_true = [negatives], [np.full(len(negatives), NONE)]
    for gi, (clips, sup) in enumerate(zip(by_gesture, support)):
        for j, c in enumerate(clips):
            if j not in sup:
                q_emb.append(emb[c.path])
                q_true.append(np.full(len(emb[c.path]), gi))
    return prototypes, np.concatenate(q_emb), np.concatenate(q_true)


def validate(model, split, val_gestures, rng_key):
    """Pick the cutoffs on the validation gestures, with episodes built like the test ones:
    per person and hand, each gesture is taught from 1 random clip, and its other clips plus the
    validation negatives are queries. The same episodes are used at every epoch.
    Returns the cutoff of every rule and the best validation macro-F1."""
    rng = np.random.default_rng(rng_key)
    emb = {c.path: embed(model, c.frames) for c in split["val_pos"] + split["val_neg"]}
    true, nearest, dist = [], [], []
    for person in people_in(split["val_pos"]):
        for hand in HANDS:
            by_gesture = clips_by_gesture(split["val_pos"], val_gestures, person, hand)
            negatives = episode_negatives(emb, split["val_neg"], person, hand)
            for _ in range(val_episodes):
                prototypes, queries, q_true = build_episode(emb, by_gesture, negatives,
                                                            draw_support(rng, by_gesture, val_shots))
                n, d = nearest_prototype(queries, prototypes)
                true.append(q_true)
                nearest.append(n)
                dist.append(d)
    return cutoff_rules(np.concatenate(true), np.concatenate(nearest), np.concatenate(dist),
                        range(len(val_gestures)))


def score_episode(true, nearest, dist, cutoff, labels, zero_dist, oracle_f1):
    pred = predict(nearest, dist, cutoff)
    f1 = float(macro_f1_curve(true, nearest, dist, labels, np.array([cutoff]))[0])
    # cross-check the fast F1 against scikit-learn
    assert np.isclose(f1, f1_score(true, pred, labels=labels, average="macro", zero_division=0))
    is_neg = true == NONE
    return {
        "f1": f1,
        "accuracy": float((pred == true).mean()),
        "closed_set_accuracy": float((nearest[~is_neg] == true[~is_neg]).mean()),
        "false_trigger_rate": float((pred[is_neg] != NONE).mean()),
        "oracle_f1": oracle_f1,
        "zero_rejected": float(zero_dist > cutoff),
        "per_gesture_f1": f1_score(true, pred, labels=labels, average=None, zero_division=0).tolist(),
    }


def evaluate_hidden(model, split, hidden, cutoffs, rng_key):
    """Per person and hand, teach each hidden gesture from K random clips of that person, then
    score every other clip of the hidden gestures plus the test negatives, once per cutoff rule.
    Repeated over random choices of K clips; every rule is scored on the same episodes. A gesture
    with K or fewer clips for a person and hand is left out of that person's K-shot episodes,
    since no clip would be left to score."""
    rng = np.random.default_rng(rng_key)
    emb = {c.path: embed(model, c.frames) for c in split["test_pos"] + split["test_neg"]}
    zero_emb = embed(model, np.zeros((1, input_dim), dtype=np.float32))
    results = {rule: {} for rule in cutoffs}
    for person in people_in(split["test_pos"]):
        for hand in HANDS:
            negatives = episode_negatives(emb, split["test_neg"], person, hand)
            for k in shots:
                taught = [g for g, clips in zip(hidden, clips_by_gesture(split["test_pos"], hidden, person, hand))
                          if len(clips) > k]
                assert taught, f"{person} {hand}: no gesture has enough clips for {k}-shot"
                by_gesture = clips_by_gesture(split["test_pos"], taught, person, hand)
                labels = list(range(len(taught)))
                rows = {rule: [] for rule in cutoffs}
                for _ in range(episodes):
                    prototypes, queries, true = build_episode(emb, by_gesture, negatives, draw_support(rng, by_gesture, k))
                    nearest, dist = nearest_prototype(queries, prototypes)
                    zero_dist = nearest_prototype(zero_emb, prototypes)[1][0]
                    oracle_f1 = choose_cutoff(true, nearest, dist, labels)[1]
                    for rule, cutoff in cutoffs.items():
                        rows[rule].append(score_episode(true, nearest, dist, cutoff, labels, zero_dist, oracle_f1))
                for rule, rule_rows in rows.items():
                    summary = {key: float(np.mean([r[key] for r in rule_rows])) for key in rule_rows[0] if key != "per_gesture_f1"}
                    summary["per_gesture_f1"] = dict(zip(taught, np.mean([r["per_gesture_f1"] for r in rule_rows], axis=0).tolist()))
                    results[rule][f"{person}_{hand}_{k}shot"] = summary
    return results


# ---------------------------------------------------------------- training

def between_clip_spread(clips):
    """Per feature: std of the clip means within each gesture and hand, averaged."""
    groups = {}
    for c in clips:
        groups.setdefault((c.category, c.hand), []).append(c.frames.mean(0))
    return np.mean([np.std(means, axis=0) for means in groups.values() if len(means) > 1], axis=0)


def nearest_distance(a, b, chunk=512):
    bt = torch.from_numpy(b)
    return np.concatenate([torch.cdist(torch.from_numpy(a[i:i + chunk]), bt).min(1).values.numpy()
                           for i in range(0, len(a), chunk)])


def on_gesture_distance(pos_clips, mean, std, rng, frames_per_clip=50, stride=5):
    """90th percentile of each training gesture frame's distance (z-scored) to the nearest frame
    of the same gesture in other clips. Anything closer than this to a gesture counts as on it."""
    dists = []
    for clips in pos_clips:
        for i, c in enumerate(clips):
            query = c.frames[rng.choice(len(c.frames), min(frames_per_clip, len(c.frames)), replace=False)]
            others = np.concatenate([o.frames[::stride] for j, o in enumerate(clips) if j != i])
            dists.append(nearest_distance((query - mean) / std, (others - mean) / std))
    return float(np.percentile(np.concatenate(dists), on_gesture_percentile))


def make_near_misses(pos_clips, mean, std, rng, n=20000, stride=5):
    """Synthetic near-miss negatives: a real gesture frame with one finger's features copied from
    a frame of a different gesture. Candidates that land on a training gesture are dropped."""
    frames = [np.concatenate([c.frames for c in clips]) for clips in pos_clips]
    fingers = list(FINGER_GROUPS.values())
    candidates = np.empty((n, input_dim), dtype=np.float32)
    for i in range(n):
        g, h = rng.choice(len(frames), 2, replace=False)
        x = frames[g][rng.integers(len(frames[g]))].copy()
        finger = fingers[rng.integers(len(fingers))]
        x[finger] = frames[h][rng.integers(len(frames[h]))][finger]
        candidates[i] = x
    threshold = on_gesture_distance(pos_clips, mean, std, rng)
    reference = (np.concatenate([f[::stride] for f in frames]) - mean) / std
    keep = nearest_distance((candidates - mean) / std, reference) > threshold
    print(f"  near-miss negatives: kept {keep.sum()} of {n} (on-gesture distance {threshold:.3f})")
    return candidates[keep]


def sample_batch(rng, noise_rng, near_miss_rng, pos_clips, neg_frames, near_misses, spread, augment):
    xs, ys = [], []
    for gi, clips in enumerate(pos_clips):
        for ci in rng.choice(len(clips), clips_per_gesture, replace=False):
            frames = clips[ci].frames
            xs.append(frames[rng.integers(len(frames))])
            ys.append(gi)
    pos = np.stack(xs)
    if augment:
        pos = pos + noise_rng.normal(size=pos.shape).astype(np.float32) * (augment_scale * spread)
    negs = [neg_frames[rng.integers(len(neg_frames), size=negatives_per_batch)],
            np.zeros((zeros_per_batch, input_dim), dtype=np.float32)]
    if near_misses is not None:
        negs.append(near_misses[near_miss_rng.integers(len(near_misses), size=near_misses_per_batch)])
    neg = np.concatenate(negs)
    # Every negative gets its own label, so it can only be the "different" example in a
    # triplet, never an anchor: negatives are pushed away from gestures without forming a class.
    neg_labels = len(pos_clips) + np.arange(len(neg))
    x = np.concatenate([pos, neg]).astype(np.float32)
    y = np.concatenate([np.array(ys), neg_labels]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def train_model(split, train_gestures, val_gestures, args, seed, fold, epochs):
    """Triplet loss with semi-hard negative mining (FaceNet). With validation gestures, keeps
    the epoch with the best validation macro-F1 and returns the cutoffs chosen for it.
    Without them (final model), trains for `epochs` and returns the last epoch."""
    torch.manual_seed(seed * 1000 + fold)
    rng = np.random.default_rng([seed, fold, RNG_BATCHES])
    noise_rng = np.random.default_rng([seed, fold, RNG_NOISE])
    near_miss_rng = np.random.default_rng([seed, fold, RNG_NEAR_MISS_SAMPLING])
    mean, std = fit_normalization(split)
    model = EmbeddingNet(mean, std)
    pos_clips = [[c for c in split["train_pos"] if c.category == g] for g in train_gestures]
    assert all(len(clips) >= clips_per_gesture for clips in pos_clips)
    neg_frames = np.concatenate([c.frames for c in split["train_neg"]])
    spread = between_clip_spread(split["train_pos"])
    near_misses = None
    if args.near_miss:
        near_misses = make_near_misses(pos_clips, mean, std, np.random.default_rng([seed, fold, RNG_NEAR_MISS]))

    miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard")
    loss_fn = losses.TripletMarginLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best = {"f1": -1.0, "cutoffs": None, "epoch": epochs}
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_triplets = 0.0, 0
        for _ in range(batches_per_epoch):
            x, y = sample_batch(rng, noise_rng, near_miss_rng, pos_clips, neg_frames, near_misses, spread, args.augment)
            emb = model(x)
            triplets = miner(emb, y)
            loss = loss_fn(emb, y, triplets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_triplets += len(triplets[0])
        message = f"  epoch {epoch + 1:2d}  loss {epoch_loss / batches_per_epoch:.4f}  " \
                  f"triplets/batch {epoch_triplets / batches_per_epoch:6.1f}"
        if val_gestures:
            cutoffs, val_f1 = validate(model, split, val_gestures, [seed, fold, RNG_VALIDATION])
            # Validation F1 is close to 1 at most epochs, so only a clear gain counts
            # (min_delta, as in standard early stopping)
            if val_f1 > best["f1"] + epoch_min_delta:
                best = {"f1": val_f1, "cutoffs": cutoffs, "epoch": epoch + 1, "state": copy.deepcopy(model.state_dict())}
            message += f"  val F1 {val_f1:.4f}  best-F1 cutoff {cutoffs['best_f1']:.3f}"
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(message)
    if val_gestures:
        model.load_state_dict(best["state"])
        print(f"  best epoch {best['epoch']}  val F1 {best['f1']:.4f}  cutoffs "
              + ", ".join(f"{rule} {c:.3f}" for rule, c in best["cutoffs"].items()))
    return model, best["cutoffs"], best["f1"], best["epoch"]


# ---------------------------------------------------------------- cross-validation

def run_folds(clips, gestures, args, seed):
    results = {"baseline": [], "model": [], "folds": []}
    folds = make_folds(gestures, seed)
    for fold, (hidden, val_gestures) in enumerate(folds):
        print(f"\n=== Seed {seed}, fold {fold + 1}/{len(folds)}: hidden {hidden}, validation {val_gestures} ===")
        split = split_clips(clips, hidden, val_gestures, test_person=args.test_person)
        check_split(split, hidden, val_gestures, args.test_person)
        train_gestures = [g for g in gestures if g not in hidden and g not in val_gestures]
        print("  clips: " + ", ".join(f"{part} {len(part_clips)}" for part, part_clips in split.items()))

        # Baseline and model are scored on the same support clips (same test random stream)
        test_key = [seed, fold, RNG_TEST]
        baseline = RawFeatures(*fit_normalization(split))
        b_cutoffs, b_val_f1 = validate(baseline, split, val_gestures, [seed, fold, RNG_VALIDATION])
        results["baseline"].append(evaluate_hidden(baseline, split, hidden, b_cutoffs, test_key))

        model, cutoffs, val_f1, epoch = train_model(split, train_gestures, val_gestures, args, seed, fold, args.epochs)
        results["model"].append(evaluate_hidden(model, split, hidden, cutoffs, test_key))
        results["folds"].append({"hidden": hidden, "validation": val_gestures, "model_cutoffs": cutoffs,
                                 "model_val_f1": val_f1, "best_epoch": epoch,
                                 "baseline_cutoffs": b_cutoffs, "baseline_val_f1": b_val_f1})
        for rule in CUTOFF_RULES:
            b, m = results["baseline"][-1][rule], results["model"][-1][rule]
            print(f"  {rule:18s} hidden-gesture F1 (1-shot, left/right):" + "".join(
                f"  {p}: baseline {b[f'{p}_left_1shot']['f1']:.3f}/{b[f'{p}_right_1shot']['f1']:.3f}"
                f" model {m[f'{p}_left_1shot']['f1']:.3f}/{m[f'{p}_right_1shot']['f1']:.3f}"
                for p in people_in(split["test_pos"])))
    if args.test_person:
        results["known_baseline"], results["known_model"] = known_gesture_test(clips, gestures, args, seed, results["folds"])
    return results


def known_gesture_test(clips, gestures, args, seed, folds):
    """New hands, gestures the model trained on: train on every gesture of everyone except the
    test person, then teach all gestures from the test person's own K clips and score the rest.
    No validation gestures are left, so the epochs and cutoffs are the medians the folds chose,
    as for the final model."""
    person = args.test_person
    print(f"\n=== Seed {seed}: all {len(gestures)} gestures known, tested on {person} ===")
    split = split_clips([c for c in clips if c.person != person], hidden=(), val_gestures=())
    check_split(split, hidden=(), val_gestures=())
    # Validation negatives are not needed here; the newest 20% stay held out for testing
    split["train_neg"], split["val_neg"] = split["train_neg"] + split["val_neg"], []
    split["test_pos"] = [c for c in clips if c.person == person and c.kind == "positive"]
    split["test_neg"] += [c for c in clips if c.person == person and c.kind == "negative"]
    check_held_out(split, person)
    epochs = int(round(np.median([f["best_epoch"] for f in folds])))
    b_cutoffs = {rule: float(np.median([f["baseline_cutoffs"][rule] for f in folds])) for rule in CUTOFF_RULES}
    m_cutoffs = {rule: float(np.median([f["model_cutoffs"][rule] for f in folds])) for rule in CUTOFF_RULES}
    test_key = [seed, 998, RNG_TEST]
    baseline = evaluate_hidden(RawFeatures(*fit_normalization(split)), split, gestures, b_cutoffs, test_key)
    model, *_ = train_model(split, gestures, (), args, seed, fold=998, epochs=epochs)
    return [baseline], [evaluate_hidden(model, split, gestures, m_cutoffs, test_key)]


# ---------------------------------------------------------------- reporting

METRICS = ("f1", "accuracy", "closed_set_accuracy", "false_trigger_rate", "oracle_f1", "zero_rejected")
LOWER_IS_BETTER = ("false_trigger_rate",)


def summarize(runs, name, rule):
    """Per setting and metric: the mean over folds for each seed, then the mean and std of
    those over seeds (run-to-run noise), and the worst single fold."""
    table = {}
    for setting in runs[0][name][0][rule]:
        table[setting] = {}
        for k in METRICS:
            seed_means = [np.mean([fold[rule][setting][k] for fold in run[name]]) for run in runs]
            values = [fold[rule][setting][k] for run in runs for fold in run[name]]
            worst = max(values) if k in LOWER_IS_BETTER else min(values)
            table[setting][k] = {"mean": float(np.mean(seed_means)), "seed_std": float(np.std(seed_means)),
                                 "worst_fold": float(worst)}
    return table


def per_gesture_table(runs, name, rule):
    """Each gesture's F1, averaged over every fold (and seed) where it was hidden."""
    table = {}
    for run in runs:
        for fold in run[name]:
            for setting, values in fold[rule].items():
                for g, f1 in values["per_gesture_f1"].items():
                    table.setdefault(setting, {}).setdefault(g, []).append(f1)
    return {s: {g: float(np.mean(v)) for g, v in sorted(t.items())} for s, t in table.items()}


def print_table(title, tables, n_seeds):
    print(f"\n{title}  (mean over folds" + ("; ± = std over seeds)" if n_seeds > 1 else ")"))
    header = ["F1", "worst fold F1", "accuracy", "closed-set acc", "false triggers", "oracle F1", "zeros rejected"]
    print(f"  {'cutoff rule':18s} {'setting':18s} " + " ".join(f"{h:>15s}" for h in header))

    def cell(m):
        return f"{m['mean']:.3f} ± {m['seed_std']:.3f}" if n_seeds > 1 else f"{m['mean']:.3f}"

    for rule, table in tables.items():
        for setting, m in table.items():
            cells = [cell(m["f1"]), f"{m['f1']['worst_fold']:.3f}"] + [cell(m[k]) for k in METRICS[1:]]
            print(f"  {rule:18s} {setting:18s} " + " ".join(f"{c:>15s}" for c in cells))


def print_per_gesture(title, gesture_f1, names, rule, gestures):
    """1-shot F1 per gesture, left / right, for each person (baseline -> model)."""
    b, m = gesture_f1[names[0]][rule], gesture_f1[names[1]][rule]
    print(f"\n{title}, 1-shot, left / right, cutoff rule {rule} (baseline -> model):")
    for person in sorted({setting.rsplit("_", 2)[0] for setting in m}):
        print(f"  {person}")
        for g in gestures:
            row = [d[f"{person}_{hand}_1shot"].get(g, float("nan")) for d in (b, m) for hand in HANDS]
            print(f"    {g:14s} {row[0]:.3f} / {row[1]:.3f}  ->  {row[2]:.3f} / {row[3]:.3f}")


# ---------------------------------------------------------------- final model and ONNX

def export_onnx(model, path, fixture):
    import onnx
    import onnxruntime as ort
    model.eval()
    torch.onnx.export(
        model, torch.from_numpy(fixture[:1]), path,
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        export_params=True, opset_version=12, do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(path))
    expected = embed(model, fixture)
    actual = ort.InferenceSession(path).run(None, {"input": fixture})[0]
    return float(np.abs(actual - expected).max()), expected


def train_final(clips, gestures, args, epochs, cutoffs, cutoff_range):
    """Train on every gesture and every negative clip, for the number of epochs the folds
    picked, and export for Unity. With no gestures left for validation, each rule's cutoff is
    the median of the cutoffs the folds chose on their validation gestures. That cutoff comes
    from the fold models, not this one, so treat it as a starting value to tune live."""
    print(f"\n=== Final model: all {len(gestures)} gestures, {epochs} epochs ===")
    split = split_clips(clips, hidden=(), val_gestures=(), hold_out_negatives=False)
    check_split(split, hidden=(), val_gestures=())
    model, *_ = train_model(split, gestures, (), args, seed=args.seeds[0], fold=999, epochs=epochs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Fixture for comparing PyTorch, ONNX Runtime and (later) Unity: the first frame of one clip
    # per gesture, one negative frame, and an all-zero frame.
    first = {}
    for c in split["train_pos"]:
        first.setdefault(c.category, c.frames[0])
    fixture = np.stack([first[g] for g in gestures] + [split["train_neg"][0].frames[0],
                                                       np.zeros(input_dim, dtype=np.float32)])
    onnx_path = os.path.join(OUTPUT_DIR, "siamese_embedding.onnx")
    max_diff, expected = export_onnx(model, onnx_path, fixture)
    print(f"  ONNX export: {onnx_path}\n  max |ONNX Runtime - PyTorch| on fixture: {max_diff:.2e}")
    assert max_diff < 1e-5, "ONNX output does not match PyTorch"

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "siamese_embedding.pt"))
    with open(os.path.join(OUTPUT_DIR, "siamese_fixture.json"), "w") as f:
        json.dump({"input": fixture.tolist(), "expected_embedding": expected.tolist(),
                   "rows": gestures + ["negative", "all zeros"]}, f)
    config = {
        "feature_names": FEATURE_NAMES, "gestures": gestures,
        "input": "raw features, shape [batch, 17] (z-scoring is inside the model)",
        "output": "L2-normalised embedding, shape [batch, 16]", "embedding_dim": embedding_dim,
        "no_gesture_cutoff": cutoffs[args.cutoff_rule], "cutoff_rule": args.cutoff_rule,
        "cutoffs_by_rule": cutoffs, "fold_cutoff_range_by_rule": cutoff_range,
        "cutoff_meaning": "distance to the nearest prototype > cutoff means no gesture",
        "cutoff_source": "median of the cutoffs chosen on validation gestures in cross-validation; "
                         "provisional, not measured on this model",
        "epochs": epochs, "onnx_max_diff": max_diff, "onnx_opset": 12, "args": vars(args),
    }
    with open(os.path.join(OUTPUT_DIR, "siamese_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved model, config and fixture to {OUTPUT_DIR}")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--augment", action="store_true", help="add noise to gesture frames")
    parser.add_argument("--near-miss", action="store_true", help="add synthetic near-miss negatives")
    parser.add_argument("--final", action="store_true", help="also train on all gestures and export ONNX")
    parser.add_argument("--epochs", type=int, default=num_epochs)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--cutoff-rule", choices=CUTOFF_RULES, default="best_f1",
                        help="rule for the per-gesture table and the exported cutoff (all rules are scored)")
    parser.add_argument("--test-person", help="never train on this person's clips; score on them as new hands")
    args = parser.parse_args()
    assert args.epochs >= 1

    clips = load_clips()
    gestures = sorted({c.category for c in clips if c.kind == "positive"})
    n_pos = sum(c.kind == "positive" for c in clips)
    print(f"Loaded {len(clips)} clips ({n_pos} gesture, {len(clips) - n_pos} negative), {len(gestures)} gestures")
    for person in people_in(clips):
        n = Counter(c.kind for c in clips if c.person == person)
        print(f"  {person}: {n['positive']} gesture, {n['negative']} negative"
              + ("  (no negatives: scored against everyone else's)" if not n["negative"] else ""))
    counts = Counter((c.person, c.category, c.hand) for c in clips if c.kind == "positive")
    for (person, g, hand), n in sorted(counts.items()):
        if n <= max(shots):
            print(f"  {person} {g} {hand}: only {n} clips, left out of {max(shots)}-shot episodes")
    if args.test_person:
        assert args.test_person in people_in(clips), f"no data folder for {args.test_person}"
        assert not args.final, "the shipped model must train on everyone; drop --test-person for --final"
        assert any(c.kind == "negative" and c.person != args.test_person for c in clips), \
            "training needs negatives from someone other than the test person"

    start = time.time()
    runs = [run_folds(clips, gestures, args, seed) for seed in args.seeds]

    names = ("baseline", "model")
    if args.test_person:
        names += ("known_baseline", "known_model")
    summary = {name: {rule: summarize(runs, name, rule) for rule in CUTOFF_RULES} for name in names}
    gesture_f1 = {name: {rule: per_gesture_table(runs, name, rule) for rule in CUTOFF_RULES} for name in names}
    print_table("Raw-feature baseline (no training)", summary["baseline"], len(args.seeds))
    print_table("Siamese model", summary["model"], len(args.seeds))
    print_per_gesture("Per-gesture F1 on hidden gestures", gesture_f1, names[:2], args.cutoff_rule, gestures)
    if args.test_person:
        title = f"{args.test_person} as a new person, all {len(gestures)} gestures known"
        print_table(f"{title}: raw-feature baseline", summary["known_baseline"], len(args.seeds))
        print_table(f"{title}: Siamese model", summary["known_model"], len(args.seeds))
        print_per_gesture(f"Per-gesture F1, {title}", gesture_f1, names[2:], args.cutoff_rule, gestures)
    print(f"\nFolds took {time.time() - start:.0f} s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = "model" + ("_augment" if args.augment else "") + ("_nearmiss" if args.near_miss else "")
    tag += f"_epochs{args.epochs}_seeds" + "-".join(map(str, args.seeds))
    if args.test_person:
        tag += f"_test-{args.test_person}"
    with open(os.path.join(OUTPUT_DIR, f"results_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary, "per_gesture_f1": gesture_f1, "runs": runs}, f, indent=2)

    if args.final:
        folds = [fold for run in runs for fold in run["folds"]]
        epochs = int(round(np.median([f["best_epoch"] for f in folds])))
        cutoffs = {rule: float(np.median([f["model_cutoffs"][rule] for f in folds])) for rule in CUTOFF_RULES}
        cutoff_range = {rule: [float(min(f["model_cutoffs"][rule] for f in folds)),
                               float(max(f["model_cutoffs"][rule] for f in folds))] for rule in CUTOFF_RULES}
        train_final(clips, gestures, args, epochs, cutoffs, cutoff_range)


if __name__ == "__main__":
    main()
