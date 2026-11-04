"""The "no gesture" cutoff: predict() fires when distance <= cutoff, so a cutoff meant to keep a
distance from firing has to sit strictly below it, in the distances' own float32 type."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.siamese_train import (NONE, candidate_cutoffs, cutoff_rules, false_trigger_cutoff,
                               macro_f1_curve, predict)


def f32(values):
    return np.asarray(values, dtype=np.float32)


def fired(dist, cutoff):
    """How many of these distances fire a gesture at this cutoff."""
    dist = f32(dist)
    return int((predict(np.zeros(len(dist), dtype=int), dist, cutoff) != NONE).sum())


def most_that_can_fire(dist, limit):
    """Brute force: the most frames any cutoff can fire without firing more than `limit`."""
    dist = f32(dist)
    reachable = [0] + [int((dist <= v).sum()) for v in np.unique(dist)]
    return max(n for n in reachable if n <= limit)


# Two adjacent float32 values whose float32 midpoint rounds up to the larger one
ODD = np.nextafter(np.float32(0.5), np.float32(1))
NEXT = np.nextafter(ODD, np.float32(1))


def test_midpoint_of_odd_and_next_rounds_up():
    # Guards the fixture: if this stops holding, the adjacent-float tests below test nothing
    assert (ODD + NEXT) / np.float32(2) == NEXT


# ---------------------------------------------------------------- false_trigger_cutoff

@pytest.mark.parametrize("dist, rate, want_fired", [
    ([0, 0, 0, 1], 0.01, 0),                    # review example: was cutoff 0, firing 3 of 4
    ([0] * 10 + [1] * 90, 0.05, 0),             # 5 may fire, but the 10 zeros are tied
    ([0.1, 0.2, 0.2, 0.2, 0.3], 0.4, 1),        # 2 may fire; the three tied 0.2s must not
    ([0, 0, 0.5, 0.5, 1], 0.4, 2),              # 2 may fire: exactly the two zeros
    ([ODD, NEXT, NEXT, NEXT], 0.25, 1),         # boundary and the distance below are adjacent floats
    ([0.3, 0.4, 0.5, 0.6], 0.0, 0),             # rate 0: nothing fires
])
def test_false_trigger_cutoff_hand_checked(dist, rate, want_fired):
    cutoff = false_trigger_cutoff(f32(dist), rate)
    assert fired(dist, cutoff) == want_fired


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("rate", [0.01, 0.05, 0.25])
def test_false_trigger_cutoff_fires_as_many_as_allowed_and_no_more(seed, rate):
    rng = np.random.default_rng(seed)
    # Few distinct values, so ties and zeros are common
    dist = f32(rng.choice([0.0, 0.0, 0.1, 0.2, 0.2, 0.3, 0.7, ODD, NEXT], size=rng.integers(1, 200)))
    limit = int(np.floor(rate * len(dist)))
    assert fired(dist, false_trigger_cutoff(dist, rate)) == most_that_can_fire(dist, limit)


def test_false_trigger_cutoff_needs_negatives():
    with pytest.raises(ValueError, match="negative"):
        false_trigger_cutoff(f32([]), 0.01)


@pytest.mark.parametrize("rate", [-0.1, 1.0])
def test_false_trigger_cutoff_rejects_rates_outside_0_to_1(rate):
    with pytest.raises(ValueError, match="rate"):
        false_trigger_cutoff(f32([0.1, 0.2]), rate)


# ---------------------------------------------------------------- candidate_cutoffs

@pytest.mark.parametrize("dist", [
    [0, 0, 0.3, 0.5],
    [0.2, 0.2, 0.4],
    [0, 0, 0],
    [ODD, NEXT],
    [0, ODD, NEXT, NEXT, 1],
])
def test_candidate_cutoffs_reach_every_achievable_count(dist):
    """For every number of frames some cutoff can fire, including none at all, one of the
    candidates fires exactly that many, so the best-F1 search never skips an option."""
    counts = {fired(dist, c) for c in candidate_cutoffs(f32(dist))}
    want = {0} | {int((f32(dist) <= v).sum()) for v in np.unique(f32(dist))}
    assert counts == want


# ---------------------------------------------------------------- cutoff_rules

def test_every_rule_scores_the_same_as_scikit_learn_with_zero_distances_and_ties():
    """macro_f1_curve (searchsorted) and predict (<=) must agree at every cutoff a rule picks,
    including a reject-everything cutoff below a zero distance."""
    true = np.array([0, 0, 0, 1, 1, 1, NONE, NONE, NONE, NONE, NONE, NONE])
    nearest = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    dist = f32([0, 0.2, 0.2, 0, 0.3, 0.3, 0, 0, 0.2, 0.3, 0.9, 0.9])
    labels = [0, 1]
    rules, best_f1 = cutoff_rules(true, nearest, dist, labels)
    for rule, cutoff in rules.items():
        ours = float(macro_f1_curve(true, nearest, dist, labels, np.array([cutoff], dtype=np.float32))[0])
        theirs = f1_score(true, predict(nearest, dist, cutoff), labels=labels, average="macro", zero_division=0)
        assert np.isclose(ours, theirs), rule
        assert ours <= best_f1 + 1e-9, rule
    negatives = dist[true == NONE]
    assert fired(negatives, rules["false_trigger_1pct"]) == 0   # 6 negatives: 1% allows none
    assert fired(negatives, rules["false_trigger_5pct"]) == 0
