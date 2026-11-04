"""Smoke tests: a few training batches on small synthetic recordings, through the same code
paths as a real run, including the command line and the ONNX export for Unity."""

import argparse
import json
import sys

import numpy as np
import pytest
import torch

from src import siamese_train as st
from tests.synthetic import recordings

GESTURES = ["a", "b", "c", "d", "e", "f", "g", "h"]
HIDDEN, VAL, TRAIN = ["a", "b", "c"], ["d", "e"], ["f", "g", "h"]


@pytest.fixture
def few_batches(monkeypatch):
    monkeypatch.setattr(st, "batches_per_epoch", 5)
    monkeypatch.setattr(st, "episodes", 2)


def options(augment=False, near_miss=False):
    return argparse.Namespace(augment=augment, near_miss=near_miss)


@pytest.mark.parametrize("augment, near_miss", [(False, False), (True, True)])
def test_training_updates_the_model_and_picks_cutoffs_on_validation(few_batches, augment, near_miss):
    split = st.split_clips(recordings(gestures=GESTURES), HIDDEN, VAL)
    st.check_split(split, HIDDEN, VAL)
    model, cutoffs, val_f1, epoch = st.train_model(split, TRAIN, VAL, options(augment, near_miss),
                                                   seed=0, fold=0, epochs=2)
    assert list(cutoffs) == list(st.CUTOFF_RULES)
    assert 0 <= val_f1 <= 1 and epoch in (1, 2)

    torch.manual_seed(0)  # train_model seeds with seed * 1000 + fold before building the model
    untrained = st.EmbeddingNet(*st.fit_normalization(split))
    assert any(not torch.equal(a, b) for a, b in zip(untrained.state_dict().values(), model.state_dict().values()))

    frames = np.concatenate([c.frames for c in split["test_pos"]])
    np.testing.assert_allclose(np.linalg.norm(st.embed(model, frames), axis=1), 1, atol=1e-5)

    # The synthetic gestures are far apart, so a trained model still tells unseen ones apart
    results = st.evaluate_hidden(model, split, HIDDEN, cutoffs, rng_key=0)
    for rule, settings in results.items():
        for setting, scores in settings.items():
            assert scores["closed_set_accuracy"] > 0.9, (rule, setting)


def test_training_gesture_with_too_few_clips_fails_clearly(few_batches):
    clips = [c for c in recordings(gestures=GESTURES, clips_per_hand=3)]  # 6 clips, batches need 8
    split = st.split_clips(clips, HIDDEN, VAL)
    with pytest.raises(ValueError, match=f"at least {st.clips_per_gesture} clips"):
        st.train_model(split, TRAIN, VAL, options(), seed=0, fold=0, epochs=1)


def test_training_without_negatives_fails_clearly(few_batches):
    clips = [c for c in recordings(gestures=GESTURES) if c.kind == "positive"]
    split = st.split_clips(clips, HIDDEN, VAL)
    with pytest.raises(ValueError, match="negative"):
        st.train_model(split, TRAIN, VAL, options(), seed=0, fold=0, epochs=1)


# ---------------------------------------------------------------- whole command line

def run_main(monkeypatch, tmp_path, clips, *argv):
    monkeypatch.setattr(st, "load_clips", lambda: clips)
    monkeypatch.setattr(st, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["siamese_train", "--epochs", "1", *argv])
    st.main()


def test_cross_validation_and_final_export(few_batches, monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path, recordings(gestures=GESTURES[:6]), "--final")
    results = json.loads((tmp_path / "results_model_epochs1_seeds0.json").read_text())
    assert len(results["runs"][0]["folds"]) == 6

    config = json.loads((tmp_path / "siamese_config.json").read_text())
    assert config["gestures"] == GESTURES[:6]
    assert config["no_gesture_cutoff"] == config["cutoffs_by_rule"][config["cutoff_rule"]]
    assert config["onnx_max_diff"] < 1e-5
    for name in ("siamese_embedding.onnx", "siamese_embedding.pt", "siamese_fixture.json"):
        assert (tmp_path / name).exists(), name


def test_held_out_person_without_negatives(few_batches, monkeypatch, tmp_path):
    """Mirrors the real data: edy recorded no negatives, so edy is scored against bryan's."""
    clips = [c for c in recordings(people=("bryan", "edy"), gestures=GESTURES[:6])
             if not (c.person == "edy" and c.kind == "negative")]
    run_main(monkeypatch, tmp_path, clips, "--test-person", "edy")
    results = json.loads((tmp_path / "results_model_epochs1_seeds0_test-edy.json").read_text())
    # Hidden gestures are new to the model for everyone, so both people are scored on them
    assert {"edy_left_1shot", "bryan_left_1shot"} <= set(results["summary"]["model"]["best_f1"])
    # Known gestures are only new for edy's hands, so only edy is scored on them
    assert {s.split("_")[0] for s in results["summary"]["known_model"]["best_f1"]} == {"edy"}
