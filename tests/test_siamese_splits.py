"""Whole-clip, per-person splits and few-shot episodes: nothing scored may be trained on,
validated on, or used to teach the gesture it is scored against."""

import numpy as np
import pytest

from src import siamese_train as st
from tests.synthetic import clip, recordings, write_recording

HIDDEN, VAL = ["a", "b", "c"], ["d", "e"]


def parts_of(split):
    return {c.path: part for part, part_clips in split.items() for c in part_clips}


# ---------------------------------------------------------------- split_clips / check_split

def test_gestures_go_to_test_validation_or_training_by_fold():
    clips = recordings(people=("bryan", "edy"))
    split = st.split_clips(clips, HIDDEN, VAL)
    st.check_split(split, HIDDEN, VAL)
    part = parts_of(split)
    for c in clips:
        if c.kind == "positive":
            want = "test_pos" if c.category in HIDDEN else "val_pos" if c.category in VAL else "train_pos"
            assert part[c.path] == want, c.path
    assert len(part) == len(clips)  # every clip used exactly once


def test_negatives_split_by_time_within_each_person_category_and_hand():
    clips = recordings(people=("bryan",), negative_categories=("everyday", "resting"), negatives_per_hand=10)
    part = parts_of(st.split_clips(clips, HIDDEN, VAL))
    for c in clips:
        if c.kind == "negative":
            index = int(c.path[-8:-5])  # oldest 6 train, next 2 validation, newest 2 test
            assert part[c.path] == ("train_neg" if index < 6 else "val_neg" if index < 8 else "test_neg"), c.path


def test_test_person_is_never_trained_or_validated_on():
    clips = recordings(people=("bryan", "edy"))
    split = st.split_clips(clips, HIDDEN, VAL, test_person="edy")
    st.check_split(split, HIDDEN, VAL, test_person="edy")
    part = parts_of(split)
    for c in clips:
        if c.person != "edy":
            continue
        if c.kind == "negative":
            assert part[c.path] == "test_neg"
        elif c.category in HIDDEN:
            assert part[c.path] == "test_pos"
        else:
            assert c.path not in part  # edy's other gestures are left out entirely


@pytest.mark.parametrize("leak", ["clip in two parts", "hidden gesture trained", "negatives out of time order",
                                  "test person validated"])
def test_check_split_catches_leaks(leak):
    clips = recordings(people=("bryan", "edy"))
    split = st.split_clips(clips, HIDDEN, VAL, test_person="edy")
    if leak == "clip in two parts":
        split["train_pos"].append(split["test_pos"][0])
    elif leak == "hidden gesture trained":
        split["train_pos"].append(split["test_pos"].pop())
    elif leak == "negatives out of time order":
        split["train_neg"][0], split["test_neg"][0] = split["test_neg"][0], split["train_neg"][0]
    elif leak == "test person validated":
        split["val_neg"].append(next(c for c in split["test_neg"] if c.person == "edy"))
        split["test_neg"] = [c for c in split["test_neg"] if c is not split["val_neg"][-1]]
    with pytest.raises(AssertionError):
        st.check_split(split, HIDDEN, VAL, test_person="edy")


def test_folds_hide_every_gesture_three_times_and_validate_it_twice():
    gestures = ["a", "b", "c", "d", "e", "f", "g"]
    folds = st.make_folds(gestures, seed=0)
    assert len(folds) == len(gestures)
    for hidden, val in folds:
        assert len(set(hidden)) == 3 and len(set(val)) == 2 and not set(hidden) & set(val)
    for g in gestures:
        assert sum(g in hidden for hidden, _ in folds) == 3
        assert sum(g in val for _, val in folds) == 2


# ---------------------------------------------------------------- the real recordings

@pytest.fixture(scope="module")
def real_clips():
    return st.load_clips()


@pytest.mark.parametrize("test_person", [None, "edy"])
def test_real_data_folds_are_isolated(real_clips, test_person):
    gestures = sorted({c.category for c in real_clips if c.kind == "positive"})
    for hidden, val in st.make_folds(gestures, seed=0):
        split = st.split_clips(real_clips, hidden, val, test_person=test_person)
        st.check_split(split, hidden, val, test_person)
        part = parts_of(split)
        unused = [c for c in real_clips if c.path not in part]
        # Only the test person's non-hidden gestures are left out
        assert all(c.person == test_person and c.kind == "positive" and c.category not in hidden for c in unused)


def test_real_data_has_negatives_for_everyone_or_a_fallback(real_clips):
    """A person with no negatives (edy) is scored against other people's: there must be some."""
    for hand in st.HANDS:
        assert any(c.kind == "negative" and c.hand == hand for c in real_clips), hand


# ---------------------------------------------------------------- episodes

def test_episode_queries_never_include_support_clips():
    a = [clip("p", "positive", "a", "left", i, np.full((n, 2), v)) for i, (n, v) in enumerate([(2, 1.0), (3, 3.0), (1, 5.0)])]
    b = [clip("p", "positive", "b", "left", i, np.full((2, 2), v)) for i, v in enumerate([-1.0, -3.0])]
    emb = {c.path: c.frames for c in a + b}
    negatives = np.full((4, 2), 10.0, dtype=np.float32)
    prototypes, queries, true = st.build_episode(emb, [a, b], negatives, support=[{0, 1}, {1}])
    np.testing.assert_allclose(prototypes, [[2.2, 2.2], [-3.0, -3.0]], rtol=1e-6)  # (2*1 + 3*3) / 5
    np.testing.assert_array_equal(queries[:, 0], [10, 10, 10, 10, 5, -1, -1])
    np.testing.assert_array_equal(true, [st.NONE] * 4 + [0, 1, 1])


def test_support_draws_k_different_clips():
    rng = np.random.default_rng(0)
    by_gesture = [list(range(4)), list(range(5))]
    for _ in range(50):
        for support, clips in zip(st.draw_support(rng, by_gesture, 3), by_gesture):
            assert len(support) == 3 and support <= set(range(len(clips)))


def test_episode_negatives_prefer_the_persons_own():
    negatives = [clip(p, "negative", "everyday", h, 0, np.full((1, 2), v))
                 for p, h, v in [("bryan", "left", 1.0), ("bryan", "right", 2.0), ("amy", "left", 3.0)]]
    emb = {c.path: c.frames for c in negatives}
    assert st.episode_negatives(emb, negatives, "amy", "left")[:, 0].tolist() == [3.0]
    # edy recorded none: scored against everyone else's, same hand only
    assert st.episode_negatives(emb, negatives, "edy", "left")[:, 0].tolist() == [1.0, 3.0]


def test_episode_negatives_fail_clearly_when_nobody_has_any_for_that_hand():
    negatives = [clip("bryan", "negative", "everyday", "left", 0, np.zeros((1, 2)))]
    emb = {c.path: c.frames for c in negatives}
    with pytest.raises(ValueError, match="right-hand negative"):
        st.episode_negatives(emb, negatives, "edy", "right")


# ---------------------------------------------------------------- missing or insufficient clips

def test_too_few_gestures_for_a_fold_fails_clearly():
    # 3 hidden + 2 validation leaves nothing to train on with 5 gestures
    with pytest.raises(ValueError, match="at least 6 gestures"):
        st.make_folds(["a", "b", "c", "d", "e"], seed=0)


def test_hidden_gesture_with_too_few_clips_for_one_shot_is_rejected():
    clips = [c for c in recordings() if not (c.category == "a" and c.hand == "left" and c.path[-8:-5] != "000")]
    with pytest.raises(AssertionError, match="too few for 1-shot"):
        st.check_split(st.split_clips(clips, HIDDEN, VAL), HIDDEN, VAL)


def test_gesture_with_k_or_fewer_clips_is_left_out_of_k_shot_episodes():
    clips = [c for c in recordings() if not (c.category == "a" and int(c.path[-8:-5]) >= 3)]  # 3 clips of "a"
    split = st.split_clips(clips, HIDDEN, VAL)
    st.check_split(split, HIDDEN, VAL)
    model = st.RawFeatures(*st.fit_normalization(split))
    results = st.evaluate_hidden(model, split, HIDDEN, {"best_f1": 1.0}, rng_key=0)["best_f1"]
    assert set(results["bryan_left_1shot"]["per_gesture_f1"]) == {"a", "b", "c"}
    assert set(results["bryan_left_3shot"]["per_gesture_f1"]) == {"b", "c"}


def test_load_clips_reads_a_person_with_no_negatives(tmp_path):
    frames = np.zeros((3, st.input_dim))
    write_recording(tmp_path, "bryan", "positive", "pinch", "left", "t1.json", frames, 1)
    write_recording(tmp_path, "bryan", "negative", "everyday", "left", "t2.json", frames, 0)
    write_recording(tmp_path, "edy", "positive", "pinch", "right", "t3.json", frames, 1)
    clips = st.load_clips(str(tmp_path))
    assert sorted((c.person, c.kind, c.hand) for c in clips) == [
        ("bryan", "negative", "left"), ("bryan", "positive", "left"), ("edy", "positive", "right")]


@pytest.mark.parametrize("kind, confidence, n_features, match", [
    ("positive", 0, st.input_dim, "label mismatch"),
    ("negative", 1, st.input_dim, "label mismatch"),
    ("positive", 1, st.input_dim - 1, "t1.json"),
])
def test_load_clips_rejects_malformed_recordings(tmp_path, kind, confidence, n_features, match):
    write_recording(tmp_path, "bryan", kind, "pinch", "left", "t1.json", np.zeros((3, n_features)), confidence)
    with pytest.raises(AssertionError, match=match):
        st.load_clips(str(tmp_path))
