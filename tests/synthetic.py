"""Small, separable synthetic recordings for exercising src/siamese_train.py without the real data."""

import json
import os

import numpy as np

from src.siamese_train import HANDS, Clip, input_dim


def clip(person, kind, category, hand, index, frames):
    """A clip whose path sorts by `index`, like the timestamped file names do."""
    return Clip(f"{person}/{kind}/{category}/{hand}/{index:03d}.json", person, kind, category, hand,
                np.asarray(frames, dtype=np.float32))


def recordings(people=("bryan",), gestures=("a", "b", "c", "d", "e", "f"), clips_per_hand=8,
               negatives_per_hand=10, negative_categories=("everyday",), frames=12, seed=0):
    """Each gesture's frames sit near its own centre; negative frames are spread widely."""
    rng = np.random.default_rng(seed)
    centres = {g: rng.normal(size=input_dim) * 3 for g in gestures}
    clips = []
    for person in people:
        for hand in HANDS:
            for g in gestures:
                for i in range(clips_per_hand):
                    clips.append(clip(person, "positive", g, hand, i,
                                      centres[g] + 0.1 * rng.normal(size=(frames, input_dim))))
            for category in negative_categories:
                for i in range(negatives_per_hand):
                    clips.append(clip(person, "negative", category, hand, i,
                                      rng.normal(size=(frames, input_dim)) * 3))
    return clips


def write_recording(data_dir, person, kind, category, hand, name, frames, confidence):
    """Write a clip in the JSON format Unity's UGDataExtractorScript records."""
    folder = os.path.join(data_dir, person, kind, category, hand)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "w") as f:
        json.dump([{"handData": list(map(float, row)), "confidence": confidence} for row in frames], f)
