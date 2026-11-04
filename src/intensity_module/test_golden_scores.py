"""
test_golden_scores.py

Locks the component scores that existed before duration_score was added: every window of
every development and testing level clip, scored at the recorded rate and at 15 Hz input, must
give exactly the same windows and the same angular_speed, angular_peak_speed, angular
acceleration, opposition speed and confidence values as in golden_scores.json. `intensity`
itself is not locked, because it now also uses duration_score; evaluate.py checks it.

Regenerate only when these components are meant to change (last: the low-pass filter before
resampling):
  python test_golden_scores.py --update
"""

import json
import os
import sys
import unittest

from evaluate import downsample
from intensity_scorer import load_calibration, load_manifest, score_recording

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(HERE, "golden_scores.json")
MANIFESTS = (
    "recordings/development/bryan/manifest.csv",
    "recordings/testing/bryan/manifest.csv",
)
LOCKED = ("start_s", "angular_speed_score", "angular_peak_speed_score", "angular_acceleration_score",
          "opposition_speed_score", "confidence")


def locked(windows):
    return [{k: w[k] for k in LOCKED} for w in windows]


def current_scores():
    cal = load_calibration()
    out = {}
    for manifest in MANIFESTS:
        for c in load_manifest(os.path.join(HERE, manifest)):
            key = f"{os.path.dirname(manifest)}/{c['file']}"
            out[key] = {
                "recorded_rate": score_recording(c["frames"], c["fps"], cal),
                "15hz": score_recording(downsample(c["frames"], c["fps"], 15), 15, cal),
            }
    return out


class GoldenScoreTests(unittest.TestCase):
    def test_existing_component_scores_are_unchanged(self):
        with open(GOLDEN_PATH) as f:
            golden = json.load(f)
        current = current_scores()

        self.assertEqual(sorted(current), sorted(golden))
        for key in golden:
            for mode in ("recorded_rate", "15hz"):
                with self.subTest(clip=key, mode=mode):
                    self.assertEqual(locked(current[key][mode]), locked(golden[key][mode]))


if __name__ == "__main__":
    if "--update" in sys.argv:
        with open(GOLDEN_PATH, "w") as f:
            json.dump(current_scores(), f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {GOLDEN_PATH}")
    else:
        unittest.main()
