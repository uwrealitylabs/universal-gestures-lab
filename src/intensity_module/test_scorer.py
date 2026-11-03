import json
import os
import tempfile
import unittest

import numpy as np

from evaluate import check
from intensity_scorer import (
    ABDUCTION_IDX,
    ANGLE_IDX,
    CURL_IDX,
    FLEXION_IDX,
    MAX_ANGULAR_SPEED,
    OPPOSITION_IDX,
    calibrate,
    load_calibration,
    load_recording,
    repair_glitches,
    resample,
    save_calibration,
    score,
    score_recording,
    step_speeds,
    windows,
)

THUMB = [0, 1]
FIST = [2, 4, 6, 8, 10, 12, 14, 15]


def hand(seconds, fps):
    t = np.arange(int(round(seconds * fps))) / fps
    frames = np.full((len(t), 17), 90.0)
    frames[:, OPPOSITION_IDX] = 0.05
    return t, frames


def oscillate(joints, amplitude, hz, fps=72.0, seconds=3.0):
    """Joints swing back and forth by `amplitude` degrees, `hz` times per second."""
    t, frames = hand(seconds, fps)
    for j in joints:
        frames[:, j] += amplitude * np.sin(2 * np.pi * hz * t)
    return frames


def single_move(joints, size, duration, fps=72.0, seconds=1.5):
    """Joints move once by `size` degrees over `duration` seconds, ending at the last frame."""
    t, frames = hand(seconds, fps)
    start = t[-1] - duration
    progress = np.clip((t - start) / duration, 0.0, 1.0)
    for j in joints:
        frames[:, j] += size * progress
    return frames


def synthetic_calibration():
    rng = np.random.default_rng(0)
    rest = [(90.0 + rng.normal(0, 0.3, (720, 17)), 72.0)]
    max_effort = [(oscillate(THUMB, 25.0, 4.5, seconds=10.0), 72.0),
                  (oscillate(FIST, 45.0, 3.5, seconds=10.0), 72.0)]
    return calibrate(max_effort, rest)


class FeatureLayoutTests(unittest.TestCase):
    def test_feature_groups_cover_all_inputs_once(self):
        indices = CURL_IDX + ABDUCTION_IDX + FLEXION_IDX + OPPOSITION_IDX

        self.assertEqual(sorted(indices), list(range(17)))


class LoadRecordingTests(unittest.TestCase):
    def write(self, data):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        self.addCleanup(os.remove, f.name)
        return f.name

    def test_static_clip_is_one_sequence(self):
        path = self.write([{"confidence": 1, "handData": [0.0] * 17} for _ in range(5)])

        (frames, confidence), = load_recording(path)

        self.assertEqual(frames.shape, (5, 17))
        self.assertEqual(confidence, 1)

    def test_dynamic_file_keeps_each_sequence(self):
        path = self.write([{"confidence": 0, "sequenceData": [[0.0] * 17] * 15} for _ in range(3)])

        entries = load_recording(path)

        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0][0].shape, (15, 17))

    def test_wrong_width_is_rejected(self):
        path = self.write([{"confidence": 1, "handData": [0.0] * 20}])

        with self.assertRaises(ValueError):
            load_recording(path)


class TimingTests(unittest.TestCase):
    def test_ten_seconds_gives_the_same_windows_at_72_and_15_hz(self):
        at_72 = windows(np.zeros((720, 17)), 72.0)
        at_15 = windows(np.zeros((150, 17)), 15.0)

        self.assertEqual(len(at_72), len(at_15))

    def test_same_motion_scores_alike_at_72_and_15_hz(self):
        cal = synthetic_calibration()

        fast = score(oscillate(THUMB, 25.0, 2.0, fps=72.0), 72.0, cal)
        slow = score(oscillate(THUMB, 25.0, 2.0, fps=15.0), 15.0, cal)

        self.assertAlmostEqual(fast["intensity"], slow["intensity"], delta=0.1)

    def test_resample_keeps_constant_speed(self):
        _, frames = hand(1.0, 72.0)
        frames[:, 0] += 200.0 * np.arange(len(frames)) / 72.0

        angular, _ = step_speeds(resample(frames, 72.0))

        np.testing.assert_allclose(angular[:, 0], 200.0, rtol=1e-6)


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = synthetic_calibration()

    def test_still_hand_scores_zero(self):
        self.assertEqual(score(np.full((108, 17), 90.0), 72.0, self.cal)["intensity"], 0.0)

    def test_tracking_noise_scores_near_zero(self):
        noise = 90.0 + np.random.default_rng(1).normal(0, 0.3, (108, 17))

        self.assertLess(score(noise, 72.0, self.cal)["intensity"], 0.05)

    def test_incidental_twitches_of_other_fingers_barely_count(self):
        clean = oscillate(THUMB, 25.0, 3.0)
        twitchy = clean.copy()
        twitchy[:, [6, 3]] += 4.0 * np.sin(2 * np.pi * 0.7 * np.arange(len(clean)) / 72.0)[:, None]

        self.assertAlmostEqual(score(clean, 72.0, self.cal)["intensity"],
                               score(twitchy, 72.0, self.cal)["intensity"], delta=0.05)

    def test_movement_size_does_not_change_the_score(self):
        big = score(oscillate(THUMB, 30.0, 2.0), 72.0, self.cal)
        small = score(oscillate(THUMB, 15.0, 2.0), 72.0, self.cal)

        self.assertAlmostEqual(big["intensity"], small["intensity"], delta=0.05)

    def test_faster_repetition_scores_higher(self):
        slow = score(oscillate(THUMB, 25.0, 0.7), 72.0, self.cal)
        fast = score(oscillate(THUMB, 25.0, 3.0), 72.0, self.cal)

        self.assertLess(slow["intensity"], 0.3)
        self.assertGreater(fast["intensity"], slow["intensity"] + 0.3)

    def test_two_joint_and_eight_joint_gestures_score_alike_at_the_same_tempo(self):
        thumb = score(oscillate(THUMB, 25.0, 2.5), 72.0, self.cal)
        fist = score(oscillate(FIST, 45.0, 2.5), 72.0, self.cal)

        self.assertAlmostEqual(thumb["intensity"], fist["intensity"], delta=0.05)

    def test_slow_sustained_movement_scores_near_zero(self):
        # Moving into a pose over three seconds, as in the ticket's slow finger gun example.
        slow = score(single_move(THUMB, 60.0, 3.0, seconds=3.0), 72.0, self.cal)

        self.assertLess(slow["intensity"], 0.1)

    def test_single_quick_flick_beats_single_slow_move(self):
        quick = score(single_move(THUMB, 40.0, 0.15), 72.0, self.cal)
        slow = score(single_move(THUMB, 40.0, 1.0), 72.0, self.cal)

        self.assertGreater(quick["intensity"], slow["intensity"] + 0.2)

    def test_glitch_jump_out_and_back_is_repaired(self):
        frames = single_move(THUMB, 30.0, 1.0, seconds=1.5)
        glitched = frames.copy()
        glitched[50:57, 1] += 80.0  # an impossible 80 degree jump that returns 0.1 s later

        np.testing.assert_allclose(repair_glitches(glitched, 72.0), frames, atol=1.0)

    def test_fast_real_motion_is_not_repaired(self):
        frames = oscillate(THUMB, 30.0, 4.5)

        np.testing.assert_array_equal(repair_glitches(frames, 72.0), frames)

    def test_glitch_in_slow_movement_barely_changes_the_score(self):
        frames = oscillate(THUMB, 25.0, 0.7)
        glitched = frames.copy()
        glitched[-30:-23, 1] += 80.0

        self.assertAlmostEqual(score(frames, 72.0, self.cal)["intensity"],
                               score(glitched, 72.0, self.cal)["intensity"], delta=0.05)

    def test_glitch_steps_are_capped(self):
        frames = np.zeros((10, 17))
        frames[5:, ANGLE_IDX] = 180.0  # an impossible jump within one step

        angular, _ = step_speeds(frames)

        self.assertLessEqual(angular.max(), MAX_ANGULAR_SPEED)

    def test_output_keys(self):
        result = score(oscillate(THUMB, 25.0, 2.0), 72.0, self.cal, confidence=1)

        self.assertEqual(
            set(result),
            {"intensity", "angular_speed_score", "angular_peak_speed_score", "angular_acceleration_score",
             "opposition_speed_score", "confidence"},
        )
        self.assertEqual(result["confidence"], 1)

    def test_recording_is_scored_every_half_second(self):
        results = score_recording(oscillate(THUMB, 25.0, 2.0, seconds=10.0), 72.0, self.cal)

        # 149 steps at 15 Hz: 18 full windows of 8 steps, plus one aligned to the end.
        self.assertEqual(len(results), 19)
        self.assertEqual(results[1]["start_s"], round(8 / 15, 3))
        self.assertEqual(results[-1]["start_s"], round((149 - 8) / 15, 3))

    def test_flick_in_the_trailing_partial_window_is_scored(self):
        # 15 frames at 15 Hz: one full window of 8 steps, then a flick in the remaining 6.
        frames = np.full((15, 17), 90.0)
        frames[:, OPPOSITION_IDX] = 0.05
        frames[9:, 0] += [0.0, 40.0, 80.0, 80.0, 40.0, 0.0]

        results = score_recording(frames, 15.0, self.cal)

        self.assertEqual(len(results), 2)
        self.assertGreater(results[-1]["intensity"], 0.5)
        self.assertEqual(results[-1]["intensity"], score(frames, 15.0, self.cal)["intensity"])


class CalibrationTests(unittest.TestCase):
    def test_round_trip_through_json(self):
        cal = synthetic_calibration()
        clip = oscillate(THUMB, 20.0, 1.5)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "feature_ranges.json")
            save_calibration(cal, path)
            loaded = load_calibration(path)

        self.assertEqual(score(clip, 72.0, cal), score(clip, 72.0, loaded))

    def test_gates_come_from_rest_and_max_from_effort(self):
        cal = synthetic_calibration()

        self.assertLess(cal["gates"]["angle_degrees"], 5)
        self.assertGreater(cal["components"]["angular_speed"]["max"], cal["components"]["angular_speed"]["zero"])
        self.assertIsNone(cal["components"]["opposition_speed"]["max"])  # the synthetic hand never moves it


def clip(gesture, condition, level, intensity):
    return {"gesture": gesture, "condition": condition, "level": level,
            "file": f"{gesture}_{condition}{level}.json", "intensity": intensity}


def passing_clips():
    return [clip("fist", "level", lv, s) for lv, s in (("1", [0.1]), ("2", [0.4]), ("3", [0.6]), ("4", [0.95]))] + [
        clip("none", "still", "", [0.0]),
        clip("fist", "wrist_only", "", [0.01]),
    ]


class EvaluateTests(unittest.TestCase):
    def test_complete_passing_set_has_no_failures(self):
        self.assertEqual(check(passing_clips()), [])

    def test_clip_with_no_scores_fails(self):
        clips = passing_clips()
        clips[1]["intensity"] = []

        self.assertIn("fist_level2.json produced no scores", check(clips))

    def test_empty_control_fails(self):
        clips = passing_clips()
        clips[4]["intensity"] = []

        self.assertIn("none_still.json produced no scores", check(clips))

    def test_non_finite_scores_fail(self):
        clips = passing_clips()
        clips[5]["intensity"] = [float("nan")]

        self.assertIn("fist_wrist_only.json produced non-finite scores", check(clips))

    def test_missing_level_fails(self):
        clips = [c for c in passing_clips() if c["level"] != "4"]

        self.assertIn("fist has no level 4 clips", check(clips))

    def test_missing_controls_fail(self):
        clips = [c for c in passing_clips() if c["condition"] == "level"]

        self.assertEqual(check(clips), ["no still clips", "no wrist_only clips"])

    def test_no_level_clips_fails(self):
        clips = [c for c in passing_clips() if c["condition"] != "level"]

        self.assertIn("no level clips", check(clips))


if __name__ == "__main__":
    unittest.main()
