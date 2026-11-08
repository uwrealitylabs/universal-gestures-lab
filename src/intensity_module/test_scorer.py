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
    LIVE_BUFFER_S,
    LOWPASS_MIN_RATE_HZ,
    MAX_ANGULAR_SPEED,
    OPPOSITION_IDX,
    calibrate,
    find_transitions,
    load_calibration,
    load_recording,
    repair_glitches,
    resample,
    save_calibration,
    score,
    score_recording,
    score_transitions,
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


SNAP_STARTS = (2.0, 5.0, 8.0)


def snaps(joints, size, duration, fps=72.0, seconds=10.0, seed=0):
    """Rest -> gesture -> rest, three times: a smooth move into the pose over `duration`
    seconds, a 1 s hold, and a 0.3 s return, with a little tracking noise."""
    t, frames = hand(seconds, fps)
    shape = np.zeros_like(t)
    for s in SNAP_STARTS:
        into = np.clip((t - s) / duration, 0.0, 1.0)
        back = np.clip((t - (s + duration + 1.0)) / 0.3, 0.0, 1.0)
        shape += (1 - np.cos(np.pi * into)) / 2 - (1 - np.cos(np.pi * back)) / 2
    for j in joints:
        frames[:, j] += size * shape
    frames[:, ANGLE_IDX] += np.random.default_rng(seed).normal(0, 0.3, (len(t), len(ANGLE_IDX)))
    return frames


def synthetic_calibration():
    rng = np.random.default_rng(0)
    rest = [(90.0 + rng.normal(0, 0.3, (720, 17)), 72.0)]
    max_effort = [(oscillate(THUMB, 25.0, 4.5, seconds=10.0), 72.0),
                  (oscillate(FIST, 45.0, 3.5, seconds=10.0), 72.0)]
    return calibrate(max_effort, rest, transitions=[(snaps(FIST, 60.0, 0.08), 72.0)])


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

    def test_jitter_faster_than_voluntary_movement_scores_zero_at_any_amplitude(self):
        # Wiggling a finger faster scores higher up to about 3 Hz, but motion faster than a person
        # moves on purpose (tremor, tracking jitter) scores about 0 rather than as maximum effort,
        # however large it is.
        cal = load_calibration()
        def median_score(amplitude, hz):
            frames = oscillate([2, 4], amplitude, hz, seconds=4.0)
            frames[:, ANGLE_IDX] += np.random.default_rng(0).normal(0, 0.3, (len(frames), len(ANGLE_IDX)))
            return float(np.median([r["intensity"] for r in score_recording(frames, 72.0, cal)[2:]]))

        for amplitude in (13.0, 20.0, 40.0):
            with self.subTest(amplitude=amplitude):
                slow, normal, fast = (median_score(amplitude, hz) for hz in (1.0, 2.0, 3.0))
                self.assertLess(slow, normal)
                self.assertLess(normal, fast)
                self.assertGreater(fast, 0.75)
                for hz in (8.0, 10.0, 12.0):
                    self.assertLess(median_score(amplitude, hz), 0.05, f"{hz} Hz")

    def test_deliberate_movement_keeps_its_full_score(self):
        # Sharp snaps and fast repetition are deliberate, so the jitter check leaves them alone.
        for frames in (snaps(FIST, 60.0, 0.08), oscillate(FIST, 45.0, 4.5, seconds=4.0)):
            moving = [r for r in score_recording(frames, 72.0, self.cal) if r["angular_peak_speed_score"] > 0.5]
            self.assertTrue(moving)
            self.assertGreater(min(r["features"]["deliberate_share"] for r in moving), 0.8)

    def test_jitter_check_needs_at_least_30_hz_input(self):
        # Below LOWPASS_MIN_RATE_HZ, motion above half the input rate is already folded into
        # slower motion at capture, so the check is skipped rather than guessing.
        self.assertEqual(LOWPASS_MIN_RATE_HZ, 30.0)
        frames = oscillate([2, 4], 40.0, 10.0, fps=30.0, seconds=4.0)
        self.assertLess(max(r["features"]["deliberate_share"] for r in score_recording(frames, 30.0, self.cal)), 0.4)
        slow = oscillate([2, 4], 40.0, 10.0, fps=15.0, seconds=4.0)
        self.assertEqual({r["features"]["deliberate_share"] for r in score_recording(slow, 15.0, self.cal)}, {1.0})

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
        result = score(oscillate(THUMB, 25.0, 2.0), 72.0, self.cal, confidence=1, gesture="finger_gun")

        self.assertEqual(
            set(result),
            {"gesture", "intensity", "angular_speed_score", "angular_peak_speed_score", "duration_score",
             "path_score", "angular_acceleration_score", "opposition_speed_score", "confidence", "features"},
        )
        self.assertEqual(
            set(result["features"]),
            {"move_duration_s", "path_length_degrees", "average_speed_deg_per_s", "peak_speed_deg_per_s",
             "peak_acceleration_deg_per_s2", "movement_size_degrees", "deliberate_share"},
        )
        self.assertEqual(result["confidence"], 1)
        self.assertEqual(result["gesture"], "finger_gun")
        for key in ("intensity", "duration_score", "path_score"):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

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


class LiveConsistencyTests(unittest.TestCase):
    """score_recording() scores each window from only the frames up to its end, as score() on a
    live rolling buffer would, so later frames never change an earlier window."""

    @classmethod
    def setUpClass(cls):
        cls.cal = synthetic_calibration()

    def live_scores(self, frames, fps):
        # score() on the last LIVE_BUFFER_S of frames at the end of each 0.5 s window.
        steps = int(np.floor((len(frames) - 1) / fps * 15 + 1e-9))
        ends = list(range(8, steps + 1, 8)) + ([steps] if steps % 8 else [])
        buffer = int(round(LIVE_BUFFER_S * fps))
        lasts = [len(frames) if e == steps else int(np.floor(e / 15 * fps + 1e-9)) + 1 for e in ends]
        return [score(frames[max(0, last - buffer):last], fps, self.cal) for last in lasts]

    def test_every_window_matches_live_scoring(self):
        stream = snaps(FIST, 60.0, 0.1) + oscillate(THUMB, 20.0, 2.5, seconds=10.0) - 90.0
        for fps in (72.0, 15.0):
            frames = stream if fps == 72.0 else stream[np.round(np.arange(0, 10, 1 / 15) * 72).astype(int)]
            with self.subTest(fps=fps):
                offline = score_recording(frames, fps, self.cal)
                live = self.live_scores(frames, fps)
                self.assertEqual(len(offline), len(live))
                for a, b in zip(offline, live):
                    self.assertEqual({k: v for k, v in a.items() if k != "start_s"}, b)

    def test_step_at_a_window_boundary_uses_no_later_frames(self):
        # 15 Hz, still hand, one joint steps 60 degrees at frame 15 and holds. The second window
        # ends at frame 16, before the hold could be confirmed, so it must score as live does
        # then, whatever comes after.
        frames = np.full((40, 17), 90.0)
        frames[:, OPPOSITION_IDX] = 0.05
        frames[15:, 2] += 60.0

        second = score_recording(frames, 15.0, self.cal)[1]
        live = score(frames[:17], 15.0, self.cal)

        self.assertEqual({k: v for k, v in second.items() if k != "start_s"}, live)
        self.assertEqual(second, score_recording(frames[:17], 15.0, self.cal)[1])
        self.assertEqual(second["duration_score"], 0.0)


class RestToGestureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = synthetic_calibration()

    def scores(self, frames, fps=72.0):
        return [r["intensity"] for r in score_transitions(frames, fps, self.cal)]

    def test_faster_moves_into_a_pose_score_higher(self):
        means = [np.mean(self.scores(snaps(FIST, 60.0, d))) for d in (1.2, 0.5, 0.25, 0.08)]

        self.assertEqual(means, sorted(means))
        self.assertLess(means[0], 0.1)
        self.assertGreaterEqual(means[-1], 0.8)

    def test_slowly_moving_into_finger_gun_pose_scores_near_zero(self):
        # The ticket's example: curling the other fingers into a finger gun over 1.5 s.
        self.assertLess(max(self.scores(snaps([6, 8, 10, 12, 14, 15], 70.0, 1.5))), 0.1)

    def test_snap_score_does_not_depend_on_movement_size(self):
        small = self.scores(snaps(FIST, 35.0, 0.15))
        big = self.scores(snaps(FIST, 70.0, 0.15))

        self.assertAlmostEqual(np.mean(small), np.mean(big), delta=0.05)

    def test_only_moves_into_the_gesture_are_found(self):
        reached = np.array(find_transitions(snaps(FIST, 60.0, 0.2), 72.0, self.cal["gates"]["angle_degrees"])) / 72.0

        self.assertEqual(len(reached), 3)
        np.testing.assert_allclose(reached, np.array(SNAP_STARTS) + 0.2, atol=0.2)

    def test_snap_gets_duration_credit_but_flick_and_repetition_do_not(self):
        snap = score_transitions(snaps(FIST, 60.0, 0.08), 72.0, self.cal)
        t, flick = hand(3.0, 72.0)
        flick[:, THUMB] += 40.0 * np.exp(-((t - 1.5) / 0.05) ** 2)[:, None]   # out and straight back
        repeat = oscillate(FIST, 45.0, 3.0, seconds=6.0)

        self.assertTrue(all(r["duration_score"] > 0.9 for r in snap))
        self.assertEqual(max(r["duration_score"] for r in score_recording(flick, 72.0, self.cal)), 0.0)
        self.assertEqual(max(r["duration_score"] for r in score_recording(repeat, 72.0, self.cal)), 0.0)

    def test_live_stream_with_short_gestures(self):
        # One stream, scored the way an app would: score() on a rolling 2 s buffer every 0.1 s,
        # with the shipped calibration. A fast snap held 0.4 s, a 1.5 s slow move held 0.4 s,
        # then fast repetition; relaxed in between.
        cal = load_calibration()
        fps = 72.0
        t, stream = hand(14.0, fps)
        def move(start, duration, hold, size=60.0):
            into = np.clip((t - start) / duration, 0, 1)
            back = np.clip((t - start - duration - hold) / 0.3, 0, 1)
            return size * ((1 - np.cos(np.pi * into)) / 2 - (1 - np.cos(np.pi * back)) / 2)
        shape = move(2.0, 0.08, 0.4) + move(5.0, 1.5, 0.4)
        shape += np.where((t > 9.5) & (t < 12.5), 30.0 * (1 - np.cos(2 * np.pi * 3.0 * (t - 9.5))), 0.0)
        stream[:, FIST] += shape[:, None]
        stream[:, ANGLE_IDX] += np.random.default_rng(4).normal(0, 0.3, (len(t), len(ANGLE_IDX)))
        buffer = int(2.0 * fps)
        live = {round(e / fps, 1): score(stream[max(0, e - buffer):e], fps, cal)["intensity"]
                for e in range(int(0.5 * fps), len(stream) + 1, int(0.1 * fps))}
        peak = lambda a, b: max(v for k, v in live.items() if a <= k <= b)

        self.assertGreater(peak(2.1, 2.6), 0.85)      # the fast snap, read just after it
        self.assertLess(peak(3.4, 4.8), 0.05)         # relaxed again
        self.assertLess(peak(5.0, 6.8), 0.3)          # the slow move: at most a brief bump
        self.assertLess(peak(6.6, 6.8), 0.05)         # and about 0 once in the pose
        self.assertGreater(peak(10.0, 12.4), 0.85)    # fast repetition, same formula

    def test_timestamps_score_like_the_measured_rate(self):
        frames = snaps(FIST, 60.0, 0.15)
        times = np.arange(len(frames)) / 72.0 + np.random.default_rng(5).normal(0, 0.001, len(frames))
        times.sort()
        with_rate = score(frames[300:444], 72.0, self.cal)
        with_times = score(frames[300:444], calibration=self.cal, timestamps=times[300:444])

        self.assertAlmostEqual(with_rate["intensity"], with_times["intensity"], delta=0.05)

    def test_still_hand_has_no_moves(self):
        still = 90.0 + np.random.default_rng(2).normal(0, 0.3, (720, 17))
        results = score_recording(still, 72.0, self.cal)

        self.assertEqual(find_transitions(still, 72.0, self.cal["gates"]["angle_degrees"]), [])
        self.assertEqual(max(r["duration_score"] for r in results), 0.0)
        self.assertLess(np.mean([r["intensity"] for r in results]), 0.05)


class CalibrationTests(unittest.TestCase):
    def test_duration_and_path_are_added_without_changing_the_rest(self):
        rng = np.random.default_rng(0)
        rest = [(90.0 + rng.normal(0, 0.3, (720, 17)), 72.0)]
        max_effort = [(oscillate(FIST, 45.0, 3.5, seconds=10.0), 72.0)]
        without = calibrate(max_effort, rest)
        with_snaps = calibrate(max_effort, rest, transitions=[(snaps(FIST, 60.0, 0.08), 72.0)])

        for key in ("gates", "components", "curve", "path"):
            self.assertEqual(without[key], with_snaps[key])
        self.assertIsNone(without["duration"]["full_rise_s"])
        self.assertLess(with_snaps["duration"]["full_rise_s"], 0.1)

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

    def test_rest_to_gesture_clips_meet_the_same_targets(self):
        snaps_ = [clip("pinch", "transition", lv, s) for lv, s in (("1", [0.0]), ("2", [0.3]), ("3", [0.6]), ("4", [0.9]))]
        self.assertEqual(check(passing_clips() + snaps_), [])

        snaps_[3]["intensity"] = [0.6]
        snaps_[0]["intensity"] = [0.2]
        failures = check(passing_clips() + snaps_)

        self.assertIn("pinch (rest -> gesture) level 4 averages 0.600 (target >= 0.85)", failures)
        self.assertIn("pinch (rest -> gesture) level 1 averages 0.200 (target <= 0.15)", failures)

    def test_rest_to_gesture_gesture_needs_every_level(self):
        clips = passing_clips() + [clip("pinch", "transition", "4", [0.9])]

        self.assertIn("pinch (rest -> gesture) has no level 1, 2, 3 clips", check(clips))


if __name__ == "__main__":
    unittest.main()
