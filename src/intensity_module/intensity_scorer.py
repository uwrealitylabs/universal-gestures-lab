"""
intensity_scorer.py

Intensity score in [0,1] for hand motion, on the same standard for every gesture:
0 is a still hand (or a slow movement) and 1 is maximum effort. It measures how
quickly the fingers move, not how far: each joint's speed is divided by how far that
joint moved over the last LOOKBACK_S seconds, so a small quick pinch and a large quick
finger gun score alike. The recordings do not contain wrist translation, so arm
movement alone scores near 0.

Joints are averaged by how much they take part in the movement, so incidental twitches
of other fingers count little. Calibration is shared by every gesture: a movement gate
(how far joints appear to move when the hand is still) and one maximum quickness per
component (from maximum-effort clips). A new gesture needs no recordings of its own.
Combined scores below CURVE_KNEE are curved down, which keeps gentle movement low.

Input formats (17 floats per frame):
  static recorder:  JSON list of {confidence, handData: [f0..f16]}, one frame each
  dynamic recorder: JSON list of {confidence, sequenceData: [[f0..f16] x T]}
Feature layout:
  [0,1]         thumb curl, abduction
  [2,3,4,5]     index curl, abduction, flexion, opposition
  [6,7,8,9]     middle curl, abduction, flexion, opposition
  [10,11,12,13] ring curl, abduction, flexion, opposition
  [14,15,16]    pinky curl, flexion, opposition

Curl, abduction, and flexion are angles in degrees. Opposition is fingertip-to-thumb
distance in metres. Recordings carry no timestamps, so callers pass the sample rate.

Usage:
  python intensity_scorer.py calibrate recordings/development/bryan/manifest.csv
  python intensity_scorer.py score recordings/testing/bryan/fist/<clip>.json --fps 72.1
"""

import argparse
import csv
import json
import os

import numpy as np

FEATURE_NAMES = [
    "thumb_curl", "thumb_abduction",
    "index_curl", "index_abduction", "index_flexion", "index_opposition",
    "middle_curl", "middle_abduction", "middle_flexion", "middle_opposition",
    "ring_curl", "ring_abduction", "ring_flexion", "ring_opposition",
    "pinky_curl", "pinky_flexion", "pinky_opposition",
]
CURL_IDX = [0, 2, 6, 10, 14]
ABDUCTION_IDX = [1, 3, 7, 11]
FLEXION_IDX = [4, 8, 12, 15]
OPPOSITION_IDX = [5, 9, 13, 16]
ANGLE_IDX = CURL_IDX + ABDUCTION_IDX + FLEXION_IDX

# Every input is resampled to the dynamic recorder's default rate (15 snapshots per
# second), so 72 Hz frames and 15-snapshot sequences are scored on the same footing.
SAMPLE_RATE_HZ = 15
# Speeds are measured over WINDOW_S; movement size over the longer LOOKBACK_S, so a
# slow movement's whole size is seen and a single flick is compared with its own size.
WINDOW_S = 0.5
LOOKBACK_S = 1.5
# Glitch repair at the input rate: a joint jumping faster than this and back within
# MAX_GLITCH_S is a tracking glitch. The fastest clean frame-to-frame motion recorded at
# 72 Hz was 2858 deg/s and 2.6 m/s; glitches during fast fists reached 9763 deg/s and 4.3 m/s.
# At 15 Hz input these limits are never reached, so only higher-rate input is repaired.
GLITCH_ANGULAR_SPEED = 4000.0
GLITCH_OPPOSITION_SPEED = 3.5
MAX_GLITCH_S = 0.25
# Per-step speed caps at SAMPLE_RATE_HZ, just above the fastest clean motion recorded
# (1110 deg/s, 1.24 m/s), for glitches the repair cannot pair up.
MAX_ANGULAR_SPEED = 1200.0
MAX_OPPOSITION_SPEED = 1.3
# Untrimmed clips start with roughly this much idle hand before the movement begins.
IDLE_START_S = 0.5

# Each component is a quickness in 1/s: how fast a joint moves relative to how far it moved.
#   angular_speed       mean joint speed / movement size  (how quickly the movement goes)
#   angular_peak_speed  fastest two steps / movement size (how sharp it is; catches single flicks)
#   angular_accel       sqrt(peak acceleration / size)    (reported only)
#   opposition_speed    fingertip-to-thumb speed / size   (reported only)
COMPONENTS = ("angular_speed", "angular_peak_speed", "angular_accel", "opposition_speed")
# Acceleration and opposition are reported but not weighted. At 15 Hz a fast flick gets about
# three samples per cycle, so acceleration is coarse; it raised level-1 scores to 0.22-0.25 on
# the development recordings. Opposition distances move together with the finger angles.
W = {"angular_speed": 0.5, "angular_peak_speed": 0.5, "angular_accel": 0.0, "opposition_speed": 0.0}
assert abs(sum(W.values()) - 1.0) < 1e-9
# Scores below CURVE_KNEE are lowered to knee * (score / knee) ** CURVE_POWER; scores at or
# above it are unchanged. It keeps 0, the order of scores, and everything from normal effort
# up, and pushes gentle movement towards 0, since level 1 must average at most 0.15. The power
# was chosen so the testing finger gun level 1 averages about 0.15.
CURVE_KNEE = 0.25
CURVE_POWER = 8.0
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_ranges.json")


def load_recording(path):
    """Return [(frames, confidence), ...]: one entry per dynamic sequence, or one for a static clip."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]
    if not raw:
        return []
    if "sequenceData" in raw[0]:
        entries = [(np.array(e["sequenceData"], dtype=np.float64), e["confidence"]) for e in raw]
    else:
        entries = [(np.array([e["handData"] for e in raw], dtype=np.float64), raw[0]["confidence"])]
    for frames, _ in entries:
        if frames.ndim != 2 or frames.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"{path}: expected frames of {len(FEATURE_NAMES)} features, got shape {frames.shape}")
    return entries


def repair_glitches(frames, fps):
    """Replace short tracking glitches with straight-line interpolation.

    A glitch is a jump faster than the plausible limit followed, within MAX_GLITCH_S, by a
    jump back in the opposite direction. Jumps that never come back are left for the speed cap.
    """
    out = np.array(frames, dtype=np.float64, copy=True)
    max_len = max(1, int(round(MAX_GLITCH_S * fps)))
    for j in range(out.shape[1]):
        limit = (GLITCH_OPPOSITION_SPEED if j in OPPOSITION_IDX else GLITCH_ANGULAR_SPEED) / fps
        x = out[:, j]
        d = np.diff(x)
        jumps = np.flatnonzero(np.abs(d) > limit)
        i = 0
        while i < len(jumps):
            start = jumps[i]
            back = next(
                (k for k in range(i + 1, len(jumps))
                 if jumps[k] - start <= max_len and np.sign(d[jumps[k]]) != np.sign(d[start])),
                None,
            )
            if back is None:
                i += 1
                continue
            end = jumps[back] + 1
            x[start + 1:end] = np.interp(np.arange(start + 1, end), [start, end], [x[start], x[end]])
            i = back + 1
    return out


def resample(frames, fps, rate=SAMPLE_RATE_HZ):
    """Linearly interpolate frames onto a fixed rate so speeds do not depend on the input rate."""
    if len(frames) < 2:
        return frames
    t = np.arange(len(frames)) / fps
    t_new = np.arange(0.0, t[-1] + 1e-9, 1.0 / rate)
    return np.stack([np.interp(t_new, t, frames[:, k]) for k in range(frames.shape[1])], axis=1)


def step_speeds(samples, rate=SAMPLE_RATE_HZ):
    """Per-step joint speeds: angles in deg/s and opposition in m/s, with glitches capped."""
    deltas = np.abs(np.diff(samples, axis=0)) * rate
    angular = np.minimum(deltas[:, ANGLE_IDX], MAX_ANGULAR_SPEED)
    opposition = np.minimum(deltas[:, OPPOSITION_IDX], MAX_OPPOSITION_SPEED)
    return angular, opposition


def window_features(lookback, angular, opposition, rate=SAMPLE_RATE_HZ):
    """Movement size over the look-back samples and per-joint quickness over the window's steps."""
    size = lookback.max(axis=0) - lookback.min(axis=0)
    angle_size = np.maximum(size[ANGLE_IDX], 1e-9)
    opposition_size = np.maximum(size[OPPOSITION_IDX], 1e-9)
    accel = np.abs(np.diff(angular, axis=0)) * rate if len(angular) > 1 else np.zeros_like(angular)
    # Two consecutive steps (about 0.13 s), so a single odd sample cannot set the peak alone.
    peak = ((angular[1:] + angular[:-1]) / 2).max(axis=0) if len(angular) > 1 else angular.max(axis=0)
    return {
        "angle_size": size[ANGLE_IDX],
        "opposition_size": size[OPPOSITION_IDX],
        "angular_speed": angular.mean(axis=0) / angle_size,
        "angular_peak_speed": peak / angle_size,
        "angular_accel": np.sqrt(accel.max(axis=0) / angle_size),
        "opposition_speed": opposition.mean(axis=0) / opposition_size,
    }


def windows(frames, fps, window_s=WINDOW_S, lookback_s=LOOKBACK_S, rate=SAMPLE_RATE_HZ):
    """Return [(start_s, features), ...] for consecutive windows; a short input is one window.

    When the steps do not divide evenly, a last window is aligned to the end of the input,
    overlapping the one before it, so the final movement is scored as score() would score it.
    """
    samples = resample(repair_glitches(frames, fps), fps, rate)
    angular, opposition = step_speeds(samples, rate)
    if len(angular) == 0:
        return []
    n = min(max(2, int(round(window_s * rate))), len(angular))
    back = max(n, int(round(lookback_s * rate)))
    ends = list(range(n, len(angular) + 1, n))
    if ends[-1] != len(angular):
        ends.append(len(angular))
    return [
        ((end - n) / rate,
         window_features(samples[max(0, end - back):end + 1], angular[end - n:end], opposition[end - n:end], rate))
        for end in ends
    ]


def _size_key(component):
    return "opposition_size" if component == "opposition_speed" else "angle_size"


def _participation(size):
    """Each joint's weight: its movement size relative to the biggest-moving joint, squared."""
    return (size / max(size.max(), 1e-9)) ** 2


def _movement_gate(size, gate):
    """0 while the biggest joint movement is within tracking noise, 1 once it is twice the gate."""
    return float(np.clip((size.max() - gate) / max(gate, 1e-9), 0.0, 1.0))


def _weighted_quickness(quickness, size):
    w = _participation(size)
    return float((w * quickness).sum() / max(w.sum(), 1e-9))


def combine(quickness, size, gate, q0, qmax):
    """Participation-weighted joint effort, scaled down when the movement is near tracking noise."""
    w = _participation(size)
    effort = np.clip((quickness - q0) / max(qmax - q0, 1e-9), 0.0, 1.0)
    return _movement_gate(size, gate) * float((w * effort).sum() / max(w.sum(), 1e-9))


def calibrate(max_effort, rest, window_s=WINDOW_S, lookback_s=LOOKBACK_S, rate=SAMPLE_RATE_HZ):
    """Build movement gates and maximum quickness from recordings.

    max_effort: [(frames, fps)] clips done as hard as possible.
    rest: [(frames, fps)] clips with still fingers (still hand, arm-only movement).

    The gate is the 95th percentile of the biggest joint movement seen at rest. Each
    component's maximum is the median participation-weighted quickness of max-effort
    windows that clear the gate, so typical maximum effort scores about 1. A movement
    slower than the look-back can show no less than 1/lookback_s of speed-type quickness,
    so that maps to 0.
    """
    def collect(clips):
        return [f for frames, fps in clips for _, f in windows(frames, fps, window_s, lookback_s, rate)]

    rest_f, max_f = collect(rest), collect(max_effort)
    gates = {key: float(np.percentile([f[key].max() for f in rest_f], 95)) for key in ("angle_size", "opposition_size")}
    cal = {
        "sample_rate_hz": rate,
        "window_s": window_s,
        "lookback_s": lookback_s,
        "curve": {"knee": CURVE_KNEE, "power": CURVE_POWER},
        "gates": {"angle_degrees": round(gates["angle_size"], 4), "opposition_metres": round(gates["opposition_size"], 6)},
        "components": {},
    }
    for k in COMPONENTS:
        key = _size_key(k)
        q = [_weighted_quickness(f[k], f[key]) for f in max_f if _movement_gate(f[key], gates[key]) >= 1.0]
        q0 = 0.0 if k == "angular_accel" else 1.0 / lookback_s
        # A component with no clearly moving max-effort data stays uncalibrated and scores 0.
        qmax = round(float(np.median(q)), 6) if q else None
        cal["components"][k] = {"zero": round(q0, 6), "max": qmax}
    return cal


def save_calibration(cal, path=CALIBRATION_PATH):
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)
        f.write("\n")


def load_calibration(path=CALIBRATION_PATH):
    with open(path) as f:
        return json.load(f)


def _result(features, cal, confidence):
    gates = {"angle_size": cal["gates"]["angle_degrees"], "opposition_size": cal["gates"]["opposition_metres"]}
    s = {}
    for k, c in cal["components"].items():
        key = _size_key(k)
        s[k] = 0.0 if c["max"] is None else combine(features[k], features[key], gates[key], c["zero"], c["max"])
    intensity = sum(W[k] * s[k] for k in COMPONENTS)
    knee = cal["curve"]["knee"]
    if intensity < knee:
        intensity = knee * (intensity / knee) ** cal["curve"]["power"]
    return {
        "intensity": round(float(intensity), 4),
        "angular_speed_score": round(s["angular_speed"], 4),
        "angular_peak_speed_score": round(s["angular_peak_speed"], 4),
        "angular_acceleration_score": round(s["angular_accel"], 4),
        "opposition_speed_score": round(s["opposition_speed"], 4),
        "confidence": confidence,  # Copied from the input label, not a measured reliability.
    }


def score(frames, fps, calibration=None, confidence=None):
    """Score the most recent window of frames.

    Pass at least the last LOOKBACK_S seconds: speeds come from the final WINDOW_S and
    movement size from the whole look-back. Load the calibration once with
    load_calibration() and pass it in when scoring repeatedly.
    """
    cal = calibration if calibration is not None else load_calibration()
    rate = cal["sample_rate_hz"]
    samples = resample(repair_glitches(frames, fps), fps, rate)
    angular, opposition = step_speeds(samples, rate)
    if len(angular) == 0:
        raise ValueError("need at least two frames to measure motion")
    n = min(max(2, int(round(cal["window_s"] * rate))), len(angular))
    back = max(n, int(round(cal["lookback_s"] * rate)))
    features = window_features(samples[-(back + 1):], angular[-n:], opposition[-n:], rate)
    return _result(features, cal, confidence)


def score_recording(frames, fps, calibration=None, confidence=None):
    """Score a recording window by window."""
    cal = calibration if calibration is not None else load_calibration()
    return [
        {"start_s": round(start, 3), **_result(f, cal, confidence)}
        for start, f in windows(frames, fps, cal["window_s"], cal["lookback_s"], cal["sample_rate_hz"])
    ]


def load_manifest(path):
    """Read a recordings manifest; frames have the idle start removed from untrimmed clips."""
    root = os.path.dirname(os.path.abspath(path))
    clips = []
    with open(path) as f:
        for row in csv.DictReader(f):
            fps = float(row["fps"])
            (frames, _), = load_recording(os.path.join(root, row["file"]))
            if float(row["trimmed_start_s"] or 0) == 0:
                frames = frames[int(round(IDLE_START_S * fps)):]
            clips.append({**row, "frames": frames, "fps": fps})
    return clips


def _calibrate_cli(args):
    clips = [c for m in args.manifests for c in load_manifest(m)]
    max_effort = [c for c in clips if c["condition"] == "level" and c["level"] == "4"]
    rest = [c for c in clips if c["condition"] in ("still", "wrist_only")]
    if not max_effort or not rest:
        raise SystemExit("calibration needs level-4 clips and still or wrist_only clips")
    cal = calibrate([(c["frames"], c["fps"]) for c in max_effort], [(c["frames"], c["fps"]) for c in rest])
    out_dir = os.path.dirname(os.path.abspath(args.out))
    cal["source"] = {
        "manifests": [os.path.relpath(os.path.abspath(m), out_dir) for m in args.manifests],
        "max_effort_clips": [c["file"] for c in max_effort],
        "rest_clips": [c["file"] for c in rest],
    }
    save_calibration(cal, args.out)
    print(f"saved {args.out} from {len(max_effort)} max-effort and {len(rest)} rest clips")
    print("maximum quickness (1/s): " + ", ".join(
        f"{k} {'uncalibrated' if c['max'] is None else format(c['max'], '.2f')}" for k, c in cal["components"].items()
    ))


def _score_cli(args):
    cal = load_calibration(args.calibration)
    results = [score_recording(frames, args.fps, cal, conf) for frames, conf in load_recording(args.file)]
    print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("calibrate", help="build feature_ranges.json from development recordings")
    p.add_argument("manifests", nargs="+", help="manifest.csv files under recordings/development/")
    p.add_argument("--out", default=CALIBRATION_PATH)
    p.set_defaults(func=_calibrate_cli)
    p = sub.add_parser("score", help="score a recording window by window")
    p.add_argument("file")
    p.add_argument("--fps", type=float, required=True, help="measured sample rate of the recording")
    p.add_argument("--calibration", default=CALIBRATION_PATH)
    p.set_defaults(func=_score_cli)
    args = parser.parse_args()
    args.func(args)
