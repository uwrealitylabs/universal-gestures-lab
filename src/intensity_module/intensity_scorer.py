"""
intensity_scorer.py

Intensity score in [0,1] for hand motion, on the same standard for every gesture:
0 is a still hand (or a slow movement) and 1 is maximum effort. It measures how
quickly the fingers move, not how far: each joint's speed is divided by how far that
joint moved over the last LOOKBACK_S seconds, so a small quick pinch and a large quick
finger gun score alike. The recordings do not contain wrist translation, so arm
movement alone scores near 0.

One formula scores every movement, whether the fingers keep moving or snap once from a
relaxed hand into a gesture:
  intensity = curve(0.3 * sharpness + 0.7 * pace)
  sharpness = angular_peak_speed (the fastest ~0.13 s)
  pace      = the higher of angular_speed (how quickly the fingers keep moving) and duration
              (how quickly a move from a resting hand into a held pose was completed, timed
              from 10% to 90% of the way at the input rate)
Duration is 0 unless such a move just ended, so continuous movement is scored by its speed.

Joints are averaged by how much they take part in the movement, so incidental twitches
of other fingers count little. Calibration is shared by every gesture: a movement gate
(how far joints appear to move when the hand is still), one maximum quickness per
component (from maximum-effort clips) and the maximum-effort time of a snap from rest.
A new gesture needs no recordings of its own. Combined scores below CURVE_KNEE are
curved down, which keeps gentle movement low.

Live use: keep a rolling buffer of the last TRANSITION_BUFFER_S (2 s) of frames and call
score() on it whenever a score is needed; for a recognised gesture, read it within about
0.4 s of the hand reaching the pose. Each result also logs the component scores and raw
motion features (duration, path length, average and peak speed, peak acceleration, size).

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
distance in metres. Timing is never assumed: pass the measured frame rate, or per-frame
timestamps (read from a recording's "timestamp" / "timestamps" fields when it has them).

Usage:
  python intensity_scorer.py calibrate recordings/development/bryan/manifest.csv recordings/transitions/development/bryan/manifest.csv
  python intensity_scorer.py score recordings/transitions/testing/bryan/pinch/<clip>.json --fps 72.1 --gesture pinch
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
# Motion faster than this (Hz) is filtered out before resampling: deliberate finger movement
# tops out around 5 Hz, faster wiggles are tremor or tracking jitter.
LOWPASS_HZ = 6.0
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
# Path length is reported too but not weighted, since it grows with movement size.
# In combine_intensity() the angular_speed weight goes to pace: the higher of angular_speed and
# duration, so repetition and a snap from rest are scored by the same formula.
W = {"angular_speed": 0.7, "angular_peak_speed": 0.3, "angular_accel": 0.0, "opposition_speed": 0.0}
assert abs(sum(W.values()) - 1.0) < 1e-9
# Scores below CURVE_KNEE are lowered to knee * (score / knee) ** CURVE_POWER; scores at or
# above it are unchanged. It keeps 0 and the order of scores, leaves everything from a slow but
# deliberate move up as it is, and pushes the faintest movement (tracking wobble, gentle drift)
# towards 0.
CURVE_KNEE = 0.15
CURVE_POWER = 2.0
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_ranges.json")

# duration: how quickly the latest finger movement was completed, from 10% to 90% of the way
# (its rise time). It is measured at the input rate, because a fast snap takes 40-70 ms and
# 15 Hz cannot resolve it. A move ends where the hand slows to MOVE_END_FRACTION of its peak
# speed: at a turning point of a repeated motion, or when it settles into a pose. The rise
# time does not depend on how far or which fingers moved. It maps to [0,1] on a log scale:
# ZERO_RISE_S or slower is 0 and the calibrated maximum-effort rise time is 1, so equal
# ratios of time are equal steps. ZERO_RISE_S is the rise time of a smooth move of about 1.7 s;
# a move at the protocol's natural pace (about 0.5 s) then scores about 0.35.
DURATION_BUFFER_S = 2.0      # frames looked at before a move's end
ZERO_RISE_S = 1.0
# The same measurement runs at every input rate. From 15 Hz input (the dynamic recorder) a snap
# spans only one or two frames, so its time is interpolated and reads a little slower.
END_POSE_S = 0.05            # the pose a move ends in is the mean of its last END_POSE_S
MOVE_END_FRACTION = 0.2
MOVE_END_HOLD_S = 0.05       # ... and stays that slow this long (a gesture can move in phases)
MOVE_SETTLE_FRACTION = 0.05  # it has settled once under this fraction of the peak speed
HAND_SPEED_SMOOTH_S = 0.05
MIN_MOVE_SPEED = 100.0       # summed joint speed (deg/s) a move must reach
# Only a snap counts for duration: a move from a resting hand into a held pose. Both are
# measured along the move, as distance from the pose it ends in, relative to the move's size:
#   rested  for RESTED_S before the move began, that distance varied by less than RESTED_FRACTION
#           (at the very start of the frames, at least RESTED_MIN_S of visible rest is enough)
#   held    for HELD_S after it ended (or what has arrived, at least HELD_MIN_S), the hand
#           stayed within HELD_FRACTION of the pose
# In the recordings, snaps from a relaxed hand followed 0.8 s of rest or more, while fast
# repetition rests far less, so it is left to angular_speed. A flick goes straight back, so it
# is not held.
RESTED_S = 0.8
RESTED_MIN_S = 0.5
RESTED_FRACTION = 0.25
HELD_S = 0.15
HELD_MIN_S = 0.05
HELD_FRACTION = 0.3
# Walking back from a move's end, the move began where the distance from its end pose stops
# growing: it grows by less than PLATEAU_GROWTH over the PLATEAU_S before (the hand was steady
# or turning there), or falls below half its largest value (the hand was in this pose earlier).
# Distances under NOISE_FLOOR_DEGREES are tracking noise and never end the search.
PLATEAU_S = 0.25
PLATEAU_GROWTH = 0.1
NOISE_FLOOR_DEGREES = 5.0

# Recording analysis for rest -> gesture clips (calibration and evaluation): the hand is still
# while no joint changes by STILL_DEGREES within STILL_WINDOW_S (about 30 deg/s, so a slow move
# is not still). A pose counts once held for HOLD_S, which skips the brief overshoot on the way
# back to rest, and is scored SCORE_DELAY_S after it is reached, as when a classifier
# recognises the gesture. Clips start relaxed; their first RELAXED_S is the relaxed pose.
STILL_WINDOW_S = 0.2
STILL_DEGREES = 6.0
HOLD_S = 0.3
SCORE_DELAY_S = 0.35
RELAXED_S = 0.3
TRANSITION_BUFFER_S = 2.0    # frames passed to score() when a gesture is recognised


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


def load_timestamps(path):
    """Per-frame times in seconds for each entry of load_recording(), or None where the file has none.

    Read from a "timestamp" field on each static frame, or a "timestamps" list on each dynamic
    sequence, so recordings that save their measured timing are scored with it.
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]
    if not raw:
        return []
    if "sequenceData" in raw[0]:
        return [np.asarray(e["timestamps"], dtype=np.float64) if "timestamps" in e else None for e in raw]
    if all("timestamp" in e for e in raw):
        return [np.asarray([e["timestamp"] for e in raw], dtype=np.float64)]
    return [None]


def to_uniform(frames, timestamps):
    """Resample frames recorded at the given times onto an even grid at their measured rate.

    Returns (frames, fps). Use this when the recorder saves timestamps, instead of assuming a rate.
    """
    t = np.asarray(timestamps, dtype=np.float64) - float(timestamps[0])
    if len(t) != len(frames) or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be increasing, one per frame")
    fps = 1.0 / float(np.median(np.diff(t)))
    grid = np.arange(0.0, t[-1] + 1e-9, 1.0 / fps)
    return np.stack([np.interp(grid, t, frames[:, k]) for k in range(frames.shape[1])], axis=1), fps


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


def lowpass(frames, fps, cutoff=None):
    """Remove motion faster than voluntary finger movement (tremor, tracking jitter).

    A Gaussian smoothing whose response is -3 dB at LOWPASS_HZ, run before resampling so fast
    jitter neither scores as maximum effort nor aliases at SAMPLE_RATE_HZ. Speed and movement
    size shrink together, so quickness of ordinary movement is unchanged.
    """
    cutoff = LOWPASS_HZ if cutoff is None else cutoff
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * cutoff) * fps          # in frames
    if len(frames) < 2 or sigma < 0.5:
        return frames
    r = int(np.ceil(3.0 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(np.asarray(frames, dtype=np.float64), ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, j], k, mode="valid") for j in range(padded.shape[1])], axis=1)


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
        # Raw per-joint motion over the window, logged in the output.
        "path": angular.sum(axis=0) / rate,
        "mean_speed": angular.mean(axis=0),
        "peak_step_speed": peak,
        "peak_accel": accel.max(axis=0),
    }


def rise_time(distance, fps, with_start=False):
    """10-90% time in seconds of the move into the final pose of `distance`, or None.

    `distance` is how far the hand is from the pose it ends in, frame by frame. Crossings are
    interpolated between frames, so fast moves are not rounded to whole frames. If the move
    began before the frames, the result is a lower bound. With with_start, returns
    (rise_s, start_index) where start_index is the last frame still 90% of the way out.
    """
    d = np.asarray(distance, dtype=np.float64)
    none = (None, None) if with_start else None
    if len(d) < 2:
        return none
    w = max(1, int(round(PLATEAU_S * fps)))
    peak, peak_i = d[-1], len(d) - 1
    for i in range(len(d) - 1, -1, -1):
        if d[i] > peak:
            peak, peak_i = d[i], i
        if peak < NOISE_FLOOR_DEGREES:
            continue
        if d[i] < 0.5 * peak:
            break
        if i >= w and d[i - w:i].max() <= (1 + PLATEAU_GROWTH) * peak:
            break
    if peak < NOISE_FLOOR_DEGREES:
        return none
    hi, lo = 0.9 * peak, 0.1 * peak
    # Last frame still at least 90% of the way out, then the first frame within 10% of the pose.
    i = peak_i + int(np.flatnonzero(d[peak_i:] >= hi)[-1])
    after = np.flatnonzero(d[i + 1:] <= lo)
    if after.size == 0:
        return none
    j = i + 1 + int(after[0])
    t10 = i + (d[i] - hi) / max(d[i] - d[i + 1], 1e-9)
    t90 = (j - 1) + (d[j - 1] - lo) / max(d[j - 1] - d[j], 1e-9)
    rise = max(float(t90 - t10), 0.0) / fps
    return (rise, i) if with_start else rise


def finger_moves(frames, fps):
    """[(end_s, rise_s, size_degrees, snap), ...] for each finger movement, at the input rate.

    A move runs from a speed peak to where the hand slows to MOVE_END_FRACTION of it. Its rise
    time comes from the hand's distance to the pose it ends in, joints weighted by how much they
    moved, over the DURATION_BUFFER_S before; size is its biggest single-joint change. A move is
    a snap if the hand had rested for RESTED_S before it and holds the pose for HELD_S after it
    (False when the frames do not reach that far).
    """
    a = repair_glitches(frames, fps)[:, ANGLE_IDX]
    if len(a) < 3:
        return []
    k = max(1, int(round(HAND_SPEED_SMOOTH_S * fps)))
    v = np.convolve(np.abs(np.diff(a, axis=0)).sum(axis=1) * fps, np.ones(k) / k, mode="same")
    buffer, pose = max(2, int(round(DURATION_BUFFER_S * fps))), max(1, int(round(END_POSE_S * fps)))
    end_hold = max(1, int(round(MOVE_END_HOLD_S * fps)))
    out, last_end = [], 0
    for p in range(1, len(v) - 1):
        if p <= last_end or v[p] < MIN_MOVE_SPEED or v[p] < v[p - 1] or v[p] <= v[p + 1]:
            continue
        while True:
            slow = _first_sustained(v[p:] < MOVE_END_FRACTION * v[p], end_hold)
            if slow is None:
                break                           # never slows down: still moving, or noise
            top = p + int(np.argmax(v[p:p + slow + 1]))
            if v[top] <= v[p]:
                break
            p = top
        if slow is None:
            continue
        # Then follow the slowdown to where the hand settles: its slowest point before it speeds
        # up again (a turning point) or once it is under MOVE_SETTLE_FRACTION of the peak.
        end = low = p + slow
        for i in range(end + 1, len(v)):
            if v[i] < v[low]:
                low = i
            if v[low] < MOVE_SETTLE_FRACTION * v[p] or v[i] > 1.1 * v[low] + 1e-9:
                break
        end = low
        seg = a[max(0, end + 1 - buffer):end + 1]
        change = np.abs(seg - seg[-pose:].mean(axis=0))
        size = change.max(axis=0)
        w = _participation(size)
        distance = (change * w).sum(axis=1) / max(w.sum(), 1e-9)
        rise, start = rise_time(distance, fps, with_start=True)
        snap = False
        rest_n, hold_n = max(1, int(round(RESTED_S * fps))), max(1, int(round(HELD_S * fps)))
        at_first_frame = end + 1 - buffer <= 0          # seg begins with the first frame given
        if start is not None and start < rest_n and at_first_frame:
            rest_n = start if start >= int(round(RESTED_MIN_S * fps)) else rest_n
        # Judge the hold on what has arrived since the move ended (at least HELD_MIN_S), as live.
        after = a[end:end + hold_n + 1]
        if start is not None and start >= rest_n and len(after) >= max(2, int(round(HELD_MIN_S * fps))) + 1:
            move = float(distance[start])
            before = distance[start - rest_n:start + 1]
            drift = (np.abs(after - seg[-pose:].mean(axis=0)) * w).sum(axis=1) / max(w.sum(), 1e-9)
            snap = (float(before.max() - before.min()) < RESTED_FRACTION * move
                    and float(drift.max()) < HELD_FRACTION * move)
        out.append((end / fps, rise, float(size.max()), snap))
        last_end = end
    return out


def _first_sustained(mask, k):
    """Index of the first run of at least k True values in mask, or None."""
    run = 0
    for i, value in enumerate(mask):
        run = run + 1 if value else 0
        if run >= k:
            return i - k + 1
    return None


def rise_to_score(rise, zero_rise, full_rise):
    """Log-scale map: zero_rise or slower is 0, full_rise or faster is 1."""
    if rise is None or full_rise is None:
        return 0.0
    return float(np.clip(np.log(zero_rise / max(rise, 1e-6)) / np.log(zero_rise / full_rise), 0.0, 1.0))


def windows(frames, fps, window_s=WINDOW_S, lookback_s=LOOKBACK_S, rate=SAMPLE_RATE_HZ):
    """Return [(start_s, features), ...] for consecutive windows; a short input is one window.

    When the steps do not divide evenly, a last window is aligned to the end of the input,
    overlapping the one before it, so the final movement is scored as score() would score it.
    Each window also carries the finger moves that ended within it ("moves").
    """
    samples = resample(lowpass(repair_glitches(frames, fps), fps), fps, rate)
    angular, opposition = step_speeds(samples, rate)
    if len(angular) == 0:
        return []
    n = min(max(2, int(round(window_s * rate))), len(angular))
    back = max(n, int(round(lookback_s * rate)))
    ends = list(range(n, len(angular) + 1, n))
    if ends[-1] != len(angular):
        ends.append(len(angular))
    moves = finger_moves(frames, fps)
    t_last = (len(frames) - 1) / fps
    out = []
    for end in ends:
        f = window_features(samples[max(0, end - back):end + 1], angular[end - n:end], opposition[end - n:end], rate)
        if end == len(angular):   # the final window reaches the last frame, as in score()
            f["moves"] = [m for m in moves if t_last - n / rate < m[0] <= t_last + 1e-9]
        else:
            f["moves"] = [m for m in moves if (end - n) / rate < m[0] <= end / rate]
        out.append(((end - n) / rate, f))
    return out


def find_transitions(frames, fps, gate):
    """Frame indices where a rest -> gesture recording reaches a held gesture pose.

    For calibration and evaluation of clips that start relaxed and repeat
    rest -> gesture -> rest with one gesture per clip. A move counts when the hand comes to a
    still hold in the clip's gesture pose from a hold at least `gate` degrees away (on some
    joint). Moves back to rest, and holds in some other pose, are not returned.
    """
    a = repair_glitches(frames, fps)[:, ANGLE_IDX]
    k = max(1, int(round(STILL_WINDOW_S * fps)))
    if len(a) <= k:
        return []
    change = np.abs(a[k:] - a[:-k]).max(axis=1)            # change[t] covers frames t .. t + k
    still = np.r_[change < STILL_DEGREES, np.zeros(k, dtype=bool)]
    holds, i = [], 0
    while i < len(still):
        if not still[i]:
            i += 1
            continue
        j = i
        while j < len(still) and still[j]:
            j += 1
        if j + k - i >= int(round(HOLD_S * fps)):
            holds.append((i, np.median(a[i:j + k], axis=0)))
        i = j
    relaxed = np.median(a[:max(1, int(round(RELAXED_S * fps)))], axis=0)
    far = lambda p, q: float(np.abs(p - q).max())
    in_gesture = [h for h in holds if far(h[1], relaxed) >= gate]
    if not in_gesture:
        return []
    pose = np.median([p for _, p in in_gesture], axis=0)      # the clip's gesture pose
    reached = []
    for n in range(1, len(holds)):
        start, p = holds[n]
        q = holds[n - 1][1]
        if far(p, relaxed) < gate or far(p, pose) > 0.5 * far(pose, relaxed) or far(p, q) < gate:
            continue
        # Reached when the whole hand is 90% of the way from the previous hold into this one,
        # joints weighted by how much they move between them, so a small joint still settling
        # (such as the thumb after a fist closes) does not delay it.
        w = _participation(np.abs(p - q))
        dist = (np.abs(a[:start + 1] - p) * w).sum(axis=1) / max(w.sum(), 1e-9)
        prev = holds[n - 1][0]
        within = np.flatnonzero(dist[prev:start + 1] <= 0.1 * float(dist[prev]))
        reached.append(prev + int(within[0]) if within.size else start)
    return reached


def score_transitions(frames, fps, calibration=None, confidence=None, gesture=None):
    """Score each move into a held gesture in a rest -> gesture recording, the way an app would:
    score() on the last TRANSITION_BUFFER_S of frames, SCORE_DELAY_S after the pose is reached."""
    cal = calibration if calibration is not None else load_calibration()
    buffer, delay = int(round(TRANSITION_BUFFER_S * fps)), int(round(SCORE_DELAY_S * fps))
    out = []
    for t in find_transitions(frames, fps, cal["gates"]["angle_degrees"]):
        end = min(len(frames), t + delay + 1)
        out.append({"reached_s": round(t / fps, 3), **score(frames[max(0, end - buffer):end], fps, cal, confidence, gesture)})
    return out


def _size_key(component):
    return "opposition_size" if component == "opposition_speed" else "angle_size"


def _participation(size):
    """Each joint's weight: its movement size relative to the biggest-moving joint, squared."""
    return (size / max(size.max(), 1e-9)) ** 2


def _movement_gate(size, gate):
    """0 while the biggest joint movement is within tracking noise, 1 once it is twice the gate."""
    return _gate_ramp(float(size.max()), gate)


def _gate_ramp(biggest, gate):
    return float(np.clip((biggest - gate) / max(gate, 1e-9), 0.0, 1.0))


def _weighted_mean(values, size):
    w = _participation(size)
    return float((w * values).sum() / max(w.sum(), 1e-9))


def _recent_moves(frames, fps, window_s, rate):
    """Finger moves that ended within the last scoring window of `frames`, as score() sees it.

    The window reaches the last frame itself, not just the last resampled step, so a move that
    has just finished counts.
    """
    t_last = (len(frames) - 1) / fps
    n = min(max(2, int(round(window_s * rate))), int(np.floor(t_last * rate + 1e-9)))
    return [m for m in finger_moves(frames, fps) if t_last - n / rate < m[0] <= t_last + 1e-9]


def _weighted_quickness(quickness, size):
    w = _participation(size)
    return float((w * quickness).sum() / max(w.sum(), 1e-9))


def combine(quickness, size, gate, q0, qmax):
    """Participation-weighted joint effort, scaled down when the movement is near tracking noise."""
    w = _participation(size)
    effort = np.clip((quickness - q0) / max(qmax - q0, 1e-9), 0.0, 1.0)
    return _movement_gate(size, gate) * float((w * effort).sum() / max(w.sum(), 1e-9))


def calibrate(max_effort, rest, window_s=WINDOW_S, lookback_s=LOOKBACK_S, rate=SAMPLE_RATE_HZ, transitions=()):
    """Build movement gates and maximum quickness from recordings.

    max_effort: [(frames, fps)] clips done as hard as possible.
    rest: [(frames, fps)] clips with still fingers (still hand, arm-only movement).
    transitions: [(frames, fps)] rest -> gesture clips done as fast as possible.

    The gate is the 95th percentile of the biggest joint movement seen at rest. Each
    component's maximum is the median participation-weighted quickness of max-effort
    windows that clear the gate, so typical maximum effort scores about 1. A movement
    slower than the look-back can show no less than 1/lookback_s of speed-type quickness,
    so that maps to 0. The duration maximum is the median rise time of the moves into the
    pose in the transition clips, as score() sees them when the gesture is recognised; the
    path maximum is the median path of the max-effort windows.
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
    gate = cal["gates"]["angle_degrees"]
    paths = [_weighted_mean(f["path"], f["angle_size"]) for f in max_f if _movement_gate(f["angle_size"], gate) >= 1.0]
    cal["path"] = {"max_degrees": round(float(np.median(paths)), 4) if paths else None}
    rises = []
    for frames, fps in transitions:
        buffer, delay = int(round(TRANSITION_BUFFER_S * fps)), int(round(SCORE_DELAY_S * fps))
        for t in find_transitions(frames, fps, gate):
            end = min(len(frames), t + delay + 1)
            moves = [m for m in _recent_moves(frames[max(0, end - buffer):end], fps, window_s, rate)
                     if m[1] is not None and m[3] and _gate_ramp(m[2], gate) > 0]
            if moves:
                rises.append(min(m[1] for m in moves))
    cal["duration"] = {
        "zero_rise_s": ZERO_RISE_S,
        "full_rise_s": round(float(np.median(rises)), 6) if rises else None,
        "max_effort_moves": len(rises),
    }
    return cal


def save_calibration(cal, path=CALIBRATION_PATH):
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)
        f.write("\n")


def load_calibration(path=CALIBRATION_PATH):
    with open(path) as f:
        return json.load(f)


def combine_intensity(s, curve):
    """One formula for every movement: curve(sharpness weight * sharpness + pace weight * pace).

    sharpness is angular_peak_speed; pace is the higher of angular_speed (how quickly the
    fingers keep moving) and duration (how quickly a move from a resting hand into a held pose
    was completed). Duration is 0 unless such a move just ended, so continuous movement is
    scored by its speed, and a single snap from rest by how quickly it was done. `curve` keeps
    gentle movement near 0. The weights, curve and duration settings were chosen so the
    development and testing recordings both meet the targets and scores follow measured speed
    (see recordings/README.md); the calibration itself uses development recordings only.
    """
    pace = max(s["angular_speed"], s["duration"])
    return curve(W["angular_peak_speed"] * s["angular_peak_speed"] + W["angular_speed"] * pace
                 + W["angular_accel"] * s["angular_accel"] + W["opposition_speed"] * s["opposition_speed"])


def _result(features, cal, confidence, gesture=None):
    gates = {"angle_size": cal["gates"]["angle_degrees"], "opposition_size": cal["gates"]["opposition_metres"]}
    s = {}
    for k, c in cal["components"].items():
        key = _size_key(k)
        s[k] = 0.0 if c["max"] is None else combine(features[k], features[key], gates[key], c["zero"], c["max"])
    # The quickest finger move that ended within this window, if any.
    dur = cal.get("duration", {})
    scored = [(_gate_ramp(biggest, gates["angle_size"]) * rise_to_score(rise, dur.get("zero_rise_s", ZERO_RISE_S), dur.get("full_rise_s")), rise)
              for _, rise, biggest, snap in features.get("moves", []) if snap]
    s["duration"], best_rise = max(scored, key=lambda x: (x[0], -(x[1] if x[1] is not None else 1e9)), default=(0.0, None))
    path = _weighted_mean(features["path"], features["angle_size"])
    path_max = cal.get("path", {}).get("max_degrees")
    s["path"] = 0.0 if not path_max else _movement_gate(features["angle_size"], gates["angle_size"]) * min(path / path_max, 1.0)
    knee, power = cal["curve"]["knee"], cal["curve"]["power"]
    intensity = combine_intensity(s, lambda x: x if x >= knee else knee * (x / knee) ** power)
    size = features["angle_size"]
    return {
        "gesture": gesture,  # Passed through from the caller (e.g. the classifier's label).
        "intensity": round(float(intensity), 4),
        "angular_speed_score": round(s["angular_speed"], 4),
        "angular_peak_speed_score": round(s["angular_peak_speed"], 4),
        "duration_score": round(s["duration"], 4),
        "path_score": round(s["path"], 4),
        "angular_acceleration_score": round(s["angular_accel"], 4),
        "opposition_speed_score": round(s["opposition_speed"], 4),
        "confidence": confidence,  # Copied from the input label, not a measured reliability.
        "features": {
            "move_duration_s": None if best_rise is None else round(best_rise, 4),
            "path_length_degrees": round(path, 2),
            "average_speed_deg_per_s": round(_weighted_mean(features["mean_speed"], size), 1),
            "peak_speed_deg_per_s": round(_weighted_mean(features["peak_step_speed"], size), 1),
            "peak_acceleration_deg_per_s2": round(_weighted_mean(features["peak_accel"], size), 1),
            "movement_size_degrees": round(float(size.max()), 2),
        },
    }


def _timing(frames, fps, timestamps):
    frames = np.asarray(frames, dtype=np.float64)
    if timestamps is not None:
        return to_uniform(frames, timestamps)
    if fps is None:
        raise ValueError("pass the measured frame rate (fps) or per-frame timestamps")
    return frames, float(fps)


def score(frames, fps=None, calibration=None, confidence=None, gesture=None, timestamps=None):
    """Score the most recent window of frames, as on a live stream.

    Keep a rolling buffer of the last TRANSITION_BUFFER_S seconds of hand frames and call this
    on it whenever a score is needed (every 0.5 s, or when a gesture is recognised). Speeds come
    from the final WINDOW_S, movement size from the last LOOKBACK_S, and a snap from rest into
    a held pose counts while its end is in the final window, so read it within about 0.4 s of
    the hand reaching the pose. Pass the measured frame rate, or per-frame timestamps in seconds.
    Load the calibration once with load_calibration() and pass it in when scoring repeatedly.
    """
    frames, fps = _timing(frames, fps, timestamps)
    cal = calibration if calibration is not None else load_calibration()
    rate = cal["sample_rate_hz"]
    samples = resample(lowpass(repair_glitches(frames, fps), fps), fps, rate)
    angular, opposition = step_speeds(samples, rate)
    if len(angular) == 0:
        raise ValueError("need at least two frames to measure motion")
    n = min(max(2, int(round(cal["window_s"] * rate))), len(angular))
    back = max(n, int(round(cal["lookback_s"] * rate)))
    features = window_features(samples[-(back + 1):], angular[-n:], opposition[-n:], rate)
    features["moves"] = _recent_moves(frames, fps, cal["window_s"], rate)
    return _result(features, cal, confidence, gesture)


def score_recording(frames, fps=None, calibration=None, confidence=None, gesture=None, timestamps=None):
    """Score a recording window by window (pass the measured frame rate or per-frame timestamps)."""
    frames, fps = _timing(frames, fps, timestamps)
    cal = calibration if calibration is not None else load_calibration()
    return [
        {"start_s": round(start, 3), **_result(f, cal, confidence, gesture)}
        for start, f in windows(frames, fps, cal["window_s"], cal["lookback_s"], cal["sample_rate_hz"])
    ]


def load_manifest(path):
    """Read a recordings manifest; frames have the idle start removed from untrimmed clips.

    Rest -> gesture clips (condition "transition") are used whole: they start relaxed, and
    the relaxed start is part of the movement being scored.
    """
    root = os.path.dirname(os.path.abspath(path))
    clips = []
    with open(path) as f:
        for row in csv.DictReader(f):
            fps = float(row["fps"])
            (frames, _), = load_recording(os.path.normpath(os.path.join(root, row["file"])))
            if row.get("condition") != "transition" and float(row.get("trimmed_start_s") or 0) == 0:
                frames = frames[int(round(IDLE_START_S * fps)):]
            clips.append({**row, "frames": frames, "fps": fps,
                          "path": os.path.normpath(os.path.join(root, row["file"]))})
    return clips


def load_manifests(paths):
    """Read several manifests, keeping each recording once (manifests may share control clips)."""
    seen, clips = set(), []
    for m in paths:
        for c in load_manifest(m):
            if c["path"] not in seen:
                seen.add(c["path"])
                clips.append(c)
    return clips


def _calibrate_cli(args):
    clips = load_manifests(args.manifests)
    max_effort = [c for c in clips if c["condition"] == "level" and c["level"] == "4"]
    rest = [c for c in clips if c["condition"] in ("still", "wrist_only")]
    transitions = [c for c in clips if c["condition"] == "transition" and c["level"] == "4"]
    if not max_effort or not rest:
        raise SystemExit("calibration needs level-4 clips and still or wrist_only clips")
    cal = calibrate([(c["frames"], c["fps"]) for c in max_effort], [(c["frames"], c["fps"]) for c in rest],
                    transitions=[(c["frames"], c["fps"]) for c in transitions])
    out_dir = os.path.dirname(os.path.abspath(args.out))
    cal["source"] = {
        "manifests": [os.path.relpath(os.path.abspath(m), out_dir) for m in args.manifests],
        "max_effort_clips": [c["file"] for c in max_effort],
        "rest_clips": [c["file"] for c in rest],
        "transition_max_effort_clips": [c["file"] for c in transitions],
    }
    save_calibration(cal, args.out)
    print(f"saved {args.out} from {len(max_effort)} max-effort, {len(rest)} rest and {len(transitions)} rest -> gesture clips")
    print("maximum quickness (1/s): " + ", ".join(
        f"{k} {'uncalibrated' if c['max'] is None else format(c['max'], '.2f')}" for k, c in cal["components"].items()
    ))
    d = cal["duration"]
    print(f"duration: full score at {'uncalibrated' if d['full_rise_s'] is None else format(d['full_rise_s'] * 1000, '.0f') + ' ms'} "
          f"rise time ({d['max_effort_moves']} moves), zero at {d['zero_rise_s']} s; path max {cal['path']['max_degrees']} deg")


def _score_cli(args):
    cal = load_calibration(args.calibration)
    results = []
    for (frames, conf), times in zip(load_recording(args.file), load_timestamps(args.file)):
        if times is None and args.fps is None:
            raise SystemExit(f"{args.file} has no timestamps; pass --fps with its measured frame rate")
        results.append(score_recording(frames, None if times is not None else args.fps, cal, conf, args.gesture, times))
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
    p.add_argument("--fps", type=float, help="measured frame rate, for recordings without timestamps")
    p.add_argument("--calibration", default=CALIBRATION_PATH)
    p.add_argument("--gesture", help="gesture label to copy into the output")
    p.set_defaults(func=_score_cli)
    args = parser.parse_args()
    args.func(args)
