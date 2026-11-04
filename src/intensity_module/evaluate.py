"""
evaluate.py

Score held-out recordings against the intensity targets. The same targets apply to both kinds
of recording:
  - ongoing motion (condition "level"): scored every 0.5 s window, as score_recording() does
  - rest -> gesture (condition "transition"): scored once per move into the gesture, when the
    pose is reached, as an app would when its classifier recognises the gesture

Targets, per gesture and kind:
  - level 4 averages at least 0.85 and level 1 at most 0.15
  - a score from each level beats a score from the level below at least 90% of the time
and still and wrist-only clips average at most 0.05.

Every clip must produce finite scores, every gesture needs levels 1-4, and the set needs at
least one level clip, one still and one wrist-only clip, or the run fails. Clips listed in more
than one manifest are scored once.

Usage:
  python evaluate.py recordings/testing/bryan/manifest.csv recordings/transitions/testing/bryan/manifest.csv
  python evaluate.py <same manifests> --input-hz 15   # 15 Hz input, as from the dynamic recorder

Only pass manifests under testing/ folders; development recordings built the calibration.
"""

import argparse
import sys

import numpy as np

from intensity_scorer import CALIBRATION_PATH, load_calibration, load_manifests, score_recording, score_transitions

LEVEL4_MIN = 0.85
LEVEL1_MAX = 0.15
ADJACENT_WIN_MIN = 0.9
REST_MAX = 0.05
LEVELS = ("1", "2", "3", "4")
REST_CONDITIONS = ("still", "wrist_only")
KINDS = {"level": "ongoing motion", "transition": "rest -> gesture"}


def downsample(frames, fps, hz):
    """Point-sample frames at `hz`, as the dynamic recorder's snapshots do."""
    idx = np.round(np.arange(0, len(frames) / fps, 1.0 / hz) * fps).astype(int)
    return frames[idx[idx < len(frames)]]


def win_rate(higher, lower):
    """How often a score from `higher` beats one from `lower`; ties count half."""
    h, l = np.asarray(higher)[:, None], np.asarray(lower)[None, :]
    return float(np.mean(h > l) + 0.5 * np.mean(h == l))


def summarize(clips, condition="level"):
    """Per gesture: scores by level, and the win rate of each level over the one below."""
    out = {}
    for g in sorted({c["gesture"] for c in clips if c["condition"] == condition}):
        by_level = {
            lv: sum((list(c["intensity"]) for c in clips
                     if c["condition"] == condition and c["gesture"] == g and c["level"] == lv), [])
            for lv in LEVELS
        }
        wins = [
            win_rate(by_level[hi], by_level[lo]) if by_level[hi] and by_level[lo] else None
            for lo, hi in zip(LEVELS, LEVELS[1:])
        ]
        out[g] = (by_level, wins)
    return out


def check(clips):
    """Return the list of failed targets for clips that carry their scores in "intensity"."""
    failures = []
    for c in clips:
        scores = np.asarray(c["intensity"], dtype=np.float64)
        if scores.size == 0:
            failures.append(f"{c['file']} produced no scores")
        elif not np.all(np.isfinite(scores)):
            failures.append(f"{c['file']} produced non-finite scores")
    if failures:
        return failures

    if not summarize(clips, "level"):
        failures.append("no level clips")
    for condition in KINDS:
        tag = "" if condition == "level" else f" ({KINDS[condition]})"
        for g, (by_level, wins) in summarize(clips, condition).items():
            missing = [lv for lv in LEVELS if not by_level[lv]]
            if missing:
                failures.append(f"{g}{tag} has no level {', '.join(missing)} clips")
            if by_level["4"] and np.mean(by_level["4"]) < LEVEL4_MIN:
                failures.append(f"{g}{tag} level 4 averages {np.mean(by_level['4']):.3f} (target >= {LEVEL4_MIN})")
            if by_level["1"] and np.mean(by_level["1"]) > LEVEL1_MAX:
                failures.append(f"{g}{tag} level 1 averages {np.mean(by_level['1']):.3f} (target <= {LEVEL1_MAX})")
            for lo, w in zip(LEVELS, wins):
                if w is not None and w < ADJACENT_WIN_MIN:
                    failures.append(f"{g}{tag} level {int(lo) + 1} beats level {lo} {w:.2f} of the time (target >= {ADJACENT_WIN_MIN})")

    for cond in REST_CONDITIONS:
        if not any(c["condition"] == cond for c in clips):
            failures.append(f"no {cond} clips")
    for c in clips:
        if c["condition"] in REST_CONDITIONS and np.mean(c["intensity"]) > REST_MAX:
            failures.append(f"{c['file']} averages {np.mean(c['intensity']):.2f} (target <= {REST_MAX})")
    return failures


def score_clips(clips, cal, input_hz=None):
    """Attach each clip's scores: per window, or per move into the gesture for transition clips."""
    for c in clips:
        frames, fps = c["frames"], c["fps"]
        if input_hz:
            frames, fps = downsample(frames, fps, input_hz), input_hz
        if c["condition"] == "transition":
            c["scores"] = score_transitions(frames, fps, cal, gesture=c["gesture"])
        else:
            c["scores"] = score_recording(frames, fps, cal, gesture=c["gesture"])
        c["intensity"] = [s["intensity"] for s in c["scores"]]
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--calibration", default=CALIBRATION_PATH)
    parser.add_argument("--input-hz", type=float, help="resample clips to this rate before scoring")
    args = parser.parse_args()

    cal = load_calibration(args.calibration)
    clips = load_manifests(args.manifests)
    if not clips:
        sys.exit(f"no clips in {', '.join(args.manifests)}")
    score_clips(clips, cal, args.input_hz)

    rate = f"{args.input_hz:g} Hz input" if args.input_hz else "recorded rate"
    print(f"{len(clips)} clips, {rate}, calibration {args.calibration}")

    for condition, title in KINDS.items():
        summary = summarize(clips, condition)
        if not summary:
            continue
        unit = "window" if condition == "level" else "move"
        print(f"\n{title} (scored per {unit})")
        print(f"{'gesture':14s} {'L1':>5s} {'L2':>5s} {'L3':>5s} {'L4':>5s}   {unit} win rate L1<L2 L2<L3 L3<L4")
        for g, (by_level, wins) in summary.items():
            means = [f"{np.mean(by_level[lv]):5.2f}" if by_level[lv] else "    -" for lv in LEVELS]
            print(f"{g:14s} {' '.join(means)}   " + "  ".join("    -" if w is None else f"{w:5.2f}" for w in wins))

    print()
    for c in clips:
        if c["condition"] not in ("level", "transition", ""):
            mean = f"{np.mean(c['intensity']):.2f}" if c["intensity"] else "   -"
            print(f"{c['condition']:12s} {c['file']:70s} {mean}")

    failures = check(clips)
    print()
    if failures:
        print(f"FAIL ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: all targets met")


if __name__ == "__main__":
    main()
