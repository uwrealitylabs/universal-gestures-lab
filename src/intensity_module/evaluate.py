"""
evaluate.py

Score held-out recordings against the intensity targets:
  - every gesture averages at least 0.85 at level 4 and at most 0.15 at level 1
  - a window from each level beats a window from the level below at least 90% of the time
  - still and wrist-only clips average at most 0.05

Every clip must produce finite scores, every gesture with level clips needs levels 1-4,
and the set needs at least one still and one wrist-only clip, or the run fails.

Usage:
  python evaluate.py recordings/testing/bryan/manifest.csv
  python evaluate.py recordings/testing/bryan/manifest.csv --input-hz 15   # simulate the dynamic recorder

Only pass manifests under recordings/testing/; development recordings built the calibration.
"""

import argparse
import sys

import numpy as np

from intensity_scorer import CALIBRATION_PATH, load_calibration, load_manifest, score_recording

LEVEL4_MIN = 0.85
LEVEL1_MAX = 0.15
ADJACENT_WIN_MIN = 0.9
REST_MAX = 0.05
LEVELS = ("1", "2", "3", "4")
REST_CONDITIONS = ("still", "wrist_only")


def downsample(frames, fps, hz):
    """Point-sample frames at `hz`, as the dynamic recorder's snapshots do."""
    idx = np.round(np.arange(0, len(frames) / fps, 1.0 / hz) * fps).astype(int)
    return frames[idx[idx < len(frames)]]


def win_rate(higher, lower):
    """How often a window from `higher` outscores one from `lower`; ties count half."""
    h, l = np.asarray(higher)[:, None], np.asarray(lower)[None, :]
    return float(np.mean(h > l) + 0.5 * np.mean(h == l))


def summarize(clips):
    """Per gesture: window scores by level, and the win rate of each level over the one below."""
    out = {}
    for g in sorted({c["gesture"] for c in clips if c["condition"] == "level"}):
        by_level = {
            lv: sum((list(c["intensity"]) for c in clips if c["gesture"] == g and c["level"] == lv), [])
            for lv in LEVELS
        }
        wins = [
            win_rate(by_level[hi], by_level[lo]) if by_level[hi] and by_level[lo] else None
            for lo, hi in zip(LEVELS, LEVELS[1:])
        ]
        out[g] = (by_level, wins)
    return out


def check(clips):
    """Return the list of failed targets for clips that carry their window scores in "intensity"."""
    failures = []
    for c in clips:
        scores = np.asarray(c["intensity"], dtype=np.float64)
        if scores.size == 0:
            failures.append(f"{c['file']} produced no scores")
        elif not np.all(np.isfinite(scores)):
            failures.append(f"{c['file']} produced non-finite scores")
    if failures:
        return failures

    summary = summarize(clips)
    if not summary:
        failures.append("no level clips")
    for g, (by_level, wins) in summary.items():
        missing = [lv for lv in LEVELS if not by_level[lv]]
        if missing:
            failures.append(f"{g} has no level {', '.join(missing)} clips")
        if by_level["4"] and np.mean(by_level["4"]) < LEVEL4_MIN:
            failures.append(f"{g} level 4 averages {np.mean(by_level['4']):.2f} (target >= {LEVEL4_MIN})")
        if by_level["1"] and np.mean(by_level["1"]) > LEVEL1_MAX:
            failures.append(f"{g} level 1 averages {np.mean(by_level['1']):.2f} (target <= {LEVEL1_MAX})")
        for lo, w in zip(LEVELS, wins):
            if w is not None and w < ADJACENT_WIN_MIN:
                failures.append(f"{g} level {int(lo) + 1} beats level {lo} {w:.2f} of the time (target >= {ADJACENT_WIN_MIN})")

    for cond in REST_CONDITIONS:
        if not any(c["condition"] == cond for c in clips):
            failures.append(f"no {cond} clips")
    for c in clips:
        if c["condition"] in REST_CONDITIONS and np.mean(c["intensity"]) > REST_MAX:
            failures.append(f"{c['file']} averages {np.mean(c['intensity']):.2f} (target <= {REST_MAX})")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--calibration", default=CALIBRATION_PATH)
    parser.add_argument("--input-hz", type=float, help="resample clips to this rate before scoring")
    args = parser.parse_args()

    cal = load_calibration(args.calibration)
    clips = [c for m in args.manifests for c in load_manifest(m)]
    if not clips:
        sys.exit(f"no clips in {', '.join(args.manifests)}")
    for c in clips:
        frames, fps = c["frames"], c["fps"]
        if args.input_hz:
            frames, fps = downsample(frames, fps, args.input_hz), args.input_hz
        c["windows"] = score_recording(frames, fps, cal)
        c["intensity"] = [w["intensity"] for w in c["windows"]]

    rate = f"{args.input_hz:g} Hz input" if args.input_hz else "recorded rate"
    print(f"{len(clips)} clips, {rate}, calibration {args.calibration}\n")

    print(f"{'gesture':14s} {'L1':>5s} {'L2':>5s} {'L3':>5s} {'L4':>5s}   window win rate L1<L2 L2<L3 L3<L4")
    for g, (by_level, wins) in summarize(clips).items():
        means = [f"{np.mean(by_level[lv]):5.2f}" if by_level[lv] else "    -" for lv in LEVELS]
        print(f"{g:14s} {' '.join(means)}   " + "  ".join("    -" if w is None else f"{w:5.2f}" for w in wins))

    print()
    for c in clips:
        if c["condition"] not in ("level", ""):
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
