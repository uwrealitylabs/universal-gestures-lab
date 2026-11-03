"""
evaluate.py

Score held-out recordings against the intensity targets:
  - every gesture averages at least 0.85 at level 4 and at most 0.15 at level 1
  - a window from each level beats a window from the level below at least 90% of the time
  - still and wrist-only clips average at most 0.05

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


def downsample(frames, fps, hz):
    """Point-sample frames at `hz`, as the dynamic recorder's snapshots do."""
    idx = np.round(np.arange(0, len(frames) / fps, 1.0 / hz) * fps).astype(int)
    return frames[idx[idx < len(frames)]]


def win_rate(higher, lower):
    """How often a window from `higher` outscores one from `lower`; ties count half."""
    h, l = np.asarray(higher)[:, None], np.asarray(lower)[None, :]
    return float(np.mean(h > l) + 0.5 * np.mean(h == l))


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
    failures = []

    print(f"{'gesture':14s} {'L1':>5s} {'L2':>5s} {'L3':>5s} {'L4':>5s}   window win rate L1<L2 L2<L3 L3<L4")
    gestures = sorted({c["gesture"] for c in clips if c["condition"] == "level"})
    for g in gestures:
        by_level = {
            lv: sum((c["intensity"] for c in clips if c["gesture"] == g and c["level"] == str(lv)), [])
            for lv in (1, 2, 3, 4)
        }
        means = [f"{np.mean(by_level[lv]):5.2f}" if by_level[lv] else "    -" for lv in (1, 2, 3, 4)]
        wins = [
            win_rate(by_level[lv + 1], by_level[lv]) if by_level[lv] and by_level[lv + 1] else None
            for lv in (1, 2, 3)
        ]
        print(f"{g:14s} {' '.join(means)}   " + "  ".join("    -" if w is None else f"{w:5.2f}" for w in wins))
        if by_level[4] and np.mean(by_level[4]) < LEVEL4_MIN:
            failures.append(f"{g} level 4 averages {np.mean(by_level[4]):.2f} (target >= {LEVEL4_MIN})")
        if by_level[1] and np.mean(by_level[1]) > LEVEL1_MAX:
            failures.append(f"{g} level 1 averages {np.mean(by_level[1]):.2f} (target <= {LEVEL1_MAX})")
        for lv, w in zip((1, 2, 3), wins):
            if w is not None and w < ADJACENT_WIN_MIN:
                failures.append(f"{g} level {lv + 1} beats level {lv} {w:.2f} of the time (target >= {ADJACENT_WIN_MIN})")

    print()
    for c in clips:
        if c["condition"] in ("still", "wrist_only", "small"):
            mean = np.mean(c["intensity"])
            print(f"{c['condition']:10s} {c['file']:60s} {mean:.2f}")
            if c["condition"] != "small" and mean > REST_MAX:
                failures.append(f"{c['file']} averages {mean:.2f} (target <= {REST_MAX})")

    print()
    if failures:
        print(f"FAIL ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: all targets met")


if __name__ == "__main__":
    main()
