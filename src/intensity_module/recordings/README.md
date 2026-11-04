# Intensity recordings

## Recording

Use the Quest 3S data collection scene (same recorder as the siamese data). Each clip is 10 s. Repeat the
movement back to back for the whole clip, except in the single-movement conditions.

| Condition | What to perform |
|---|---|
| `level` 1 | Gentle, small and slow (40 bpm metronome) |
| `level` 2 | Normal (80 bpm) |
| `level` 3 | Fast and sharp (120 bpm) |
| `level` 4 | As hard and fast as possible |
| `small` | Reduced-size movement |
| `still` | Hand held still |
| `wrist_only` | Gesture pose held, fingers still, arm moving |
| `single_quick` | One sharp movement, then a 1.5-2 s pause, repeated |
| `single_slow` | One slow movement (about 1 s), then a 2 s pause, repeated |
| `transition` 1-4 | From a relaxed hand, form the gesture in about 1 s / 0.5 s / 0.25 s / as fast as possible, hold about 1 s, relax, repeated |

Copy clips off the headset from `/sdcard/Android/data/com.bryanwei.universalgestures/files/JsonData/` into:

```
recordings/continuous/<batch>/<person>/<gesture>/<gesture>_level<1-4>_take<n>_<headset timestamp>.json
recordings/continuous/<batch>/<person>/controls/    still and wrist-only clips
recordings/transitions/<batch>/<person>/<gesture>/    transition clips
```

Other conditions use their name in place of `level<n>`, e.g. `fist_small_take1_<timestamp>.json`. Put
calibration clips in `development/` and evaluation clips in `testing/`; never calibrate on `testing/`.

Add one row per clip to the folder's `manifest.csv`, following the existing rows. The scorer reads `file`,
`gesture`, `condition`, `level`, `fps` (frames / clip length) and `trimmed_start_s` (leave at 0 unless you cut
the start; the first 0.5 s of idle hand is then dropped automatically).

## Running

From `src/intensity_module/`:

```bash
# Build feature_ranges.json from the development level-4, transition, still and wrist-only clips
python intensity_scorer.py calibrate recordings/continuous/development/bryan/manifest.csv recordings/transitions/development/bryan/manifest.csv

# Check the testing batch against the targets (add --input-hz 15 to simulate the dynamic recorder)
python evaluate.py recordings/continuous/testing/bryan/manifest.csv recordings/transitions/testing/bryan/manifest.csv

# Score one clip window by window
python intensity_scorer.py score recordings/continuous/testing/bryan/pinch/pinch_level4_take1_2026-11-03_19-57-56.json --fps 72.1

# Unit tests
python -m unittest test_scorer test_golden_scores
```
