# Intensity recordings

Clips for checking that `intensity_scorer.py` rises with how hard a gesture is performed.

## Format

Recorded with the Quest 3S data collection scene (same recorder as the siamese data).
Each file is a JSON list of frames, `{"confidence": 1, "handData": [17 floats]}`, in the
feature order documented in `intensity_scorer.py`. Clips are 10 s at about 72 fps
(~720 frames); the measured rate is stored per clip in the manifest's `fps` column.

## Batches

The recordings are split into two batches that must stay separate:

| Folder | Used for | Contents |
|---|---|---|
| `development/` | Building and tuning the calibration (`feature_ranges.json`) | 2026-11-02 session: finger gun and fist levels 1-4 (2 takes), small variants, still, wrist-only. 2026-11-03: finger spread at max effort. |
| `testing/` | Evaluation only, never used for calibration or tuning | 2026-11-03 session: finger gun, fist, pinch and peace sign levels 1-4 (1 take each), still, wrist-only. Pinch and peace sign were never calibrated on; pinch was used to diagnose the earlier speed-based design. 2026-11-03-final session, scored once with the finished scorer: beckon levels 1-4 (unseen gesture), single quick and single slow finger-gun fires, vigorous wrist-only. |

```
python intensity_scorer.py calibrate recordings/development/bryan/manifest.csv
python evaluate.py recordings/testing/bryan/manifest.csv
```

In development, level-4 clips set the maximum quickness and still and wrist-only clips set the
movement gate. The other development clips were used to choose the scoring design. The testing
results also led to one design fix (weighting joints by how much they take part in the
movement), so a fresh session is the cleanest final check.

## Layout and naming

```
recordings/<batch>/<person>/
  finger_gun/     finger_gun_level<1-4>_take<n>_<timestamp>.json, finger_gun_small_take<n>_...
  fist/           fist_level<1-4>_take<n>_<timestamp>.json, fist_small_take<n>_...
  finger_spread/  finger_spread_level4_take<n>_<timestamp>.json   (development only)
  pinch/          pinch_level<1-4>_take<n>_<timestamp>.json       (testing only)
  peace_sign/     peace_sign_level<1-4>_take<n>_<timestamp>.json  (testing only)
  beckon/         beckon_level<1-4>_take<n>_<timestamp>.json      (testing only)
  controls/       clips that should score near zero (still hand, wrist-only)
  manifest.csv    one row per clip: source file, session, condition, level, trimming,
                  measured rate and tempo, notes
```

The timestamp is the original headset file name, so each clip can be traced back.

## Conditions

| Condition (manifest `condition`, `level`) | What was performed |
|---|---|
| `level`, 1 | Very gentle, small and slow (40 bpm metronome) |
| `level`, 2 | Normal (80 bpm) |
| `level`, 3 | Fast and sharp (120 bpm) |
| `level`, 4 | As hard and fast as possible |
| `small` | Reduced-size movement, to separate movement size from speed |
| `wrist_only` | Gesture pose held, fingers still, arm moving |
| `still` | Hand held still |
| `single_quick` | One sharp movement at a time, with pauses of 1.5-2 s between them |
| `single_slow` | One slow movement (about 1 s) at a time, with pauses of about 2 s between them |

Apart from the single-movement clips, the movement is repeated back to back for the whole
clip. `reps_per_min` in the manifest is measured from the data, not the intended tempo.

## Cleaning

Idle frames before the first movement are dropped by removing whole frame lines, so the
remaining frames are unchanged. `trimmed_start_s` in the manifest records how much was cut.
Uncut clips still start with roughly 0.5 s of idle hand; `load_manifest` in
`intensity_scorer.py` drops it when calibrating and evaluating.
