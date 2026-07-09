# Siamese Triplet-Loss Few-Shot Model — Status and Next Steps

## What changed

`fewshotmodel.py` had a preliminary Siamese few-shot model with a few known issues:

- `TripletLoss` had two `__init__`/`forward` definitions; the second silently overwrote the first, so training was actually running contrastive loss under a triplet-loss name.
- The dataset built `(x1, x2, label)` pairs instead of `(anchor, positive, negative)` triplets.
- Positive pairs were built from sequential indices (`positives[i]`, `positives[i+1]`) instead of random sampling, which limited training diversity.
- Pair generation used `min(len(positives), len(negatives))`, discarding excess samples from whichever class was larger.
- There was no validation split, so there was no way to tune hyperparameters or catch overfitting without touching the test set.

This push fixes all five, and splits the work into two files:

- **`src/fewshottripletloss.py`** — static, single-frame gestures (thumbs up, finger gun, peace sign, closed fist), matching the data already in `src/data/`. Architecture: `Linear` + `BatchNorm` encoder.
- **`src/fewshottripletloss_dynamic.py`** — dynamic, motion-sequence gestures (fire finger gun, squeeze-palm-up), matching `src/data(Dynamic)/`. Architecture: `LSTM` encoder, since a single frame can't capture motion.

Both use: a real triplet-margin loss, random (not sequential) triplet sampling across all classes, no discarded data, a stratified validation split, and pick their final classification threshold from the validation ROC curve rather than a hardcoded guess.

## A data leak we found and fixed

Both data-prep scripts (`process_data.py` for static, `process_dynamic_data_multiclass.py` for dynamic) originally combined every frame/window from every recording into one pool and split train/test *after* that. Since dynamic windows overlap 50% and static recordings capture many near-identical frames per session, this let near-duplicate samples from the same recording land on both sides of the split — inflating test accuracy to ~100% without the model actually generalizing.

Fixed by splitting at the *recording file* level before windowing/combining, so no recording ever contributes to both train and test. Both scripts now do this.

## Honest current results (leak-free)

| | Static | Dynamic |
|---|---|---|
| Test Accuracy | 67.05% | 62.50% |
| Test AUC-ROC | 0.766 | 0.695 |

Both numbers came down significantly once the leak was fixed (previously looked like ~100%). Static is currently slightly ahead of dynamic, but the test sets are tiny (dynamic's test set is built from roughly one held-out recording per class), so this gap isn't strong evidence either architecture is better — it's mostly a sign we don't have enough data yet.

## Known limitations (not fixed in this push, flagging for visibility)

- The validation split inside both `fewshottripletloss*.py` files is still done at the individual frame/window level, not the recording-file level — so validation metrics (which looked much better than test, e.g. 93.5% val vs 67% test on the static run) are still optimistic. Test metrics are the ones to trust. Fixing this would mean carving the validation split out by file too, same technique as the train/test fix.
- `process_data.py`'s file-level split doesn't stratify by gesture type (thumbs up vs. finger gun vs. peace sign vs. closed fist) — it wasn't stratified before this push either, so this preserves existing behavior rather than introducing a regression, but it's worth revisiting once there's more data per gesture.

## Next steps

The honest numbers above point at the same thing the original ticket called for: not enough recorded data yet, for either the static or dynamic path. Next step is recording a lot more one-handed gestures via the Unity/headset setup, for both pipelines, then rerunning `process_data.py` / `process_dynamic_data_multiclass.py` and retraining. More data is likely to move the needle more than further tuning either script right now.
