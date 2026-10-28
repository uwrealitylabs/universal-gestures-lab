# Retraining the Siamese gesture model

How the model in `trained_model/siamese/` was trained (2026-10-28), and how to repeat it the
same way when new recordings come in. Everything runs from the repo root.

## 1. Setup (once)

Python 3.11. Then:

```bash
pip install -r requirements.txt
```

Key packages: `torch==2.4.1`, `pytorch-metric-learning==2.9.0`, `onnx==1.17.0`, `onnxruntime==1.19.2`, `scikit-learn==1.5.2`.

## 2. Add new data

Put recordings in `src/siamese training data/` using the same folders:

```
src/siamese training data/
  positive/<gesture name>/<left|right>/<timestamp>.json   # every frame has confidence 1
  negative/<everyday|resting|transitions>/<left|right>/<timestamp>.json   # confidence 0
```

- A new gesture is just a new folder under `positive/`. The script finds gestures by folder name.
- Keep the Unity file names (timestamps). Negatives are split by time using them.
- Aim for about 10 clips per gesture per hand. Each part of the split needs at least a few clips.
- The script stops with an error if a file doesn't have 17 features or its labels don't match its folder.

## 3. Run training (one command)

```bash
python -m src.siamese_train --seeds 0 1 2 --final --cutoff-rule f1_plateau_middle
```

Takes about 15 minutes on a laptop CPU with the current 346 clips. It does two things:

1. **Test (cross-validation).** 13 rounds per seed. Each round hides 3 gestures (test), holds out 2 more (to pick the cutoff and epoch), and trains on the rest plus the negatives. It then teaches the hidden gestures from 1 or 3 clips and scores them. All of this runs 3 times (seeds 0, 1, 2).
2. **Final model (`--final`).** Trains once on all gestures and all negatives, exports ONNX, and checks that ONNX Runtime matches PyTorch.

## 4. Read the results

The printout ends with two tables, **Raw-feature baseline** and **Siamese model**. Look at the `f1_plateau_middle` rows:

| Column | What it means | Target |
|---|---|---|
| F1 (left_1shot / right_1shot) | How well hidden gestures are recognized from 1 clip, negatives included | ≥ 0.85 |
| worst fold F1 | The single worst round | Watch it; flags weak gestures |
| false triggers | Share of negative frames that fired a gesture | Lower is better |
| zeros rejected | Share of "hand not tracked" (all-zero) frames that did not fire | Should be ~1.0 |

Then:

- **Always compare the model to the baseline.** The model is only worth keeping if it beats raw features. The fair check is the `false_trigger_1pct` / `false_trigger_5pct` rows: same false-trigger target, so higher F1 wins.
- **Per-gesture F1** is printed below the tables. Gestures well under 0.9 need more or cleaner clips.

## 5. Outputs (`trained_model/siamese/`)

| File | What it is |
|---|---|
| `siamese_embedding.onnx` | The model for Unity. Input `input` [batch, 17] raw features; output `embedding` [batch, 16]. Normalization is built in. |
| `siamese_embedding.pt` | Same model, PyTorch weights |
| `siamese_config.json` | Gesture list, feature order, the "no gesture" cutoff, epochs, ONNX check result |
| `siamese_fixture.json` | Test inputs plus expected outputs, to check Unity gives the same numbers |
| `results_model_*.json` | Every score from every round |

`trained_model/` is in `.gitignore`, so commit these with `git add -f`.

**The cutoff is a starting value.** It is the median of the cutoffs chosen across rounds (0.57 this time, range 0.25–0.80). Tune it in the headset.

## 6. Optional experiments

Neither helped on the 2026-10-28 data, so the final model uses neither:

```bash
python -m src.siamese_train --seeds 0 1 2 --augment     # noise on gesture frames
python -m src.siamese_train --seeds 0 1 2 --near-miss   # fake "almost a gesture" negatives
```

Add `--final` only to the version you decide to ship.

## 7. What the method is, and where it comes from

| Part | Choice | Source |
|---|---|---|
| Loss | Triplet loss, semi-hard mining, margin 0.2, L2-normalized output | FaceNet, Schroff et al. 2015 |
| Teaching a gesture | Average embedding of its K clips ("prototype"); nearest prototype wins | Prototypical Networks, Snell et al. 2017. Same as Unity's `PrototypeGestureRecog` |
| "No gesture" | Distance to nearest prototype above the cutoff | Threshold on distance, as in face verification |
| Testing | Gestures used for testing are never trained on; cutoff picked on separate gestures | Ravi & Larochelle 2017 |
| Splits | Whole clips, never single frames; negatives split oldest 60% / 20% / newest 20% | Standard practice, so frames from one clip can't leak across splits |
| Model | 17 → 128 → 128 → 16, BatchNorm + ReLU, Adam lr 1e-3, up to 30 epochs × 200 batches | My choice (small, fits Quest) |
| Batches | 1 frame from 8 different clips per gesture, 40 negative frames, 4 all-zero frames | My choice |
| Epoch | Best validation F1; a later epoch must beat it by > 0.001 | Early stopping |

All settings are constants at the top of `src/siamese_train.py`.

## 8. Results on 2026-10-28 (one person, 346 clips)

1-shot F1, `f1_plateau_middle`, mean of 3 seeds:

| | Left | Right | False triggers |
|---|---|---|---|
| Raw features (no training) | 0.973 | 0.961 | ~10–12% |
| Siamese model | 0.977 | 0.960 | ~7–10% |

- 85% target met.
- On one person's data the model ties raw features. Its lower false-trigger rate didn't hold round by round, and at a fixed false-trigger target raw features scored higher F1.
- The real test is a person the model never saw (plan Step 3).
- Weak gestures: thumbs up (right) and open palm.
- In 1–2% of cases an all-zero frame still fired a gesture, so Unity should skip the model when the hand isn't tracked.
