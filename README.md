# Universal Gestures: Lab

This repository houses the machine learning/neural network implementation of the Universal Gestures project. See the [Technical Specification](https://docs.google.com/document/d/1wDUTpCBaXz3XE8t48t-PqcnbpXUm-zK26sfdVPcba5U/edit?usp=sharing) for more details.

_Related_: [Universal Gestures Unity Project](https://github.com/uwrealitylabs/universal-gestures-unity/tree/main) and [Scrum Board](https://www.notion.so/7413f4e3318642aba04e34b2f83869a2?v=14850a97b88c470e922605a3cf40e0f5&pvs=4).

# Setup

## Python

Install [Python 3.12.3](https://www.python.org/downloads/) or later.

## requirements.txt

From the root directory, run the following command to install the required packages:

```
pip install -r requirements.txt
```

# Usage: gesture classifier (`model.py`)

1. Populate `data/` with json data collected from the [Unity data collection scene](https://github.com/uwrealitylabs/universal-gestures-unity).
2. Run `process_data.py` to split the dataset into test and train.
3. Run `model.py` to train the model, or run `model_two_hands.py` to train a two-handed model.
   Ensure the number of input features to the model is correct.
4. Find the outputted weights in `trained_model/`. The model is outputted in both .onnx and .json formats.
   The onnx model can then be imported into the [Universal Gestures Unity project](https://github.com/uwrealitylabs/universal-gestures-unity) and used in the inference script.

# Siamese few-shot gestures (`siamese_train.py`)

`src/siamese_train.py` trains an embedding network with triplet loss so that Unity can learn a new gesture from a few clips the user records. Each taught gesture's prototype is the mean embedding of its clips. A frame is that gesture when the nearest prototype is within the "no gesture" cutoff.

**Status: research pipeline.** The model exported to `src/siamese/trained_model/` is experimental. See [Before deploying the exported model](#before-deploying-the-exported-model).

## Data

Recordings from the Unity data collection scene go under `src/siamese/siamese training data/`, one folder per person:

```
<person>/positive/<gesture>/<left|right>/<timestamp>.json
<person>/negative/<everyday|resting|transitions>/<left|right>/<timestamp>.json
```

A person with no negatives is scored against other people's negatives. The script stops with an error if the data cannot support a run. It needs:

- at least 6 gestures (each fold hides 3, validates on 2, and trains on the rest);
- at least 8 training clips per gesture, across people and hands;
- more than K clips per person, hand and gesture to score that gesture K-shot;
- negatives from someone other than the held-out person.

## Train and evaluate

Run from the repo root:

```
python -m src.siamese_train                         # cross-validation, seed 0
python -m src.siamese_train --seeds 0 1 2           # 3 seeds, to measure run-to-run noise
python -m src.siamese_train --test-person edy       # leave one person out: never train on edy's clips
python -m src.siamese_train --seeds 0 1 2 --final   # also train on everything and export for Unity
```

`--augment` adds noise to gesture frames, `--near-miss` adds synthetic near-miss negatives, and `--cutoff-rule` picks the rule used for the exported cutoff. Every rule is scored either way.

How a run is scored:

- Splits are by whole clip, never by frame. Negatives are split by time within each recording folder.
- Each fold hides 3 gestures for testing and holds out 2 for validation. The best epoch and the cutoffs are chosen on the validation gestures only.
- Normalisation is fit on training frames only.
- A raw-feature baseline (no training) is scored on the same episodes as the model.

Results go to `src/siamese/trained_model/results_<options>.json`. With `--final`, the script also writes:

| File | Contents |
| --- | --- |
| `siamese_embedding.onnx` | Input: raw features `[batch, 17]`. Output: L2-normalised embedding `[batch, 16]`. Z-scoring is inside the model. |
| `siamese_embedding.pt` | The same weights for PyTorch. |
| `siamese_config.json` | Gestures, the exported cutoff, every rule's cutoff and its range across folds. |
| `siamese_fixture.json` | A few input frames and PyTorch's embeddings of them, to check ONNX Runtime and Unity against. |

## Tests

```
pip install -r requirements-test.txt
python -m pytest
```

The tests cover:

- Isolation: clips split by person and recording on synthetic and real data, and support clips never scored as queries.
- The cutoff rules, including zero distances and ties.
- Errors when positive or negative clips are missing or too few.
- A short training run and a full command-line run with export.
- PyTorch ↔ ONNX parity for the committed model and for a fresh export.

They also cover the legacy `fewshottripletloss.py` trainer. GitHub Actions runs the suite on every pull request.

## Before deploying the exported model

- **Calibrate the cutoff.** The exported cutoff is the median of the cutoffs the cross-validation fold models chose. Those fold cutoffs span about 0.27–0.84, and the median was not measured on the exported model. Calibrate it on held-out recordings, with as many enrolled gestures as users will actually teach.
- **Measure new users.** The committed results were run without `--test-person`, so every person's clips helped train the model. Run the leave-one-person-out evaluation (`--test-person <name>` for each person) before reporting new-user performance. Also record negatives for every person: edy has none, so edy's false-trigger rate is measured on bryan's negatives.
- **Smoke-test in the headset.** Check that Unity reproduces `siamese_fixture.json`, then try enrolling gestures live.
