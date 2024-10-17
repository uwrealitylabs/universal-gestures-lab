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

# Usage

1. Populate `data/` with json data collected from the [Unity data collection scene](https://github.com/uwrealitylabs/universal-gestures-unity).
2. Run `process_data.py` to split the dataset into test and train.
3. Run `model.py` to train the model, or run `model_two_hands.py` to train a two-handed model.
   Ensure the number of input features to the model is correct.
4. Find the outputted weights in `trained_model/`. The model is outputted in both .onnx and .json formats.
   The onnx model can then be imported into the [Universal Gestures Unity project](https://github.com/uwrealitylabs/universal-gestures-unity) and used in the inference script.
