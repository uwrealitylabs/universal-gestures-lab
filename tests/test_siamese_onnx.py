"""The exported model Unity loads must compute the same embeddings as the PyTorch model it came
from. siamese_fixture.json holds PyTorch's embeddings of a few frames at export time."""

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
import torch

from src import siamese_train as st

MODEL_DIR = st.OUTPUT_DIR


@pytest.fixture(scope="module")
def fixture():
    with open(os.path.join(MODEL_DIR, "siamese_fixture.json")) as f:
        data = json.load(f)
    return np.array(data["input"], dtype=np.float32), np.array(data["expected_embedding"], dtype=np.float32), data["rows"]


def committed_pytorch_model():
    model = st.EmbeddingNet(np.zeros(st.input_dim), np.ones(st.input_dim))
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "siamese_embedding.pt"), weights_only=True))
    return model


def test_committed_onnx_matches_fixture(fixture):
    inputs, expected, _ = fixture
    session = ort.InferenceSession(os.path.join(MODEL_DIR, "siamese_embedding.onnx"))
    np.testing.assert_allclose(session.run(None, {"input": inputs})[0], expected, atol=1e-5)


def test_committed_pytorch_weights_match_fixture(fixture):
    inputs, expected, _ = fixture
    np.testing.assert_allclose(st.embed(committed_pytorch_model(), inputs), expected, atol=1e-5)


def test_fixture_and_config_describe_the_same_model(fixture):
    inputs, expected, rows = fixture
    with open(os.path.join(MODEL_DIR, "siamese_config.json")) as f:
        config = json.load(f)
    assert rows == config["gestures"] + ["negative", "all zeros"]
    assert inputs.shape == (len(rows), len(config["feature_names"]))
    assert expected.shape == (len(rows), config["embedding_dim"])


def test_export_keeps_parity_for_any_batch_size(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = st.EmbeddingNet(rng.normal(size=st.input_dim), rng.uniform(0.5, 2, size=st.input_dim))
    model.eval()
    path = str(tmp_path / "model.onnx")
    max_diff, _ = st.export_onnx(model, path, rng.normal(size=(4, st.input_dim)).astype(np.float32))
    assert max_diff < 1e-5
    batch = rng.normal(size=(9, st.input_dim)).astype(np.float32)  # not the export batch size
    np.testing.assert_allclose(ort.InferenceSession(path).run(None, {"input": batch})[0],
                               st.embed(model, batch), atol=1e-5)
