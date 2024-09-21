'''
Tests the onnx model. Used to verify the model is working correctly and outputs same values as when run in Unity.
'''

import onnx
import onnxruntime as ort
import numpy as np

# Load the ONNX model
model_path = 'trained_model/model_weights.onnx'
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# Create an ONNX Runtime session
ort_session = ort.InferenceSession(model_path)

# Dummy input for testing
dummy_input = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
# alternatively, use a random value
# dummy_input = np.random.randn(1, 17).astype(np.float32)

# Run the model on the dummy input
outputs = ort_session.run(None, {'l_x_': dummy_input})

# Print the outputs
print("Model outputs: ", outputs)