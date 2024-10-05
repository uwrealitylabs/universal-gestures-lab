'''
Tests the onnx model. Used to verify the model is working correctly and outputs same values as when run in Unity.
'''

import onnx
import onnxruntime as ort
import numpy as np

# Load the ONNX model
model_path = 'trained_model/model_two_hands_weights.onnx' # switch to model_weights.onnx for one hand.
input_size = 44 # 44 features. switch to 17 for one hand.
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# Create an ONNX Runtime session
ort_session = ort.InferenceSession(model_path)

# Dummy input for testing
dummy_input = np.random.randn(1, input_size).astype(np.float32)
# alternatively, use a specific value
# dummy_input = np.array([[215.83316040039063,38.261817932128909,175.34902954101563,119.66554260253906,181.4375,0.09320686757564545,252.98167419433595,22.649574279785158,238.48731994628907,0.03712514787912369,251.87472534179688,19.05085563659668,252.16165161132813,0.05285436660051346,250.74102783203126,258.4759826660156,0.06365345418453217,221.0125732421875,28.43832015991211,175.9957275390625,114.1581039428711,198.5877685546875,0.08614794909954071,254.12100219726563,23.678512573242189,251.20851135253907,0.042295362800359729,251.9467010498047,22.867633819580079,267.7880859375,0.05861354619264603,248.66685485839845,280.1216125488281,0.07259343564510346,0.5990011096000671,-0.034888386726379397,0.0,0.6000162959098816,0.06954740732908249,-0.9975786209106445,0.7476283311843872,-0.6641173362731934,-0.27261248230934145,0.9621239304542542]], dtype=np.float32)

# Run the model on the dummy input
outputs = ort_session.run(None, {'l_x_': dummy_input})

# Print the outputs
print("Model outputs: ", outputs)