from flask import Flask, request, jsonify, send_file
from src import model
from src import model_two_hands
from src import process_data
import os
import shutil
app = Flask(__name__)


@app.route('/train_model_one_hand/', methods=['POST'])
def train_mode_one_hand():

  # parsing the JSON request
  try:
    data = request.json
    if data is None:
      raise ValueError("Invalid JSON")
  except Exception as e:
    return jsonify({"error":"Invalid JSON payload",
                    "type":e.__class__.__name__}), 400
  
  # clearing server data
  try:
    clearServerData()
  except Exception as e:
    return jsonify({"error":"Filesystem error",
                    "type":e.__class__.__name__}), 500

  # writing to data.json --- could probably be improved
  try:
    with open("src/serverdata/data.json", "w") as f:
      f.write(str(data).replace("'","\"" ))
  except Exception as e:
    return jsonify({"error":"Failed to write data.json",
                    "type":e.__class__.__name__}), 500
  
  # data preprocessing
  try:
    process_data.split("src/serverdata")
  except Exception as e:
    return jsonify({"error":"Preprocessing failed",
                   "type":e.__class__.__name__}), 500

  # model training
  try:
    model.main()
  except Exception as e:
    return jsonify({"error":"Model training failed",
                    "type":e.__class__.__name__}), 500

  # sending and returning file
  try:
    model_file = os.path.join("trained_model", "model_weights.onnx")
    if not os.path.exists(model_file):
      return jsonify("error":"Trained model file not found",
                     "expected_path":model_file), 500
    return send_file("model_file", as_attachment=True)
  except Exception as e:
    return jsonify({"error":"Failed to send file",
                    "type":e.__class__.__name__}), 500

  # return jsonify({"message": "Data received", "data": data}), 200


@app.route('/train_model_two_hands/', methods=['POST'])
def train_mode_two_hands():

  # parsing the JSON request
  try:
    data = request.json
  except Exception as e:
    return jsonify({"error":"Invalid JSON payload",
                    "type":e.__class__.__name__}), 400

  # writing to json.data --- also could probably be improved
  try:
    with open("src/serverdata/data.json", "w") as f:
      f.write(str(data).replace("'","\"" ))
  except Exception as e:
    return jsonify({"error":"Failed to write data.json",
                   "type":e.__class__.__name__}), 500
  
  # data preprocessing
  try:
    process_data.split("src/serverdata")
  except Exception as e:
    return jsonify({"error":"Preprocessing failed",
                    "type":e.__class__.__name__}), 500
  
  # model training
  try:
    model_two_hands.main()
  except Exception as e:
    return jsonify({"error":"Model training failed",
                    "type":e.__class__.__name__}), 500)
  
  # sending and returning file
  try:
    model_file = os.path.join("trained_model", "model_two_hands_weights.onnx")
    if not os.path.exists(model_file):
      return jsonify({"error":"Trained model file not found",
                      "expected_path":model_file}), 500
    return send_file(model_file, as_attachment=True)
  except Exception as e:
    return jsonify({"error":"Failed to send file",
                    "type":e.__class__.__name__}), 500
  
  # return jsonify({"message": "Data received", "data": data}), 200


def clearServerData():

  shutil.rmtree('src/serverdata')
  os.makedirs('src/serverdata')


app.run(port="8080", host="0.0.0.0", debug=True)