from flask import Flask, request, jsonify, send_file
import model
import model_two_hands
import os
import process_data
import shutil
app = Flask(__name__)



@app.route('/train_model_one_hand/', methods=['POST'])
def train_mode_one_hand():
  data = request.json

  
  clearServerData()

  with open("serverdata/data.json", "w") as f:
    f.write(str(data).replace("'","\"" ))

  process_data.split("serverdata")
  model.main()
  return send_file("trained_model/model_weights.onnx", as_attachment=True)
  

  return jsonify({"message": "Data received", "data": data}), 200


@app.route('/train_model_two_hands/', methods=['POST'])
def train_mode_two_hands():
  data = request.json

  with open("serverdata/data.json", "w") as f:
    f.write(str(data).replace("'","\"" ))

  process_data.split("serverdata")
  model_two_hands.main()

  return send_file("trained_model/model_two_hands_weights.onnx", as_attachment=True)
  return jsonify({"message": "Data received", "data": data}), 200


def clearServerData():
  shutil.rmtree('serverdata')
  os.makedirs('serverdata')





app.run(port="8080", host="0.0.0.0", debug=True)