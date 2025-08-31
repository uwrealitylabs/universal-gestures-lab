from flask import Flask, request, jsonify, send_file
from src import model
from src import model_two_hands
from src import process_data
import os
import shutil
import socket
app = Flask(__name__)

def print_connection_instructions(port):
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = local_ip
    print("\n" + "="*60)
    print("Flask server is starting!\n")
    print(f"Paste this address into the Uri field in Unity if you’re using the Meta XR Simulator on your computer, or a headset plugged into your computer in Link Mode:\n")
    print(f"    http://127.0.0.1:{port}\n")
    print(f"Paste this address into the Uri field in Unity if you’re using an unplugged headset on the same WiFi network as your computer:\n")
    print(f"    http://{lan_ip}:{port}\n")
    print("="*60 + "\n")



@app.route('/train_model_one_hand/', methods=['POST'])
def train_mode_one_hand():
  data = request.json

  
  clearServerData()

  with open("src/serverdata/data.json", "w") as f:
    f.write(str(data).replace("'","\"" ))

  process_data.split("src/serverdata")
  model.main()
  return send_file("trained_model/model_weights.onnx", as_attachment=True)

  # return jsonify({"message": "Data received", "data": data}), 200


@app.route('/train_model_two_hands/', methods=['POST'])
def train_mode_two_hands():
  data = request.json

  with open("src/serverdata/data.json", "w") as f:
    f.write(str(data).replace("'","\"" ))

  process_data.split("src/serverdata")
  model_two_hands.main()

  return send_file("trained_model/model_two_hands_weights.onnx", as_attachment=True)
  # return jsonify({"message": "Data received", "data": data}), 200


def clearServerData():
  shutil.rmtree('src/serverdata')
  os.makedirs('src/serverdata')



if __name__ == "__main__":
    port = 8080
    print_connection_instructions(port)
    app.run(port=port, host="0.0.0.0", debug=True)
