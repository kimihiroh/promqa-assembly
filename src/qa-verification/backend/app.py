"""
Flask-based backend for QA verification (Assembly 101)

"""

from flask import Flask, jsonify, request
from flask_cors import CORS  # Allow requests from React
import logging
import json
import os
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3030"}})

filenames = [
    "samples_10",
]

DIRPATH_INPUT = Path(os.getenv("DIRPATH_INPUT"))
DIRPATH_OUTPUT = Path(os.getenv("DIRPATH_OUTPUT"))

all_data = {}
for filename in filenames:
    filepath = DIRPATH_OUTPUT / f"{filename}.json"
    if filepath.exists():
        with open(filepath, "r") as f:
            data = json.load(f)
        all_data[filename] = data
    else:
        with open(DIRPATH_INPUT / f"{filename}.json", "r") as f:
            data = json.load(f)
        all_data[filename] = data


@app.route("/get_data", methods=["GET"])
def get_data():
    """Send data to React."""
    filename = request.args.get("filename", default="samples_10", type=str)
    idx = request.args.get("idx", default=0, type=int)

    return jsonify(
        {
            "example": all_data[filename]["examples"][idx],
            "total": all_data[filename]["metadata"]["total"],
            "ids": [x for x in range(len(all_data[filename]["examples"]))],
        }
    )


@app.route("/send_data", methods=["POST"])
def receive_data():
    """Receive data from React."""
    incoming_data = request.json
    filename = incoming_data["filename"]
    idx = incoming_data["idx"]

    all_data[filename]["examples"][idx]["verification"] = incoming_data["result"]

    with open(DIRPATH_OUTPUT / f"{filename}.json", "w") as f:
        json.dump(all_data[filename], f, indent=4)
        f.write("\n")

    return jsonify({"status": "success", "received": incoming_data})


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    app.run(debug=True)
