from flask import Flask, jsonify, request
from flask_cors import CORS  # Allow requests from React
import logging
import json
import os
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

DIRPATH_INPUT = Path(os.getenv("DIRPATH_INPUT"))
DIRPATH_OUTPUT = Path(os.getenv("DIRPATH_OUTPUT"))
if not DIRPATH_OUTPUT.exists():
    DIRPATH_OUTPUT.mkdir(parents=True)

annotator_ids = ["all"]

all_data = {}
for annotator_id in annotator_ids:
    filepath = DIRPATH_OUTPUT / f"{annotator_id}.json"
    if filepath.exists():
        with open(filepath, "r") as f:
            data = json.load(f)
        all_data[annotator_id] = data
    else:
        with open(DIRPATH_INPUT / f"{annotator_id}.json", "r") as f:
            data = json.load(f)
        all_data[annotator_id] = data


@app.route("/get_data", methods=["GET"])
def get_data():
    """Send data to React."""
    annotator_id = request.args.get("annotatorId", default="kh", type=str)
    idx = request.args.get("idx", default=1, type=int)

    return jsonify(
        {
            "example": all_data[annotator_id]["examples"][idx],
            "total": all_data[annotator_id]["metadata"]["total"],
            "ids": all_data[annotator_id]["metadata"]["toy_ids"],
        }
    )


@app.route("/send_data", methods=["POST"])
def receive_data():
    """Receive data from React."""
    incoming_data = request.json
    annotator_id, idx, nodes, edges = (
        incoming_data["annotatorId"],
        incoming_data["idx"],
        incoming_data["nodes"],
        incoming_data["edges"],
    )
    all_data[annotator_id]["examples"][idx]["nodes"] = nodes
    all_data[annotator_id]["examples"][idx]["edges"] = edges

    with open(DIRPATH_OUTPUT / f"{annotator_id}.json", "w") as f:
        json.dump(all_data[annotator_id], f, indent=4)
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
