"""
flask_app.py
------------
Flask version of serve.py, for deploying on web hosting platforms
(e.g. PythonAnywhere) where you cannot open your own port.

Same endpoints and storage as serve.py:
    GET  /sequence_labeling/                     -> labeling_tool.html
    GET  /sequence_labeling/auto-data            -> data.json
    GET  /sequence_labeling/api/user-status?user_id=X
    POST /sequence_labeling/api/label            -> append to labels/<user_id>.jsonl
    GET  /sequence_labeling/api/export           -> download all answers as one JSON file

Run locally:
    python flask_app.py        -> http://localhost:8000/sequence_labeling/

# To run on server
gunicorn --bind 0.0.0.0:8000 --workers 3 --threads 4 --timeout 120 flask_app:app

On PythonAnywhere, point the WSGI config at this module:
    from flask_app import app as application
"""

import json
import os
import re

from flask import Flask, Response, jsonify, redirect, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(BASE_DIR, "labels")

PREFIX = "/sequence_labeling"   # URL path the site lives under

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = Flask(__name__)


def user_file(user_id: str) -> str:
    return os.path.join(LABELS_DIR, f"{user_id}.jsonl")


def read_user_labels(path: str) -> list:
    labels = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(json.loads(line))
    return labels


@app.get("/")
def root():
    return redirect(PREFIX + "/")


@app.get(PREFIX + "/")
def index():
    return send_from_directory(BASE_DIR, "labeling_tool.html")


@app.get(PREFIX + "/auto-data")
def auto_data():
    return send_from_directory(BASE_DIR, "data.json")


@app.get(PREFIX + "/api/user-status")
def user_status():
    user_id = request.args.get("user_id", "")
    if not USER_ID_RE.match(user_id):
        return jsonify({"error": "invalid user_id"}), 400
    labeled = []
    if os.path.exists(user_file(user_id)):
        labeled = [r.get("sample_name") for r in read_user_labels(user_file(user_id))]
    return jsonify({"labeled": labeled})


@app.post(PREFIX + "/api/label")
def save_label():
    record = request.get_json(silent=True)
    if not isinstance(record, dict):
        return jsonify({"error": "invalid JSON"}), 400
    user_id = record.get("user_id", "")
    if not isinstance(user_id, str) or not USER_ID_RE.match(user_id):
        return jsonify({"error": "invalid user_id"}), 400
    os.makedirs(LABELS_DIR, exist_ok=True)
    with open(user_file(user_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return jsonify({"ok": True})


@app.get(PREFIX + "/api/export")
def export():
    all_labels = []
    if os.path.isdir(LABELS_DIR):
        for fname in sorted(os.listdir(LABELS_DIR)):
            if fname.endswith(".jsonl"):
                all_labels.extend(read_user_labels(os.path.join(LABELS_DIR, fname)))
    return Response(
        json.dumps(all_labels, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": 'attachment; filename="all_labels_export.json"'},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
