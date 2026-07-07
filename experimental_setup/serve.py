"""
serve.py
--------
Serves the labeling tool and stores participants' answers on the server.

Usage:
    python serve.py [port]        (default port: 8000; use 80 for a URL with no port,
                                   which requires admin/root)

The site lives under the PREFIX path below, e.g. http://<server>/sequence_labeling/
Set PREFIX = "" to serve at the root instead.

Endpoints (all under the prefix):
    GET  <prefix>/                            -> labeling_tool.html
    GET  <prefix>/auto-data                   -> data.json
    GET  <prefix>/api/user-status?user_id=X   -> {"labeled": [sample names X already answered]}
    POST <prefix>/api/label                   -> append one answer to labels/<user_id>.jsonl
    GET  <prefix>/api/export                  -> download every answer from every user as one JSON file

Each answer is stored as one JSON object per line in labels/<user_id>.jsonl,
so nothing is lost if a participant stops mid-way. Visit /api/export in a
browser to download all labels collected so far.
"""

import json
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(BASE_DIR, "labels")

PREFIX = "/sequence_labeling"   # URL path the site lives under ("" = serve at root)

USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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


class Handler(SimpleHTTPRequestHandler):
    def _send_json(self, obj, status=200, download_name=None):
        payload = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _route(self, path):
        """Strip the URL prefix; returns the app-relative path, or None if outside it."""
        if not PREFIX:
            return path
        if path.startswith(PREFIX + "/"):
            return path[len(PREFIX):]
        return None

    def do_GET(self):
        path = urlparse(self.path).path

        # send the server root and the bare prefix to the app's directory URL
        if PREFIX and path in ("/", PREFIX):
            self.send_response(301)
            self.send_header("Location", PREFIX + "/")
            self.end_headers()
            return

        sub = self._route(path)
        if sub is None:
            return self.send_error(404)

        if sub == "/":
            self.path = "/labeling_tool.html"
            return super().do_GET()

        if sub == "/auto-data":
            data_path = os.path.join(BASE_DIR, "data.json")
            if not os.path.exists(data_path):
                return self._send_json({"error": "data.json not found next to serve.py"}, 404)
            with open(data_path, "rb") as f:
                payload = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if sub == "/api/user-status":
            user_id = parse_qs(urlparse(self.path).query).get("user_id", [""])[0]
            if not USER_ID_RE.match(user_id):
                return self._send_json({"error": "invalid user_id"}, 400)
            labeled = []
            if os.path.exists(user_file(user_id)):
                labeled = [r.get("sample_name") for r in read_user_labels(user_file(user_id))]
            return self._send_json({"labeled": labeled})

        if sub == "/api/export":
            all_labels = []
            if os.path.isdir(LABELS_DIR):
                for fname in sorted(os.listdir(LABELS_DIR)):
                    if fname.endswith(".jsonl"):
                        all_labels.extend(read_user_labels(os.path.join(LABELS_DIR, fname)))
            return self._send_json(all_labels, download_name="all_labels_export.json")

        self.path = sub
        return super().do_GET()

    def do_POST(self):
        if self._route(urlparse(self.path).path) != "/api/label":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            record = json.loads(self.rfile.read(length))
            user_id = record.get("user_id", "")
        except (ValueError, json.JSONDecodeError):
            return self._send_json({"error": "invalid JSON"}, 400)
        if not isinstance(user_id, str) or not USER_ID_RE.match(user_id):
            return self._send_json({"error": "invalid user_id"}, 400)
        os.makedirs(LABELS_DIR, exist_ok=True)
        with open(user_file(user_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return self._send_json({"ok": True})

    def log_message(self, format, *args):
        pass  # silence request logs


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(BASE_DIR)

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except OSError:
        lan_ip = None

    suffix = "" if port == 80 else f":{port}"
    print(f"Open locally:          http://localhost{suffix}{PREFIX}/")
    if lan_ip:
        print(f"Participants' link:    http://{lan_ip}{suffix}{PREFIX}/")
    print(f"Labels are stored in:  {LABELS_DIR}")
    print(f"Export all labels at:  http://localhost{suffix}{PREFIX}/api/export")
    with ThreadingHTTPServer(("", port), Handler) as httpd:
        httpd.serve_forever()
