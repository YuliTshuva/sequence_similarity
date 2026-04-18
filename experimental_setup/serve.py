"""
serve.py
--------
Serves the labeling tool locally so data.json loads automatically.

Usage:
    python serve.py

Then open http://localhost:8000 in your browser.

Place this file in the same folder as labeling_tool.html and data.json.
"""

import http.server
import json
import os
import webbrowser
from http.server import SimpleHTTPRequestHandler


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/auto-data":
            data_path = os.path.join(os.path.dirname(__file__), "data.json")
            if not os.path.exists(data_path):
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"data.json not found next to serve.py")
                return
            with open(data_path, "rb") as f:
                payload = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # silence request logs


if __name__ == "__main__":
    port = 8000
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Serving at http://localhost:{port}")
    print("Opening browser...")
    webbrowser.open(f"http://localhost:{port}/labeling_tool.html")
    with http.server.HTTPServer(("", port), Handler) as httpd:
        httpd.serve_forever()
