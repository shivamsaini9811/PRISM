"""Stands in for llama.cpp's llama-server in tests: same arguments, key, and endpoints."""
import argparse, json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("-m"); ap.add_argument("--host"); ap.add_argument("--port", type=int); ap.add_argument("-c")
ap.add_argument("--api-key"); ap.add_argument("-ngl"); ap.add_argument("--reasoning-format")
a = ap.parse_args()
if "crash" in a.m:
    print("error: failed to load model", flush=True); sys.exit(1)
time.sleep(0.5)  # loading


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _auth(self):
        if self.headers.get("Authorization") != f"Bearer {a.api_key}":
            self.send_response(401); self.end_headers(); return False
        return True

    def do_GET(self):
        if self.path == "/v1/models" and self._auth():
            body = json.dumps({"data": [{"id": a.m}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        if self.path == "/v1/chat/completions" and self._auth():
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            words = ["model", a.m.split("/")[-1].split("\\\\")[-1], "says", req["messages"][-1]["content"][:20]]
            self.wfile.write(b'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n\n')
            for w in words:
                self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": w + " "}}]}) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")


print(f"listening on http://{a.host}:{a.port}", flush=True)
HTTPServer((a.host, a.port), H).serve_forever()
