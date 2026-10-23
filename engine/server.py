"""Local demo: python3 engine/server.py  ->  http://localhost:8765  (POST /predict takes JSON)."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from dravix_engine import predict

PAGE = (Path(__file__).parent / "demo.html").read_bytes()


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(PAGE)

    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            body, code = json.dumps(predict(d if isinstance(d, list) else [d])), 200
        except Exception as e:  # noqa: BLE001
            body, code = json.dumps({"error": str(e)}), 400
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body.encode())


if __name__ == "__main__":
    print("Dravix engine on http://localhost:8765")
    HTTPServer(("127.0.0.1", 8765), H).serve_forever()
