"""Smoke test a running API (stdlib only).

    DRAVIX_PASSWORD=... python3 api/smoke_test.py [BASE_URL] [ORIGIN]
    defaults: http://127.0.0.1:8000  http://localhost:5173
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ORIGIN = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5173"
PW = os.environ["DRAVIX_PASSWORD"]
EXAMPLE = {"name": "PP + 20 wt% APP", "matrix_canonical": "PP", "material_form": "bulk",
           "components": "PP|matrix|80 wt%; APP|FR|20 wt%", "heat_flux_kWm2": 50, "thickness_mm": 3,
           "fr_total_wtpct": 20}


def _json(raw):
    try:
        return json.loads(raw or b"null")
    except ValueError:
        return raw.decode()


def call(method, path, body=None, headers=None):
    req = urllib.request.Request(BASE + path, method=method, headers={"Content-Type": "application/json", **(headers or {})},
                                 data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, dict(r.headers), _json(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), _json(e.read())


results = []


def check(name, ok):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name)


s, _, b = call("GET", "/health")
check("GET /health 200", s == 200 and b["status"] == "ok" and b["weights_provisional"] is True and b["weights_version"] == "v1")
check("POST /login bad password -> 401", call("POST", "/login", {"password": "wrong"})[0] == 401)
s, _, b = call("POST", "/login", {"password": PW})
check("POST /login good password -> token", s == 200 and "." in b.get("token", ""))
token = b.get("token", "")
check("POST /predict no token -> 401", call("POST", "/predict", EXAMPLE)[0] == 401)
check("POST /predict forged token -> 401", call("POST", "/predict", EXAMPLE, {"Authorization": "Bearer 9999999999.x"})[0] == 401)
s, _, b = call("POST", "/predict", EXAMPLE, {"Authorization": f"Bearer {token}"})
r = (b or {}).get("results", [{}])[0]
check("POST /predict with token -> 4 properties + CFRPS",
      s == 200 and set(r.get("properties", {})) == {"TTI", "pHRR", "THR", "avg_HRR"} and 0 <= r["cfrps"]["score"] <= 100)
print("  ", json.dumps(r.get("properties")), "CFRPS", round(r.get("cfrps", {}).get("score", -1), 2))
no_ctrl = {k: v for k, v in EXAMPLE.items() if k != "fr_total_wtpct"}
s, _, b = call("POST", "/predict", [no_ctrl, dict(no_ctrl, is_control=1)], {"Authorization": f"Bearer {token}"})
check("POST /predict list, no is_control (engine demo bug) -> 200", s == 200 and len(b["results"]) == 2)
# phr is converted to wt% exactly once, inside the engine (c2f91e7): 100 phr PP + 25 phr APP == 80/20 wt%
phr = dict(EXAMPLE, components="PP|matrix|100 phr; APP|FR|25 phr")
s, _, b = call("POST", "/predict", [EXAMPLE, phr], {"Authorization": f"Bearer {token}"})
check("POST /predict phr input == wt% equivalent", s == 200 and b["results"][0]["properties"] == b["results"][1]["properties"])
check("POST /predict invalid body -> 422", call("POST", "/predict", {"matrix_canonical": "PP"}, {"Authorization": f"Bearer {token}"})[0] == 422)
s, _, b = call("POST", "/predict", dict(EXAMPLE, heat_flux_kWm2=500), {"Authorization": f"Bearer {token}"})
check("POST /predict flux 500 -> 422 with readable message", s == 422 and "heat_flux_kWm2" in str(b))
s, _, b = call("POST", "/predict", dict(EXAMPLE, heat_flux_kWm2=150, matrix_canonical="XYZ"), {"Authorization": f"Bearer {token}"})
check("POST /predict out-of-range flux + unknown matrix -> warnings", s == 200 and len(b["results"][0]["warnings"]) == 2)
pre = {"Origin": ORIGIN, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"}
s, h, _ = call("OPTIONS", "/predict", headers=pre)
check(f"CORS preflight from {ORIGIN} allowed", s == 200 and h.get("access-control-allow-origin") == ORIGIN)
s, h, _ = call("OPTIONS", "/predict", headers=dict(pre, Origin="https://evil.example.com"))
check("CORS preflight from other origin refused", "access-control-allow-origin" not in {k.lower() for k in h})
print(f"{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
