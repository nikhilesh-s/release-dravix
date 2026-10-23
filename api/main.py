"""Dravix engine API: password login -> bearer token -> /predict (engine properties + CFRPS).

Env (no defaults for secrets; login and predict fail closed when unset):
  DRAVIX_PASSWORD          shared dashboard password
  DRAVIX_SECRET            token signing key (long random string)
  DRAVIX_TOKEN_TTL         token lifetime in seconds (default 28800 = 8 h)
  DRAVIX_CORS_ORIGINS      comma-separated exact origins (e.g. http://localhost:5173)
  DRAVIX_CORS_ORIGIN_REGEX regex for Vercel preview URLs
                           (default: this project's <hash>.vercel.app and <hash|git-branch>-<team>.vercel.app URLs)

Run: uvicorn api.main:app --port 8000
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from api import model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine/score"))
from cfrps import ENGINE_OUTPUTS, UNITS, WEIGHTS, WEIGHTS_DIR, WEIGHTS_VERSION, cfrps  # noqa: E402

# provisional until the expert (AHP) prior is applied to the served weights config
WEIGHTS_PROVISIONAL = json.loads((WEIGHTS_DIR / f"cfrs_weights_{WEIGHTS_VERSION}.json").read_text())["expert_prior"]["weights"] is None

TTL = int(os.environ.get("DRAVIX_TOKEN_TTL", "28800"))
META = model.load()

app = FastAPI(title="Dravix engine API", version="2.0.0-preview")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("DRAVIX_CORS_ORIGINS", "").split(",") if o.strip()],
    allow_origin_regex=os.environ.get("DRAVIX_CORS_ORIGIN_REGEX",
                                      r"https://dravix-master-engine-preview(?:-(?:[a-z0-9]{9}|[a-z0-9-]+-nikhilesh-suravarjjalas-projects))?\.vercel\.app"),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def _invalid(_: Request, exc: RequestValidationError):
    """One readable string for the UI. Also avoids a 500 when the rejected input is NaN (not JSON-encodable)."""
    errs = [e for e in exc.errors() if e["type"] not in ("list_type", "model_attributes_type")] or exc.errors()
    msg = "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in errs[:5])
    return JSONResponse({"detail": f"invalid input: {msg}"}, status_code=422)


def _secret() -> bytes:
    s = os.environ.get("DRAVIX_SECRET", "")
    if len(s) < 16:
        raise HTTPException(503, "server auth not configured")
    return s.encode()


def _sign(exp: str) -> str:
    return base64.urlsafe_b64encode(hmac.new(_secret(), exp.encode(), hashlib.sha256).digest()).decode().rstrip("=")


def make_token() -> str:
    exp = str(int(time.time()) + TTL)
    return f"{exp}.{_sign(exp)}"


def require_token(cred: Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))]):
    exp, _, sig = (cred.credentials if cred else "").partition(".")
    if not (exp.isdigit() and hmac.compare_digest(sig, _sign(exp)) and int(exp) > time.time()):
        raise HTTPException(401, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"})


# ponytail: per-process in-memory limiter; fine for one free-tier instance, use a shared store if scaled out
FAILS: dict[str, deque] = defaultdict(deque)
MAX_FAILS, WINDOW = 10, 900


class Login(BaseModel):
    password: str = Field(max_length=256)


class Formulation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    name: str | None = Field(None, max_length=200)
    matrix_canonical: str = Field(min_length=1, max_length=100)
    material_form: str = Field(min_length=1, max_length=50)
    components: str = Field(min_length=1, max_length=2000)
    heat_flux_kWm2: float | None = Field(None, gt=0, le=200)
    thickness_mm: float | None = Field(None, gt=0, le=500)
    fr_total_wtpct: float | None = Field(None, ge=0, le=100)
    filler_total_wtpct: float | None = Field(None, ge=0, le=100)
    density_kgm3: float | None = Field(None, gt=0, le=20000)
    area_cm2: float | None = Field(None, gt=0, le=10000)
    is_control: int | None = Field(None, ge=0, le=1)  # omitted -> engine defaults to 0 (fixed upstream in c2f91e7)


# training-set ranges (data/v2/cone_lit_benchmark_v1.csv): flux 10-95 kW/m2, thickness 0.04-70 mm
def warn(f: Formulation) -> list[str]:
    w = []
    if f.matrix_canonical not in META["matrices"]:
        w.append(f"matrix '{f.matrix_canonical}' was not in the training data (treated as 'other')")
    if f.material_form not in META["forms"]:
        w.append(f"material form '{f.material_form}' was not in the training data")
    if not all(len(c.split("|")) == 3 for c in f.components.split(";") if c.strip()):
        w.append("components do not all look like name|role|amount; additives may be ignored")
    if f.heat_flux_kWm2 is None:
        w.append("no heat flux given; value imputed, prediction is unreliable")
    elif not 10 <= f.heat_flux_kWm2 <= 95:
        w.append("heat flux outside the 10-95 kW/m2 training range; extrapolation")
    if f.thickness_mm is not None and not 0.04 <= f.thickness_mm <= 70:
        w.append("thickness outside the 0.04-70 mm training range; extrapolation")
    return w


@app.get("/health")
def health():
    return {"status": "ok", "model": META, "weights_provisional": WEIGHTS_PROVISIONAL, "weights_version": WEIGHTS_VERSION}


@app.post("/login")
def login(body: Login, request: Request):
    # rightmost X-Forwarded-For entry = the one Render's proxy appended; the left side is client-controlled (spoofable)
    ip = request.headers.get("x-forwarded-for", "").split(",")[-1].strip() or (request.client.host if request.client else "?")
    q, now = FAILS[ip], time.time()
    while q and q[0] < now - WINDOW:
        q.popleft()
    if len(q) >= MAX_FAILS:
        raise HTTPException(429, "too many attempts, try again later")
    pw = os.environ.get("DRAVIX_PASSWORD", "")
    if not pw:
        raise HTTPException(503, "server auth not configured")
    if not hmac.compare_digest(body.password.encode(), pw.encode()):
        q.append(now)
        raise HTTPException(401, "incorrect password")
    return {"token": make_token(), "expires_in": TTL}


@app.post("/predict", dependencies=[Depends(require_token)])
def predict(body: Formulation | Annotated[list[Formulation], Field(min_length=1, max_length=50)]):
    items = body if isinstance(body, list) else [body]
    rows = [f.model_dump(exclude={"name"} | ({"is_control"} if f.is_control is None else set())) for f in items]
    try:
        preds = model.E.predict(rows)
    except Exception as e:  # noqa: BLE001  engine raises plain errors for bad input
        raise HTTPException(400, f"engine error: {e}") from e
    results = []
    for f, p in zip(items, preds):
        props = {k: p[k] for k in ENGINE_OUTPUTS}
        results.append({"name": f.name, "properties": props, "uncertainty": None, "cfrps": cfrps(props), "warnings": warn(f)})
    return {"results": results, "units": {k: UNITS[k] for k in ENGINE_OUTPUTS},
            "cfrps_weights": WEIGHTS, "weights_version": WEIGHTS_VERSION, "weights_provisional": WEIGHTS_PROVISIONAL,
            "notes": ["The engine returns point predictions only (no uncertainty); CFRPS interval95 is therefore the point score.",
                      "CFRPS is 0-100, higher = more fire risk; compare formulations at the same heat flux.",
                      f"CFRPS weights {WEIGHTS_VERSION}: fitted to UL-94 outcomes; expert review pending." if WEIGHTS_PROVISIONAL else f"CFRPS weights {WEIGHTS_VERSION}."]}
