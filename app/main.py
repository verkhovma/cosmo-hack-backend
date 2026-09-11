from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import geometry
from . import metrics as metrics_mod
from . import routing
from . import storage

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
SCHEMA_IN = "cosmo-A-1.0"
SCHEMA_OUT = "cosmo-A-result-1.0"

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent.parent / "data"))

JOBS: dict = {}
JOBS_META: dict = {}

app = FastAPI(title="CosmoHackaton 2026 — Satellite Constellation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "schema_in": SCHEMA_IN,
        "schema_out": SCHEMA_OUT,
        "data_dir": str(DATA_DIR),
        "data_dir_exists": DATA_DIR.is_dir(),
    }



# ---------------------------------------------------------------------------
# Pydantic-модели запросов
# ---------------------------------------------------------------------------
class ScenarioPayload(BaseModel):
    scenario: dict


class EditPlane(BaseModel):
    plane_id: str
    raan_deg: Optional[float] = Field(None, ge=0, lt=360)
    phase_deg: Optional[float] = Field(None, ge=0, lt=360)


class OutagePayload(BaseModel):
    satellite_id: str
    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., gt=0)


class GatewayOutagePayload(BaseModel):
    gateway_id: str
    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., gt=0)


class EditScenarioPayload(BaseModel):
    scenario: dict
    launch_stage: Optional[int] = Field(None, ge=1, le=3)
    planes: Optional[list[EditPlane]] = None
    add_failures: Optional[list[OutagePayload]] = None
    add_gateway_outages: Optional[list[GatewayOutagePayload]] = None


class SaveProjectPayload(BaseModel):
    title: str
    scenario: dict


class ComparePayload(BaseModel):
    job_ids: list[str] = Field(..., min_length=2)


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def _scenario_from_file(name: str) -> dict:
    path = DATA_DIR / name
    if not path.is_file():
        raise HTTPException(404, f"Scenario file not found: {name}")
    try:
        return geometry.load(path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))


def _apply_edits(scenario: dict, edits: EditScenarioPayload) -> dict:
    """Применяет изменения из интерфейса к сценарию (in-place copy)."""
    s = json.loads(json.dumps(scenario))  # deep copy

    if edits.launch_stage is not None:
        s["design"]["launch_stage"] = edits.launch_stage

    if edits.planes:
        pmap = {p["id"]: p for p in s["design"]["planes"]}
        for edit in edits.planes:
            if edit.plane_id not in pmap:
                raise HTTPException(400, f"Unknown plane: {edit.plane_id}")
            if edit.raan_deg is not None:
                pmap[edit.plane_id]["raan_deg"] = edit.raan_deg
            if edit.phase_deg is not None:
                pmap[edit.plane_id]["phase_deg"] = edit.phase_deg

    if edits.add_failures:
        s["failures"].extend([f.model_dump() for f in edits.add_failures])

    if edits.add_gateway_outages:
        s["gateway_outages"].extend(
            [f.model_dump() for f in edits.add_gateway_outages]
        )

    # повторная валидация после правок
    try:
        geometry.validate(s)
    except ValueError as exc:
        raise HTTPException(400, f"Edited scenario is invalid: {exc}")
    return s


def _run_computation(scenario: dict) -> dict:
    """
    Полный расчёт:
      1) snapshot на каждом шаге сетки;
      2) маршруты BFS от каждого клиента до любого доступного шлюза;
      3) метрики по каждому клиенту.
    """
    e = scenario["environment"]
    step = e["step_s"]
    horizon = e["horizon_s"]
    times = list(range(0, horizon, step))

    clients = [g for g in scenario["ground_sites"] if g["role"] == "client"]
    gateways = [g["id"] for g in scenario["ground_sites"] if g["role"] == "gateway"]

    # routes[client_id][t_index] = path | None
    routes: dict[str, list[Optional[list[str]]]] = {c["id"]: [] for c in clients}
    snapshots_compact: list[dict] = []

    t0 = time.perf_counter()

    for t in times:
        snap = geometry.snapshot(scenario, float(t))
        adj = routing.build_adjacency(snap)

        # компактное представление для фронта
        snapshots_compact.append({
            "t_s": t,
            "satellites": snap["satellites"],
            "edges": snap["edges"],
        })

        for c in clients:
            cid = c["id"]
            best: Optional[list[str]] = None
            for gw in gateways:
                path = routing.bfs_route(adj, cid, gw)
                if path is not None:
                    # предпочитаем более короткий маршрут
                    if best is None or len(path) < len(best):
                        best = path
            routes[cid].append(best)

    elapsed = time.perf_counter() - t0

    # метрики
    per_client: dict[str, dict] = {}
    for c in clients:
        cid = c["id"]
        rs = routes[cid]
        per_client[cid] = {
            "availability": metrics_mod.availability(rs),
            "max_gap_s": metrics_mod.max_gap_steps(rs) * step,
            "hop_counts": metrics_mod.hop_counts(rs),
            "routes": [
                {"t_s": times[k], "path": rs[k]}
                for k in range(len(times))
            ],
        }

    result = {
        "schema_version": SCHEMA_OUT,
        "effective_scenario": scenario,
        "computed_at": time.time(),
        "elapsed_s": elapsed,
        "time_grid": times,
        "snapshots": snapshots_compact,
        "routes": {
            cid: per_client[cid]["routes"] for cid in per_client
        },
        "metrics": {
            cid: {
                "availability": per_client[cid]["availability"],
                "max_gap_s": per_client[cid]["max_gap_s"],
            }
            for cid in per_client
        },
        "target_availability": e["target_availability"],
    }
    return result


# ---------------------------------------------------------------------------
# Эндпоинты: сценарии
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "schema_in": geometry.SCHEMA_IN, "schema_out": SCHEMA_OUT}


@app.get("/api/scenarios")
def list_scenarios() -> list[dict]:
    if not DATA_DIR.is_dir():
        return []
    out = []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            s = geometry.load(p)
            out.append({
                "file": p.name,
                "id": s["meta"]["id"],
                "title": s["meta"]["title"],
            })
        except Exception:
            out.append({"file": p.name, "error": "invalid"})
    return out


@app.get("/api/scenarios/{name}")
def get_scenario(name: str) -> dict:
    if not name.endswith(".json"):
        name += ".json"
    return _scenario_from_file(name)


@app.post("/api/scenarios/validate")
def validate_scenario(payload: ScenarioPayload) -> dict:
    try:
        geometry.validate(payload.scenario)
    except ValueError as exc:
        raise HTTPException(400, {"error": "validation_failed", "detail": str(exc)})
    return {"valid": True}


@app.post("/api/scenarios/upload")
async def upload_scenario(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        scenario = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"Invalid JSON: {exc}")
    try:
        geometry.validate(scenario)
    except ValueError as exc:
        raise HTTPException(400, f"Validation failed: {exc}")
    return {"scenario": scenario, "meta": scenario["meta"]}


# ---------------------------------------------------------------------------
# Эндпоинты: редактирование
# ---------------------------------------------------------------------------
@app.post("/api/scenarios/edit")
def edit_scenario(edits: EditScenarioPayload) -> dict:
    scenario = _apply_edits(edits.scenario, edits)
    return {"scenario": scenario}


# ---------------------------------------------------------------------------
# Эндпоинты: расчёт
# ---------------------------------------------------------------------------
@app.post("/api/compute")
def compute(payload: ScenarioPayload) -> dict:
    scenario = payload.scenario
    try:
        geometry.validate(scenario)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid scenario: {exc}")

    job_id = str(uuid.uuid4())
    result = _run_computation(scenario)
    JOBS[job_id] = result
    JOBS_META[job_id] = {
        "id": job_id,
        "scenario_id": scenario["meta"]["id"],
        "title": scenario["meta"]["title"],
        "created_at": result["computed_at"],
        "elapsed_s": result["elapsed_s"],
    }
    return {
        "job_id": job_id,
        "elapsed_s": result["elapsed_s"],
        "metrics": result["metrics"],
        "target_availability": result["target_availability"],
    }


@app.get("/api/compute/{job_id}/snapshot")
def get_snapshot(job_id: str, t_s: float) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    result = JOBS[job_id]
    times = result["time_grid"]
    # ближайший шаг
    idx = min(range(len(times)), key=lambda k: abs(times[k] - t_s))
    snap = result["snapshots"][idx]
    return {"t_s": times[idx], "snapshot": snap}


@app.get("/api/compute/{job_id}/routes")
def get_routes(job_id: str, client_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    result = JOBS[job_id]
    if client_id not in result["routes"]:
        raise HTTPException(404, f"Unknown client: {client_id}")
    return {
        "client_id": client_id,
        "routes": result["routes"][client_id],
        "metrics": result["metrics"][client_id],
    }


@app.get("/api/compute/{job_id}/metrics")
def get_metrics(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    result = JOBS[job_id]
    return {
        "metrics": result["metrics"],
        "target_availability": result["target_availability"],
        "time_grid_len": len(result["time_grid"]),
    }


@app.get("/api/compute/{job_id}/export")
def export_result(job_id: str) -> JSONResponse:
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    result = JOBS[job_id]
    # компактный формат выгрузки
    payload = {
        "schema_version": SCHEMA_OUT,
        "effective_scenario": result["effective_scenario"],
        "routes": [
            {"t_s": r["t_s"], "client_id": cid, "path": r["path"]}
            for cid, rs in result["routes"].items()
            for r in rs
        ],
        "metrics": result["metrics"],
    }
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{job_id}.json"'},
    )


# ---------------------------------------------------------------------------
# Эндпоинты: сохранение проектов
# ---------------------------------------------------------------------------
@app.post("/api/projects")
def create_project(payload: SaveProjectPayload) -> dict:
    try:
        geometry.validate(payload.scenario)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid scenario: {exc}")
    pid = storage.save_project(payload.title, payload.scenario)
    return {"id": pid, "title": payload.title}


@app.get("/api/projects")
def get_projects() -> list[dict]:
    return storage.list_projects()


@app.get("/api/projects/{pid}")
def get_project(pid: str) -> dict:
    scenario = storage.load_project(pid)
    if scenario is None:
        raise HTTPException(404, "Project not found")
    return {"scenario": scenario}


# ---------------------------------------------------------------------------
# Эндпоинты: сравнение
# ---------------------------------------------------------------------------
@app.post("/api/compare")
def compare(payload: ComparePayload) -> dict:
    jobs = []
    for jid in payload.job_ids:
        if jid not in JOBS:
            raise HTTPException(404, f"Job not found: {jid}")
        jobs.append((jid, JOBS[jid]))

    # собираем общие client_id
    all_clients = set()
    for _, r in jobs:
        all_clients.update(r["metrics"].keys())

    diff = {}
    for cid in sorted(all_clients):
        diff[cid] = {}
        for jid, r in jobs:
            m = r["metrics"].get(cid)
            diff[cid][jid] = {
                "availability": m["availability"] if m else None,
                "max_gap_s": m["max_gap_s"] if m else None,
            }

    # различия в конфигурации
    config_diff = {}
    base = jobs[0][1]["effective_scenario"]
    for jid, r in jobs[1:]:
        s = r["effective_scenario"]
        changes = {}
        if s["design"]["launch_stage"] != base["design"]["launch_stage"]:
            changes["launch_stage"] = [
                base["design"]["launch_stage"], s["design"]["launch_stage"]
            ]
        if s["environment"]["isl_range_km"] != base["environment"]["isl_range_km"]:
            changes["isl_range_km"] = [
                base["environment"]["isl_range_km"], s["environment"]["isl_range_km"]
            ]
        changes["failures_count"] = [
            len(base["failures"]), len(s["failures"])
        ]
        config_diff[jid] = changes

    return {
        "job_ids": payload.job_ids,
        "metrics_diff": diff,
        "config_diff": config_diff,
    }


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return list(JOBS_META.values())


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
