from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from collections import deque

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
    save_anyway: bool = False


class ComparePayload(BaseModel):
    job_ids: list[str] = Field(..., min_length=2)



# ---------------------------------------------------------------------------
# Объединение интервалов отказов (п.7)
# ---------------------------------------------------------------------------
def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _normalize_outages(scenario: dict) -> dict:
    """Сливает пересекающиеся интервалы отказов по каждому satellite_id / gateway_id."""
    s = json.loads(json.dumps(scenario))

    by_sat: dict[str, list[tuple[float, float]]] = {}
    for f in s["failures"]:
        by_sat.setdefault(f["satellite_id"], []).append((f["start_s"], f["end_s"]))
    s["failures"] = [
        {"satellite_id": sid, "start_s": a, "end_s": b}
        for sid, ivs in by_sat.items()
        for a, b in _merge_intervals(ivs)
    ]

    by_gw: dict[str, list[tuple[float, float]]] = {}
    for f in s["gateway_outages"]:
        by_gw.setdefault(f["gateway_id"], []).append((f["start_s"], f["end_s"]))
    s["gateway_outages"] = [
        {"gateway_id": gid, "start_s": a, "end_s": b}
        for gid, ivs in by_gw.items()
        for a, b in _merge_intervals(ivs)
    ]
    return s


# ---------------------------------------------------------------------------
# Классификация причины перерыва (п.17)
# ---------------------------------------------------------------------------
REASON_NO_VISIBLE = "no_visible_satellite"
REASON_NO_ISL = "no_isl_path"
REASON_NO_GATEWAY = "no_gateway_contact"
REASON_GATEWAY_OFFLINE = "gateway_offline"
REASON_OK = "ok"



def _classify_reason(
    scenario: dict,
    snap: dict,
    client_id: str,
    adj: dict,
    gateway_ids: list[str],
    t_s: float,
) -> str:
    """Классификация причины перерыва. Возвращает одну из REASON_*."""
    # 1. Все шлюзы offline?
    gws_offline = {
        f["gateway_id"]
        for f in scenario.get("gateway_outages", [])
        if f["start_s"] <= t_s < f["end_s"]
    }
    if gws_offline and set(gateway_ids).issubset(gws_offline):
        return REASON_GATEWAY_OFFLINE

    # 2. Есть ли видимые спутники у клиента?
    visible = [
        e[1] for e in snap["edges"]
        if e[0] == client_id and e[1].startswith("S")
    ]
    if not visible:
        return REASON_NO_VISIBLE

    # 3. Достижим ли хотя бы один видимый спутник до любого шлюза?
    for v in visible:
        q = deque([v])
        seen = {v}
        while q:
            u = q.popleft()
            for w, _ in adj.get(u, ()):
                if w in seen:
                    continue
                if w in gateway_ids:
                    return REASON_OK
                seen.add(w)
                q.append(w)

    # 4. Видим, но не доходит. Различаем причины.
    any_isl = any(
        e[0].startswith("S") and e[1].startswith("S")
        for e in snap["edges"]
    )
    if not any_isl:
        return REASON_NO_ISL

    gw_visible = any(
        (e[0] in gateway_ids and e[1].startswith("S"))
        or (e[1] in gateway_ids and e[0].startswith("S"))
        for e in snap["edges"]
    )
    return REASON_NO_ISL if gw_visible else REASON_NO_GATEWAY



# ---------------------------------------------------------------------------
# Полный расчёт с маршрутами и причинами
# ---------------------------------------------------------------------------

def _run_computation(scenario: dict) -> dict:
    """
    Полный расчёт:
      1) snapshot на каждом шаге сетки;
      2) маршруты BFS от каждого клиента до любого доступного шлюза;
      3) причины перерывов;
      4) лог видимости спутников для каждого клиента;
      5) метрики по каждому клиенту.
    """
    scenario = _normalize_outages(scenario)
    e = scenario["environment"]
    step = e["step_s"]
    horizon = e["horizon_s"]
    times = list(range(0, horizon, step))

    clients = [g["id"] for g in scenario["ground_sites"] if g["role"] == "client"]
    gateways = [g["id"] for g in scenario["ground_sites"] if g["role"] == "gateway"]

    # routes[client_id] = [{t_s, path, reason}, ...]
    routes: dict[str, list[dict]] = {c: [] for c in clients}
    # visible_log[client_id][t_index] = [sat_id, ...]
    visible_log: dict[str, list[list[str]]] = {c: [] for c in clients}

    snapshots_compact: list[dict] = []

    t0 = time.perf_counter()

    for t in times:
        snap = geometry.snapshot(scenario, float(t))

        # строим граф смежности
        adj: dict[str, list[tuple[str, float]]] = {}
        for a, b, w in snap["edges"]:
            adj.setdefault(a, []).append((b, w))
            adj.setdefault(b, []).append((a, w))

        # видимость спутников для каждого наземного пункта
        sat_visible: dict[str, list[str]] = {
            s["id"]: [] for s in scenario["design"]["satellites"]
        }
        for a, b, _ in snap["edges"]:
            if a.startswith("S") and not b.startswith("S"):
                sat_visible.setdefault(a, []).append(b)
            elif b.startswith("S") and not a.startswith("S"):
                sat_visible.setdefault(b, []).append(a)

        snapshots_compact.append({
            "t_s": t,
            "satellites": [
                {**s, "visible_to": sat_visible.get(s["id"], [])}
                for s in snap["satellites"]
            ],
            "edges": snap["edges"],
        })

        for cid in clients:
            # BFS до ближайшего шлюза
            best_path: list[str] | None = None
            for gw in gateways:
                path = _bfs(adj, cid, gw)
                if path and (best_path is None or len(path) < len(best_path)):
                    best_path = path

            reason = (
                REASON_OK
                if best_path
                else _classify_reason(scenario, snap, cid, adj, gateways, float(t))
            )
            routes[cid].append({
                "t_s": t,
                "path": best_path,
                "reason": reason,
            })

            # список видимых клиенту спутников
            visible_sats = [
                e2[1] for e2 in snap["edges"]
                if e2[0] == cid and e2[1].startswith("S")
            ]
            visible_log[cid].append(visible_sats)

    elapsed = time.perf_counter() - t0

    # метрики
    metrics: dict[str, dict] = {}
    for cid in clients:
        rs = routes[cid]
        avail = [r["path"] is not None for r in rs]
        max_gap = cur = 0
        for a in avail:
            cur = 0 if a else cur + 1
            max_gap = max(max_gap, cur)
        metrics[cid] = {
            "availability": sum(avail) / len(avail) if avail else 0.0,
            "max_gap_s": max_gap * step,
            "hop_counts": [
                len(r["path"]) - 1 if r["path"] else None for r in rs
            ],
        }

    return {
        "schema_version": "cosmo-A-result-1.0",
        "effective_scenario": scenario,
        "computed_at": time.time(),
        "elapsed_s": elapsed,
        "time_grid": times,
        "snapshots": snapshots_compact,
        "routes": routes,
        "visible_log": visible_log,
        "metrics": metrics,
        "target_availability": e["target_availability"],
    }



def _bfs(adj: dict, src: str, dst: str) -> list[str] | None:
    if src == dst:
        return [src]
    if src not in adj or dst not in adj:
        return None
    prev = {src: None}
    q = deque([src])
    while q:
        v = q.popleft()
        for w, _ in adj.get(v, ()):
            if w in prev:
                continue
            prev[w] = v
            if w == dst:
                path = [w]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(w)
    return None



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
async def upload_scenario(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        scenario = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"Invalid JSON: {exc}")

    validation_errors: list[str] = []
    try:
        geometry.validate(scenario)
    except ValueError as exc:
        validation_errors.append(str(exc))

    return {"scenario": scenario, "validation_errors": validation_errors}


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
def get_routes(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    r = JOBS[job_id]
    return {
        "routes": r["routes"],           # {cid: [{t_s, path, reason}]}
        "visible_log": r["visible_log"], # {cid: [[sat_id, ...], ...]}
        "metrics": r["metrics"],
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
def export_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    r = JOBS[job_id]
    payload = {
        "schema_version": "cosmo-A-result-1.0",
        "effective_scenario": r["effective_scenario"],
        "routes": [
            {"t_s": e["t_s"], "client_id": cid, "path": e["path"]}
            for cid, entries in r["routes"].items()
            for e in entries
        ],
        "metrics": r["metrics"],
    }
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{job_id}.json"'},
    )


# ---------------------------------------------------------------------------
# Эндпоинты: сохранение проектов
# ---------------------------------------------------------------------------
@app.post("/api/projects")
def create_project(payload: SaveProjectPayload):
    validation_errors: list[str] = []
    try:
        geometry.validate(payload.scenario)
    except ValueError as exc:
        validation_errors.append(str(exc))

    if validation_errors and not payload.save_anyway:
        raise HTTPException(400, {
            "error": "validation_failed",
            "detail": validation_errors,
            "hint": "Передайте save_anyway=true, чтобы сохранить всё равно",
        })

    pid = storage.save_project(payload.title, payload.scenario)
    return {
        "id": pid,
        "title": payload.title,
        "validation_warnings": validation_errors,
    }


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
