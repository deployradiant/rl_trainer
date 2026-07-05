"""Modal web app for visualizing the latest trajectory training artifacts."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import modal

app = modal.App("rl-trainer-visualizer")

data_vol = modal.Volume.from_name("soccernet-tracking", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi")
    .add_local_dir("web", "/root/web")
)

WEB_ROOT = Path("/root/web")
ARTIFACT_ROOT = Path(os.environ.get("RL_TRAINER_ARTIFACT_ROOT", "/data"))
TRACKING_ROOT = ARTIFACT_ROOT / "tracking-2023"
FRAME_CACHE_HEADERS = {"Cache-Control": "public, max-age=3600"}


def _read_json(path: Path, fallback):
    if not path.exists():
        return fallback
    return json.loads(path.read_text())


def _rollouts():
    rollouts_path = ARTIFACT_ROOT / "trajectory_rollouts.json"
    data = _read_json(rollouts_path, {})
    return data.get("rollouts", [])


def _reload_artifacts():
    data_vol.reload()


@app.function(
    image=image,
    volumes={"/data": data_vol},
)
@modal.asgi_app()
def visualizer():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse, Response

    server = FastAPI(title="RL Trainer Visualizer")

    @server.get("/")
    def index():
        return FileResponse(WEB_ROOT / "index.html", media_type="text/html")

    @server.get("/api/results")
    def results():
        _reload_artifacts()
        manifest = _read_json(ARTIFACT_ROOT / "sequence_manifest.json", {})
        history = _read_json(ARTIFACT_ROOT / "training_history.json", [])
        rollouts = _rollouts()
        preview_path = ARTIFACT_ROOT / "trajectory_preview.png"
        rollouts_path = ARTIFACT_ROOT / "trajectory_rollouts.json"
        latest = history[-1] if history else {}
        rollout_updated_at = rollouts_path.stat().st_mtime if rollouts_path.exists() else None

        return JSONResponse(
            {
                "manifest": manifest,
                "history": history,
                "latest": latest,
                "rollouts": {
                    "available": rollouts_path.exists(),
                    "updated_at": rollout_updated_at,
                    "data": rollouts,
                    "frame_url": "/frames/{split}/{sequence}/{frame_number}.jpg",
                },
                "preview": {
                    "url": "/artifacts/trajectory_preview.png",
                    "available": preview_path.exists(),
                    "updated_at": preview_path.stat().st_mtime if preview_path.exists() else None,
                },
            }
        )

    @server.get("/api/rollouts")
    def rollouts():
        _reload_artifacts()
        data = _rollouts()
        if not data:
            return JSONResponse({"error": "trajectory_rollouts.json not found"}, status_code=404)
        return JSONResponse({"rollouts": data})

    @server.get("/artifacts/trajectory_preview.png")
    def preview():
        _reload_artifacts()
        path = ARTIFACT_ROOT / "trajectory_preview.png"
        if not path.exists():
            return JSONResponse({"error": "trajectory_preview.png not found"}, status_code=404)
        return FileResponse(path, media_type="image/png")

    def _serve_frame(split: str, sequence: str, frame_number: int):
        extracted = TRACKING_ROOT / split / sequence / "img1" / f"{frame_number:06d}.jpg"
        if extracted.exists():
            return FileResponse(extracted, media_type="image/jpeg", headers=FRAME_CACHE_HEADERS)

        archive_path = TRACKING_ROOT / f"{split}.zip"
        if not archive_path.exists():
            return JSONResponse({"error": f"{split}.zip not found"}, status_code=404)

        member = f"{split}/{sequence}/img1/{frame_number:06d}.jpg"
        try:
            with zipfile.ZipFile(archive_path) as archive:
                return Response(
                    archive.read(member), media_type="image/jpeg", headers=FRAME_CACHE_HEADERS
                )
        except KeyError:
            return JSONResponse({"error": f"{member} not found"}, status_code=404)

    @server.get("/frames/{split}/{sequence}/{frame_number}.jpg")
    def split_sequence_frame(split: str, sequence: str, frame_number: int):
        return _serve_frame(split, sequence, frame_number)

    @server.get("/frames/{sequence}/{frame_number}.jpg")
    def sequence_frame(sequence: str, frame_number: int):
        _reload_artifacts()
        rollouts = _rollouts()
        rollout = next((item for item in rollouts if item.get("sequence") == sequence), {})
        split = rollout.get("split")
        if not split:
            return JSONResponse({"error": f"rollout for {sequence} not found"}, status_code=404)
        return _serve_frame(split, sequence, frame_number)

    @server.get("/health")
    def health():
        return {"ok": True}

    return server
