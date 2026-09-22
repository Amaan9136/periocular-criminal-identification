import json
import logging
import os
import uuid
import datetime
from typing import List
import cv2
import numpy as np
from anyio import to_thread
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from config import settings
from schemas import (
    CriminalResponse,
    SearchConfig,
    ProbeReconstruction,
    CandidateFace,
    SearchResponse,
    ModelInfo,
    RuntimeConfigUpdate,
)
from services.vector_store import VectorStore
from services.sqlite_store import SqliteVectorStore
from services.insightface_service import InsightFaceService
from services.reconstruction_service import ReconstructionService
from services import model_registry
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("periocular_api")
app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs(settings.static_dir, exist_ok=True)
os.makedirs(settings.models_dir, exist_ok=True)
app.mount("/media", StaticFiles(directory=settings.static_dir), name="media")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
runtime_config = None
if settings.vector_store_backend == "sqlite":
    vector_store = SqliteVectorStore(settings.sqlite_path, settings.embeddings_dim)
    runtime_config = vector_store
    for k, v in runtime_config.load().items():
        setattr(settings, k, v)
else:
    vector_store = VectorStore(dim=settings.embeddings_dim)
@app.on_event("shutdown")
def _close_store():
    if hasattr(vector_store, "close"):
        vector_store.close()
insight_service = InsightFaceService(vector_store=vector_store)
reconstruction_service = ReconstructionService(gpen_model_path=settings.gpen_model_path)
VALID_RECON_MODES = {"none", "enhance", "hallucinate"}
MAX_IMAGES_PER_REQUEST = 20
def _decode_upload(raw: bytes):
    nparr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
def _save_image(img: np.ndarray, prefix: str) -> str:
    fname = f"{prefix}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(settings.static_dir, fname)
    cv2.imwrite(path, img)
    return f"/media/{fname}"
def _log_search(probe_id: str, mode: str, candidates: List[CandidateFace]):
    if settings.vector_store_backend != "sqlite":
        return
    top = candidates[0].criminal_id if candidates else None
    vector_store.log_search(probe_id, mode, top)
@app.post("/criminals", response_model=CriminalResponse)
async def register_criminal(
    criminal_id: str = Form(...),
    name: str = Form(None),
    metadata: str = Form("{}"),
    images: List[UploadFile] = File(...),
):
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON")
    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"max {MAX_IMAGES_PER_REQUEST} images per request")
    embeddings, image_urls = [], []
    rejected = 0
    for f in images:
        raw = await f.read()
        img = await to_thread.run_sync(_decode_upload, raw)
        if img is None:
            rejected += 1
            continue
        emb = await to_thread.run_sync(insight_service.extract_embedding, img)
        if emb is None:
            rejected += 1
            continue
        embeddings.append(emb)
        image_urls.append(_save_image(img, prefix=f"criminal_{criminal_id}"))
    if not embeddings:
        raise HTTPException(status_code=422, detail="No face detected in any submitted image")
    insight_service.add_criminal(criminal_id, embeddings, meta)
    return CriminalResponse(
        criminal_id=criminal_id,
        name=name,
        metadata=meta,
        image_urls=image_urls,
        images_registered=len(embeddings),
        images_rejected=rejected,
    )
@app.post("/search", response_model=SearchResponse)
async def search_similar_faces(
    probe_id: str = Form(...),
    config_json: str = Form("{}"),
    images: List[UploadFile] = File(...),
):
    try:
        cfg_dict = json.loads(config_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="config_json must be valid JSON")
    cfg = SearchConfig(**cfg_dict)
    if cfg.reconstruction_mode not in VALID_RECON_MODES:
        raise HTTPException(status_code=400, detail=f"reconstruction_mode must be one of {sorted(VALID_RECON_MODES)}")
    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"max {MAX_IMAGES_PER_REQUEST} images per request")
    warnings: List[str] = []
    if cfg.reconstruction_mode == "hallucinate":
        warnings.append(
            "reconstruction_mode='hallucinate' is not implemented (no available DW-KSVD/GAN "
            "weights); probe images were used as-is. See the Models tab for details."
        )
    probe_embeddings = []
    probe_reconstructions: List[ProbeReconstruction] = []
    for idx, f in enumerate(images):
        raw = await f.read()
        img = await to_thread.run_sync(_decode_upload, raw)
        if img is None:
            probe_reconstructions.append(ProbeReconstruction(index=idx, face_detected=False, image_url=None))
            warnings.append(f"image[{idx}] could not be decoded")
            continue
        face = await to_thread.run_sync(insight_service.largest_face, img)
        if face is None:
            probe_reconstructions.append(ProbeReconstruction(index=idx, face_detected=False, image_url=None))
            warnings.append(f"image[{idx}] no face detected")
            continue
        recon_img = await to_thread.run_sync(reconstruction_service.reconstruct, img, cfg.reconstruction_mode, face.bbox)
        if cfg.use_periocular_only:
            crop = insight_service.periocular_crop(recon_img, face)
            embed_input = crop if crop is not None else recon_img
        else:
            embed_input = recon_img
        emb = await to_thread.run_sync(insight_service.extract_embedding, embed_input)
        if emb is None:
            probe_reconstructions.append(ProbeReconstruction(index=idx, face_detected=True, image_url=None))
            warnings.append(f"image[{idx}] face detected but embedding failed on reconstructed/cropped image")
            continue
        probe_embeddings.append(emb)
        url = _save_image(recon_img, prefix=f"probe_{probe_id}")
        probe_reconstructions.append(ProbeReconstruction(index=idx, face_detected=True, image_url=url))
    if not probe_embeddings:
        raise HTTPException(status_code=422, detail="No usable face found in any probe image")
    matches = insight_service.search(probe_embeddings, top_k=cfg.top_k or settings.top_k)
    candidates = [CandidateFace(criminal_id=cid, similarity=sim, metadata=meta) for cid, sim, meta in matches]
    _log_search(probe_id, cfg.reconstruction_mode, candidates)
    return SearchResponse(
        probe_id=probe_id,
        reconstruction_mode=cfg.reconstruction_mode,
        probe_reconstructions=probe_reconstructions,
        candidates=candidates,
        warnings=warnings,
    )
@app.get("/health")
async def health_check():
    return {"status": "ok", "gallery_size": len(vector_store)}
@app.get("/api/stats")
async def api_stats():
    logs = vector_store.recent_logs(10) if settings.vector_store_backend == "sqlite" else []
    return {
        "gallery_size": len(vector_store),
        "vector_store_backend": settings.vector_store_backend,
        "vector_index_status": vector_store.index_status() if hasattr(vector_store, "index_status") else "n/a (in_memory)",
        "recent_searches": logs,
    }
@app.get("/api/models", response_model=List[ModelInfo])
async def api_models():
    return model_registry.get_registry(settings.gpen_model_path)
@app.post("/api/models/{model_id}/install")
async def api_install_model(model_id: str, file: UploadFile = File(None), local_path: str = Form(None), source: str = Form(None), download_url: str = Form(None)):
    if model_id == "buffalo_l":
        if source == "local_path":
            if not local_path:
                raise HTTPException(status_code=400, detail="local_path is required for source=local_path")
            try:
                status = await to_thread.run_sync(model_registry.install_buffalo_l_from_local_path, local_path)
            except (FileNotFoundError, ValueError) as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {"id": model_id, "status": status}
        status = await to_thread.run_sync(model_registry.install_buffalo_l, settings.prefer_gpu)
        return {"id": model_id, "status": status}
    if model_id == "gpen_bfr_256":
        global reconstruction_service
        if source == "local_path":
            if not local_path:
                raise HTTPException(status_code=400, detail="local_path is required for source=local_path")
            try:
                dest = await to_thread.run_sync(model_registry.install_gpen_from_local_path, local_path, settings.models_dir)
            except (FileNotFoundError, ValueError) as e:
                raise HTTPException(status_code=400, detail=str(e))
        elif source == "download":
            url = download_url or model_registry.GPEN_DOWNLOAD_URL
            if not url:
                raise HTTPException(status_code=400, detail="provide a download URL for a source you trust")
            model_registry.start_gpen_download(url, settings.models_dir)
            return {"id": model_id, "status": "downloading"}
        else:
            if file is None:
                raise HTTPException(status_code=400, detail="upload a GPEN-BFR-256.onnx file")
            tmp_path = os.path.join(settings.models_dir, f"_upload_{uuid.uuid4().hex}.onnx")
            with open(tmp_path, "wb") as out:
                out.write(await file.read())
            dest = model_registry.install_gpen(tmp_path, settings.models_dir)
            os.remove(tmp_path)
        settings.gpen_model_path = dest
        if runtime_config is not None:
            runtime_config.save({"gpen_model_path": dest})
        reconstruction_service = ReconstructionService(gpen_model_path=dest)
        return {"id": model_id, "status": "active", "path": dest}
    raise HTTPException(status_code=404, detail="model not installable here")
@app.get("/api/models/{model_id}/status")
async def api_model_status(model_id: str):
    if model_id != "gpen_bfr_256":
        raise HTTPException(status_code=404, detail="no background job for this model")
    job = model_registry.get_job(model_id)
    if not job:
        return {"id": model_id, "state": "idle"}
    if job.get("state") == "done" and job.get("dest") and settings.gpen_model_path != job["dest"]:
        global reconstruction_service
        dest = job["dest"]
        settings.gpen_model_path = dest
        if runtime_config is not None:
            runtime_config.save({"gpen_model_path": dest})
        reconstruction_service = ReconstructionService(gpen_model_path=dest)
        model_registry.clear_job(model_id)
        return {"id": model_id, "state": "done", "path": dest}
    return {"id": model_id, **job}
@app.delete("/api/models/{model_id}")
async def api_delete_model(model_id: str):
    if model_id == "buffalo_l":
        model_registry.delete_buffalo_l()
        return {"id": model_id, "status": "missing"}
    if model_id == "gpen_bfr_256":
        model_registry.delete_gpen(settings.gpen_model_path)
        model_registry.clear_job(model_id)
        settings.gpen_model_path = None
        if runtime_config is not None:
            runtime_config.save({"gpen_model_path": None})
        global reconstruction_service
        reconstruction_service = ReconstructionService(gpen_model_path=None)
        return {"id": model_id, "status": "missing"}
    raise HTTPException(status_code=404, detail="model not deletable here")
@app.get("/api/config")
async def api_get_config():
    return {
        "top_k": settings.top_k,
        "reconstruction_mode_default": settings.reconstruction_mode_default,
        "prefer_gpu": settings.prefer_gpu,
        "min_det_score": settings.min_det_score,
        "gpen_model_path": settings.gpen_model_path,
    }
@app.post("/api/config")
async def api_save_config(update: RuntimeConfigUpdate):
    values = {k: v for k, v in update.model_dump().items() if v is not None}
    for k, v in values.items():
        setattr(settings, k, v)
    if runtime_config is not None:
        runtime_config.save(values)
    return values
@app.get("/", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {"app_name": settings.app_name, "active": "dashboard"})
@app.get("/ui/register", response_class=HTMLResponse)
async def ui_register(request: Request):
    return templates.TemplateResponse(request, "register.html", {"app_name": settings.app_name, "active": "register"})
@app.get("/ui/search", response_class=HTMLResponse)
async def ui_search(request: Request):
    return templates.TemplateResponse(request, "search.html", {"app_name": settings.app_name, "active": "search"})
@app.get("/ui/models", response_class=HTMLResponse)
async def ui_models(request: Request):
    return templates.TemplateResponse(request, "models.html", {"app_name": settings.app_name, "active": "models"})
@app.get("/ui/settings", response_class=HTMLResponse)
async def ui_settings(request: Request):
    return templates.TemplateResponse(request, "settings.html", {"app_name": settings.app_name, "active": "settings", "backend": settings.vector_store_backend, "sqlite_path": os.path.abspath(settings.sqlite_path)})