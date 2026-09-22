import os
import shutil
import logging
import zipfile
import threading
import time
import urllib.request
from typing import List, Dict, Optional
logger = logging.getLogger("periocular_api.model_registry")
INSIGHTFACE_HOME = os.path.expanduser(os.environ.get("INSIGHTFACE_HOME", "~/.insightface"))
BUFFALO_DIR = os.path.join(INSIGHTFACE_HOME, "models", "buffalo_l")
BUFFALO_FILES = ["det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx", "2d106det.onnx", "genderage.onnx"]
BUFFALO_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
GPEN_DOWNLOAD_URL = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GPEN-BFR-256.onnx"
_download_jobs: Dict[str, Dict] = {}
_download_lock = threading.Lock()
def _set_job(model_id: str, **fields):
    with _download_lock:
        job = _download_jobs.setdefault(model_id, {})
        job.update(fields)
def get_job(model_id: str) -> Optional[Dict]:
    with _download_lock:
        job = _download_jobs.get(model_id)
        return dict(job) if job else None
def clear_job(model_id: str):
    with _download_lock:
        _download_jobs.pop(model_id, None)
def _buffalo_status() -> str:
    if all(os.path.exists(os.path.join(BUFFALO_DIR, f)) for f in ["det_10g.onnx", "w600k_r50.onnx"]):
        return "active"
    return "missing"
def _gpen_status(path: Optional[str]) -> str:
    if path and os.path.exists(path):
        return "active"
    return "missing"
def get_registry(gpen_model_path: Optional[str]) -> List[Dict]:
    return [
        {"id": "buffalo_l", "name": "InsightFace buffalo_l (SCRFD-10G + ArcFace R50)", "required": True, "status": _buffalo_status(), "path": BUFFALO_DIR, "installable": True, "install_methods": ["download", "local_path"], "download_url": BUFFALO_ZIP_URL, "description": "Face detection, landmarks and 512-d ArcFace embeddings. Required for /criminals and /search to function at all."},
        {"id": "gpen_bfr_256", "name": "GPEN-BFR-256 (blind face restoration)", "required": False, "status": _gpen_status(gpen_model_path), "path": gpen_model_path or "", "installable": True, "install_methods": ["download", "local_path", "upload"], "download_url": GPEN_DOWNLOAD_URL, "description": "Optional. Powers reconstruction_mode='enhance'. Sharpens an already-captured face crop; does not invent missing regions. Can subtly alter identity-bearing detail - keep opt-in."},
        {"id": "dwksvd_gan_hallucinate", "name": "DW-KSVD + GAN periocular-to-full-face hallucination", "required": False, "status": "not_implemented", "path": "", "installable": False, "install_methods": [], "description": "Optional, NOT AVAILABLE. No maintained open-source checkpoint exists for this; would require training a dictionary + conditional GAN on paired periocular/full-face data. reconstruction_mode='hallucinate' calls this and no-ops with a warning rather than fabricating output."},
        {"id": "inswapper_128", "name": "inswapper_128 (face-swap, excluded)", "required": False, "status": "excluded", "path": "", "installable": False, "install_methods": [], "description": "Excluded on purpose. This is a face-swap model, not a reconstruction model - it pastes a separate source identity onto a target image. Using it here would hand investigators a photorealistic image of the wrong person with no indication it was fabricated. Not wired into this pipeline."},
    ]
def _download_to(url: str, dest_path: str, timeout: int = 120, model_id: Optional[str] = None) -> None:
    resume_from = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    headers = {"User-Agent": "periocular-criminal-identification/1.0"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total_header = resp.headers.get("Content-Length")
        if resp.status == 206 and total_header:
            total = resume_from + int(total_header)
        elif total_header:
            total = int(total_header)
            resume_from = 0
        else:
            total = None
        mode = "ab" if resp.status == 206 and resume_from else "wb"
        if mode == "wb":
            resume_from = 0
        written = resume_from
        chunk_size = 1024 * 256
        last_update = 0.0
        with open(dest_path, mode) as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                now = time.time()
                if model_id and (now - last_update > 0.2):
                    last_update = now
                    _set_job(model_id, bytes_done=written, bytes_total=total, state="downloading")
        if model_id:
            _set_job(model_id, bytes_done=written, bytes_total=total or written, state="finalizing")
def _run_gpen_download_job(model_id: str, url: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "GPEN-BFR-256.onnx")
    tmp_path = os.path.join(dest_dir, "GPEN-BFR-256.onnx.part")
    _set_job(model_id, state="downloading", bytes_done=0, bytes_total=None, error=None, dest=None)
    try:
        _download_to(url, tmp_path, model_id=model_id)
        shutil.move(tmp_path, dest)
        _set_job(model_id, state="done", dest=dest)
    except Exception as e:
        logger.exception("GPEN download failed")
        _set_job(model_id, state="error", error=str(e))
def start_gpen_download(url: str, dest_dir: str) -> None:
    existing = get_job("gpen_bfr_256")
    if existing and existing.get("state") == "downloading":
        return
    t = threading.Thread(target=_run_gpen_download_job, args=("gpen_bfr_256", url, dest_dir), daemon=True)
    t.start()
def install_buffalo_l(prefer_gpu: bool):
    if _buffalo_status() == "active":
        return _buffalo_status()
    from insightface.app import FaceAnalysis
    from config import detect_providers
    providers = detect_providers(prefer_gpu)
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return _buffalo_status()
def install_buffalo_l_from_local_path(local_path: str) -> str:
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"path does not exist: {local_path}")
    os.makedirs(BUFFALO_DIR, exist_ok=True)
    if zipfile.is_zipfile(local_path):
        with zipfile.ZipFile(local_path) as zf:
            zf.extractall(BUFFALO_DIR)
    elif os.path.isdir(local_path):
        for fname in BUFFALO_FILES:
            src = os.path.join(local_path, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(BUFFALO_DIR, fname))
    else:
        raise ValueError("expected a .zip file or a directory containing the buffalo_l .onnx files")
    return _buffalo_status()
def install_gpen(upload_path: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "GPEN-BFR-256.onnx")
    shutil.copy(upload_path, dest)
    return dest
def install_gpen_from_local_path(local_path: str, dest_dir: str) -> str:
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"path does not exist: {local_path}")
    if not os.path.isfile(local_path):
        raise ValueError("expected a file path, not a directory")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "GPEN-BFR-256.onnx")
    shutil.copy(local_path, dest)
    return dest
def delete_buffalo_l():
    if os.path.isdir(BUFFALO_DIR):
        shutil.rmtree(BUFFALO_DIR)
def delete_gpen(path: Optional[str]):
    if path and os.path.exists(path):
        os.remove(path)