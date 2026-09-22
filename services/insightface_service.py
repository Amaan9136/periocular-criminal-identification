import logging
from typing import List, Tuple, Optional
import numpy as np
from insightface.app import FaceAnalysis
from config import settings, detect_providers
logger = logging.getLogger("periocular_api.insightface")
class InsightFaceService:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        providers = detect_providers(settings.prefer_gpu)
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        logger.info("Loading InsightFace pack=%s providers=%s", settings.insightface_model, providers)
        self.app = FaceAnalysis(name=settings.insightface_model, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(settings.det_size, settings.det_size))
    def detect_faces(self, image: np.ndarray):
        if image is None:
            return []
        faces = self.app.get(image)
        return [f for f in faces if float(f.det_score) >= settings.min_det_score]
    def largest_face(self, image: np.ndarray):
        faces = self.detect_faces(image)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces[0]
    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        face = self.largest_face(image)
        if face is None:
            return None
        return face.embedding
    def periocular_crop(self, image: np.ndarray, face) -> Optional[np.ndarray]:
        if face is None or not hasattr(face, "kps") or face.kps is None:
            return None
        kps = face.kps
        left_eye, right_eye, nose = kps[0], kps[1], kps[2]
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        if eye_dist < 1e-3:
            return None
        cx, cy = (left_eye + right_eye) / 2.0
        half_w = eye_dist * 1.3
        top = cy - eye_dist * 0.9
        bottom = min(cy + eye_dist * 0.9, nose[1] + eye_dist * 0.2)
        x1, x2 = int(max(cx - half_w, 0)), int(min(cx + half_w, image.shape[1]))
        y1, y2 = int(max(top, 0)), int(min(bottom, image.shape[0]))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()
    def add_criminal(self, criminal_id: str, embeddings: List[np.ndarray], metadata: dict):
        for emb in embeddings:
            self.vector_store.add(criminal_id, emb, metadata)
    def search(self, probe_embeddings: List[np.ndarray], top_k: int) -> List[Tuple[str, float, dict]]:
        if not probe_embeddings:
            return []
        aggregated = np.mean(probe_embeddings, axis=0)
        return self.vector_store.search(aggregated, top_k)