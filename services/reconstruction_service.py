import logging
import os
from typing import List, Optional
import numpy as np
import cv2
logger = logging.getLogger("periocular_api.reconstruction")
class ReconstructionService:
    def __init__(self, gpen_model_path: Optional[str] = None):
        self.gpen_session = None
        self.gpen_input_name = None
        if gpen_model_path and os.path.exists(gpen_model_path):
            try:
                import onnxruntime as ort
                from config import detect_providers, settings
                providers = detect_providers(settings.prefer_gpu)
                self.gpen_session = ort.InferenceSession(gpen_model_path, providers=providers)
                self.gpen_input_name = self.gpen_session.get_inputs()[0].name
                logger.info("Loaded GPEN restoration model from %s", gpen_model_path)
            except Exception:
                logger.exception("Failed to load GPEN model at %s; 'enhance' mode will no-op", gpen_model_path)
                self.gpen_session = None
        elif gpen_model_path:
            logger.warning("PERIOCULAR_GPEN_MODEL_PATH=%s does not exist; 'enhance' mode will no-op", gpen_model_path)
    def _gpen_restore(self, face_crop: np.ndarray) -> np.ndarray:
        if self.gpen_session is None:
            return face_crop
        size = 256
        img = cv2.resize(face_crop, (size, size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5
        inp = np.transpose(img, (2, 0, 1))[None, ...]
        try:
            out = self.gpen_session.run(None, {self.gpen_input_name: inp})[0]
        except Exception:
            logger.exception("GPEN inference failed; returning original crop")
            return face_crop
        out = np.clip((out[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return cv2.resize(out, (face_crop.shape[1], face_crop.shape[0]))
    def reconstruct(self, image: np.ndarray, mode: str, face_bbox=None) -> np.ndarray:
        if mode == "none" or image is None:
            return image
        if mode == "hallucinate":
            logger.warning("reconstruction_mode='hallucinate' requested but not implemented; returning input unchanged")
            return image
        if mode == "enhance":
            if self.gpen_session is None:
                return image
            if face_bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in face_bbox]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
                if x2 <= x1 or y2 <= y1:
                    return image
                restored_crop = self._gpen_restore(image[y1:y2, x1:x2])
                out = image.copy()
                out[y1:y2, x1:x2] = restored_crop
                return out
            return self._gpen_restore(image)
        logger.warning("Unknown reconstruction_mode=%r; returning input unchanged", mode)
        return image
    def reconstruct_faces(self, images: List[np.ndarray], mode: str, bboxes: Optional[List] = None) -> List[np.ndarray]:
        bboxes = bboxes or [None] * len(images)
        return [self.reconstruct(img, mode, bbox) for img, bbox in zip(images, bboxes)]