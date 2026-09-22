from typing import List, Optional, Dict, Any
from pydantic import BaseModel
class CriminalResponse(BaseModel):
    criminal_id: str
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    image_urls: List[str]
    images_registered: int
    images_rejected: int  # e.g. no face detected
class SearchConfig(BaseModel):
    top_k: int = 10
    reconstruction_mode: str = "none"  # "none" | "enhance" | "hallucinate"
    use_periocular_only: bool = False  # embed only the periocular crop, not the full detected face
class ProbeReconstruction(BaseModel):
    index: int
    face_detected: bool
    image_url: Optional[str] = None
class CandidateFace(BaseModel):
    criminal_id: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None
class SearchResponse(BaseModel):
    probe_id: str
    reconstruction_mode: str
    probe_reconstructions: List[ProbeReconstruction]
    candidates: List[CandidateFace]
    warnings: List[str] = []
class ModelInfo(BaseModel):
    id: str
    name: str
    required: bool
    status: str
    path: str
    installable: bool
    install_methods: List[str] = []
    download_url: Optional[str] = None
    description: str
class RuntimeConfigUpdate(BaseModel):
    top_k: Optional[int] = None
    reconstruction_mode_default: Optional[str] = None
    prefer_gpu: Optional[bool] = None
    min_det_score: Optional[float] = None
    gpen_model_path: Optional[str] = None