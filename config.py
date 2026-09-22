from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
def detect_providers(prefer_gpu: bool) -> list:
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        available = ["CPUExecutionProvider"]
    if prefer_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
class Settings(BaseSettings):
    app_name: str = "Periocular Criminal Identification API"
    top_k: int = 10
    embeddings_dim: int = 512
    insightface_model: str = "buffalo_l"
    det_size: int = 640
    prefer_gpu: bool = True
    min_det_score: float = 0.5
    reconstruction_mode_default: str = "none"
    gpen_model_path: Optional[str] = None
    vector_store_backend: str = "sqlite"
    sqlite_path: str = "data/periocular.db"
    static_dir: str = "static"
    models_dir: str = "models"
    model_config = SettingsConfigDict(env_prefix="PERIOCULAR_", env_file=".env", extra="ignore")
settings = Settings()