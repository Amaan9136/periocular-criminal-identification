from typing import List, Tuple, Dict, Optional
import threading
import numpy as np
class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._embeddings: List[np.ndarray] = []
        self._ids: List[str] = []
        self._metadata: List[Dict] = []
        self._norm_matrix: Optional[np.ndarray] = None
        self._lock = threading.Lock()
    def add(self, criminal_id: str, embedding: np.ndarray, metadata: Dict):
        if embedding.shape[0] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.dim}")
        with self._lock:
            self._embeddings.append(embedding.astype(np.float32))
            self._ids.append(criminal_id)
            self._metadata.append(metadata or {})
            self._norm_matrix = None
    def _ensure_cache(self):
        if self._norm_matrix is None and self._embeddings:
            embs = np.stack(self._embeddings, axis=0)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            self._norm_matrix = embs / np.clip(norms, 1e-9, None)
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict]]:
        with self._lock:
            if not self._embeddings:
                return []
            self._ensure_cache()
            query = query.astype(np.float32)
            query_norm = query / max(np.linalg.norm(query), 1e-9)
            sims = self._norm_matrix @ query_norm
            k = min(top_k, len(sims))
            idxs = np.argpartition(-sims, k - 1)[:k]
            idxs = idxs[np.argsort(-sims[idxs])]
            return [(self._ids[i], float(sims[i]), self._metadata[i]) for i in idxs]
    def __len__(self):
        return len(self._embeddings)