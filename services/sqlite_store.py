import json
import sqlite3
import threading
import datetime
import os
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
EDITABLE_KEYS = ["top_k", "reconstruction_mode_default", "prefer_gpu", "min_det_score", "gpen_model_path"]
class SqliteVectorStore:
    def __init__(self, path: str, dim: int):
        self.dim = dim
        self.path = path
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute("CREATE TABLE IF NOT EXISTS criminals (id INTEGER PRIMARY KEY AUTOINCREMENT, criminal_id TEXT NOT NULL, embedding BLOB NOT NULL, metadata TEXT NOT NULL DEFAULT '{}')")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_criminal_id ON criminals(criminal_id)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS search_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, probe_id TEXT, reconstruction_mode TEXT, top_candidate TEXT, timestamp TEXT)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS app_config (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()
        self._ids: List[str] = []
        self._metadata: List[Dict] = []
        self._raw: List[np.ndarray] = []
        self._norm_matrix: Optional[np.ndarray] = None
        self._load()
    def _load(self):
        with self._lock:
            rows = self._conn.execute("SELECT criminal_id, embedding, metadata FROM criminals ORDER BY id").fetchall()
            self._ids = [r[0] for r in rows]
            self._raw = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
            self._metadata = [json.loads(r[2]) for r in rows]
            self._norm_matrix = None
    def _ensure_cache(self):
        if self._norm_matrix is None and self._raw:
            embs = np.stack(self._raw, axis=0)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            self._norm_matrix = embs / np.clip(norms, 1e-9, None)
    def add(self, criminal_id: str, embedding: np.ndarray, metadata: Dict):
        if embedding.shape[0] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.dim}")
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        meta = metadata or {}
        with self._lock:
            self._conn.execute("INSERT INTO criminals (criminal_id, embedding, metadata) VALUES (?, ?, ?)", (criminal_id, emb.tobytes(), json.dumps(meta)))
            self._conn.commit()
            self._ids.append(criminal_id)
            self._raw.append(emb)
            self._metadata.append(meta)
            self._norm_matrix = None
    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[str, float, Dict]]:
        with self._lock:
            if not self._raw:
                return []
            self._ensure_cache()
            q = query.astype(np.float32)
            q = q / max(np.linalg.norm(q), 1e-9)
            sims = self._norm_matrix @ q
            k = min(top_k, len(sims))
            idxs = np.argpartition(-sims, k - 1)[:k]
            idxs = idxs[np.argsort(-sims[idxs])]
            return [(self._ids[i], float(sims[i]), self._metadata[i]) for i in idxs]
    def index_status(self) -> str:
        return "exact (numpy cosine)"
    def log_search(self, probe_id: str, mode: str, top: Optional[str]):
        with self._lock:
            self._conn.execute("INSERT INTO search_logs (probe_id, reconstruction_mode, top_candidate, timestamp) VALUES (?, ?, ?, ?)", (probe_id, mode, top, datetime.datetime.utcnow().isoformat()))
            self._conn.commit()
    def recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT probe_id, reconstruction_mode, top_candidate, timestamp FROM search_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"probe_id": r[0], "reconstruction_mode": r[1], "top_candidate": r[2], "timestamp": r[3]} for r in rows]
    def load(self) -> Dict[str, Any]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM app_config").fetchall()
        return {k: json.loads(v) for k, v in rows if k in EDITABLE_KEYS}
    def save(self, values: Dict[str, Any]):
        clean = {k: v for k, v in values.items() if k in EDITABLE_KEYS}
        with self._lock:
            for k, v in clean.items():
                self._conn.execute("INSERT INTO app_config (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, json.dumps(v)))
            self._conn.commit()
        return clean
    def close(self):
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error:
                pass
    def __len__(self):
        with self._lock:
            return len(self._ids)