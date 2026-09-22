# Periocular / Masked Criminal Identification - Dashboard + API

FastAPI backend and web dashboard for periocular/masked face identification.

- InsightFace `buffalo_l` (SCRFD-10G detector + ArcFace/ResNet50 recognizer) for detection + embeddings.
- SQLite as the default store (`vector_store_backend=sqlite`): a single local file (`data/periocular.db`), built into Python, nothing to install or run. Embeddings are persisted in SQLite and searched with an exact cosine scan over an in-memory numpy matrix. Search logs and dashboard settings live in the same file.
- Optional GPEN-based face restoration ("enhance" mode) on the detected face crop.
- Full web dashboard: overview, register criminal, identify/search, model diagnostics (required/optional,
  install/delete/status), settings.
- `vector_store_backend=in_memory` is kept for throwaway testing (nothing is persisted).

## What's still not implemented, on purpose

- `reconstruction_mode="hallucinate"` (periocular -> full-face DW-KSVD/GAN generation): no maintained
  open-source checkpoint exists to "reuse." Building it is a real research project (train a dictionary +
  a conditional GAN on paired data), not a drop-in module. The API and Models tab expose this mode
  honestly as `not_implemented`; calling it logs a warning and passes the input through unchanged rather
  than fabricating a face.
- `inswapper_128` (present in your local model inventory) is excluded from this pipeline. It's a
  face-swap model, not a reconstruction model - it pastes a separate source identity onto a target
  image. Using it here would hand an investigator a photorealistic image of the wrong person with
  nothing to indicate it was fabricated. The Models tab lists it as `excluded` with this reasoning
  rather than silently dropping it, so it doesn't look like an oversight.

## On "keep 100% exact functionality" vs. this changeset

The original API endpoints (`POST /criminals`, `POST /search`, `GET /health`) keep identical request/response
shapes and behavior. Everything else here - the dashboard, SQLite backend, model management, settings -
is new functionality this changeset explicitly asked for; there was no prior UI or DB layer to preserve
for those parts.

## Code style note

Per your rules, application code carries no comments/docstrings and minimal whitespace. The safety-relevant
context that would normally live in code comments (why `hallucinate` no-ops, why `inswapper_128` is excluded,
) lives here in the README and in the Models tab's description text instead,
since stripping it from the code entirely would leave the next person maintaining this with no way to know
those decisions were deliberate.

## SQLite notes

- The database is `data/periocular.db` (plus `-wal` and `-shm` side files while the app runs). Stop the app before copying it for a backup.
- Run a single uvicorn process (no `--workers N`). Each process keeps its own in-memory copy of the embeddings, so extra workers would not see each other's new registrations until restart.
- To browse or edit the data by hand, open the file in DB Browser for SQLite (MongoDB Compass cannot open SQLite files). Tables: `criminals`, `search_logs`, `app_config`.
- Copy `.env.example` to `.env` to override settings.

## Endpoints

- `POST /criminals` - register a criminal (multipart: `criminal_id`, `name`, `metadata` JSON string, `images[]`).
- `POST /search` - search probe images (multipart: `probe_id`, `config_json`, `images[]`).
  `config_json`: `{"top_k": 10, "reconstruction_mode": "none", "use_periocular_only": false}`
- `GET /health` - status + gallery size.
- `GET /api/stats`, `GET /api/models`, `POST /api/models/{id}/install`, `DELETE /api/models/{id}`,
  `GET /api/config`, `POST /api/config` - power the dashboard.
- `GET /`, `/ui/register`, `/ui/search`, `/ui/models`, `/ui/settings` - the dashboard pages.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://localhost:8000/` for the dashboard.

To use the `buffalo_l` pack you already have cached, no action needed - it's picked up automatically
from `~/.insightface/models/buffalo_l/`. To enable face restoration, either set
`PERIOCULAR_GPEN_MODEL_PATH=/path/to/GPEN-BFR-256.onnx` before starting, or upload the file from the
Models tab once the app is running.

## Environment variables (prefix `PERIOCULAR_`)

`VECTOR_STORE_BACKEND` (`sqlite` | `in_memory`), `SQLITE_PATH`, `INSIGHTFACE_MODEL`, `PREFER_GPU`,
`GPEN_MODEL_PATH`, `TOP_K`, `MIN_DET_SCORE`, `RECONSTRUCTION_MODE_DEFAULT`.

## Honest limitations

- No auth, no rate limiting on the dashboard or API - add both before this touches real investigative data.
- `use_periocular_only` embeds a periocular crop through ArcFace, which wasn't trained for that; expect a
  real accuracy drop versus full-face probes.
- Every candidate is an investigative lead for a human analyst, never a standalone identification.