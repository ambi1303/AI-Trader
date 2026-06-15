"""Model registry: versioned save/load with audit metadata.

A registered model bundle contains:
  - <run_dir>/model.joblib     -- the (base, calibrator) pair
  - <run_dir>/metadata.json    -- audit metadata
  - <run_dir>/feature_columns.json -- exact feature ordering used at training time
  - row in `model_runs` table (insert-or-replace)

Audit metadata:
  - run_id (UUID-ish + timestamp)
  - model_name (human-readable)
  - git_sha (HEAD commit if available, else 'no-git')
  - feature_hash (sha256 of canonical feature_columns list, sorted by INSERT
    order which is the column order at training time)
  - trained_from / trained_to (date strings from training window)
  - metrics_json (the validation/test metrics dict)
  - threshold (the production decision threshold)

Why we serialise the *pair* not just the booster:
  - The isotonic calibrator is what we actually call at inference. The base
    booster is kept for reproducibility, debug, and SHAP later.
  - joblib handles sklearn objects natively; xgboost models inside sklearn
    estimators round-trip through joblib correctly in 2.x.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.models.calibration import CalibratedXGB
from src.utils.db import execute, fetch_one
from src.utils.logger import get_logger

log = get_logger("models.registry")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "no-git"


def feature_hash(feature_columns: list[str]) -> str:
    """Stable hash of the *ordered* feature column list.

    Order matters: XGBoost stores features by index, so swapping two
    columns silently changes the model. We hash the ordered list so any
    drift between training and inference is detected at predict time.
    """
    payload = json.dumps(list(feature_columns), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class ModelMetadata:
    run_id: str
    model_name: str
    created_at: str
    git_sha: str
    feature_hash: str
    feature_columns: list[str]
    trained_from: str | None
    trained_to: str | None
    metrics: dict = field(default_factory=dict)
    threshold: float | None = None
    target_return_threshold: float = 0.005
    # Inference contract. `horizon` is the forward-return horizon the model
    # was trained on (1 = next-day, 20 ~ one month). `cross_sectional_features`
    # tells the predictor it must compute per-day universe ranks before
    # scoring (otherwise the xs_* columns the model expects won't exist).
    # Defaults keep older metadata.json files loadable.
    horizon: int = 1
    cross_sectional_features: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _new_run_id(model_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_name}-{ts}-{uuid.uuid4().hex[:8]}"


def save_model(
    cxgb: CalibratedXGB,
    *,
    model_name: str,
    feature_columns: list[str],
    trained_from: str | None,
    trained_to: str | None,
    metrics: dict,
    threshold: float | None,
    target_return_threshold: float = 0.005,
    horizon: int = 1,
    cross_sectional_features: bool = False,
    base_dir: str | Path = "data/models",
) -> ModelMetadata:
    base_path = Path(base_dir).resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    run_id = _new_run_id(model_name)
    run_dir = base_path / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    meta = ModelMetadata(
        run_id=run_id,
        model_name=model_name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        feature_hash=feature_hash(feature_columns),
        feature_columns=list(feature_columns),
        trained_from=trained_from,
        trained_to=trained_to,
        metrics=metrics,
        threshold=threshold,
        target_return_threshold=target_return_threshold,
        horizon=horizon,
        cross_sectional_features=cross_sectional_features,
    )

    joblib.dump(cxgb, run_dir / "model.joblib", compress=3)
    (run_dir / "metadata.json").write_text(
        json.dumps(meta.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "feature_columns.json").write_text(
        json.dumps(meta.feature_columns, indent=2), encoding="utf-8"
    )

    execute(
        """
        INSERT OR REPLACE INTO model_runs
            (run_id, model_name, git_sha, feature_hash,
             trained_from, trained_to, metrics_json, artifact_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta.run_id, meta.model_name, meta.git_sha, meta.feature_hash,
            meta.trained_from, meta.trained_to,
            json.dumps(meta.metrics, default=str),
            str(run_dir),
        ),
    )

    log.info(
        "Saved model run_id={} -> {} | feature_hash={}",
        run_id, run_dir, meta.feature_hash[:12],
    )
    return meta


def load_model(
    run_id: str,
    *,
    base_dir: str | Path = "data/models",
) -> tuple[CalibratedXGB, ModelMetadata]:
    base_path = Path(base_dir).resolve()
    run_dir = base_path / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    cxgb: CalibratedXGB = joblib.load(run_dir / "model.joblib")
    meta_dict = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    meta = ModelMetadata(**meta_dict)

    expected_hash = feature_hash(meta.feature_columns)
    if expected_hash != meta.feature_hash:
        raise ValueError(
            f"Feature-hash mismatch on load (registry corruption?): "
            f"{meta.feature_hash} vs recomputed {expected_hash}"
        )
    return cxgb, meta


def latest_run_id(model_name: str) -> str | None:
    # Tie-break on rowid so two runs registered in the same wall-clock second
    # still resolve deterministically to the most-recently-inserted one.
    row = fetch_one(
        "SELECT run_id FROM model_runs WHERE model_name = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (model_name,),
    )
    return row["run_id"] if row else None


def delete_run(run_id: str, *, base_dir: str | Path = "data/models") -> None:
    base_path = Path(base_dir).resolve()
    run_dir = base_path / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    execute("DELETE FROM model_runs WHERE run_id = ?", (run_id,))
