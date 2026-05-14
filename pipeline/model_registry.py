"""
Model registry for version control, governance, and champion/challenger tracking.

Provides:
- Versioned model storage with full metadata
- Champion/challenger model comparison
- Automatic promotion based on validation metrics
- Lineage tracking (data hash → model version → deployment)
- Compliance-ready audit trail
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from xgboost import XGBClassifier

from pipeline.config import REGISTRY_DIR, MonitoringConfig

logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a single versioned model with metadata."""

    def __init__(
        self,
        version: str,
        model: XGBClassifier,
        scaler: Any,
        metadata: Dict[str, Any],
        path: Optional[Path] = None,
    ):
        self.version = version
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
        self.path = path

    @property
    def accuracy(self) -> float:
        return self.metadata.get("aggregate_accuracy", 0.0)

    @property
    def f1(self) -> float:
        return self.metadata.get("aggregate_f1", 0.0)

    @property
    def mcc(self) -> float:
        return self.metadata.get("aggregate_mcc", 0.0)

    @property
    def created_at(self) -> str:
        return self.metadata.get("timestamp", "unknown")


class ModelRegistry:
    """
    Production model registry with governance controls.

    Directory structure:
        model_registry/
        ├── champion/           # Currently deployed model
        │   ├── model.json
        │   ├── scaler.pkl
        │   └── metadata.json
        ├── v001/               # Historical versions
        ├── v002/
        └── registry.json       # Registry index
    """

    def __init__(self, registry_dir: Optional[Path] = None):
        self.registry_dir = registry_dir or REGISTRY_DIR
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "registry.json"
        self._index = self._load_index()

    # ── Public API ────────────────────────────────────────────────────────

    def register_model(
        self,
        model: XGBClassifier,
        scaler: Any,
        training_summary: Dict[str, Any],
        feature_names: List[str],
        data_hash: str,
    ) -> str:
        """
        Register a new model version.

        Returns:
            Version string (e.g., "v003")
        """
        version = self._next_version()
        version_dir = self.registry_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save artifacts
        model.save_model(str(version_dir / "model.json"))
        joblib.dump(scaler, version_dir / "scaler.pkl")

        # Build metadata
        metadata = {
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "data_hash": data_hash,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "aggregate_accuracy": training_summary.get("aggregate_accuracy"),
            "aggregate_f1": training_summary.get("aggregate_f1"),
            "aggregate_mcc": training_summary.get("aggregate_mcc"),
            "per_fold_accuracy": training_summary.get("per_fold_accuracy"),
            "classification_report": training_summary.get("classification_report"),
            "config": training_summary.get("config"),
            "status": "registered",  # registered → validated → champion → retired
        }

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update index
        self._index["versions"][version] = {
            "timestamp": metadata["timestamp"],
            "accuracy": metadata["aggregate_accuracy"],
            "f1": metadata["aggregate_f1"],
            "status": metadata["status"],
            "data_hash": data_hash,
        }
        self._save_index()

        logger.info(
            f"Registered model {version}: "
            f"accuracy={metadata['aggregate_accuracy']:.4f}, "
            f"f1={metadata['aggregate_f1']:.4f}"
        )

        return version

    def validate_model(
        self,
        version: str,
        monitoring_config: Optional[MonitoringConfig] = None,
    ) -> Dict[str, Any]:
        """
        Validate a registered model against minimum thresholds.

        Returns:
            Validation report with pass/fail status.
        """
        config = monitoring_config or MonitoringConfig()
        model_version = self.load_version(version)

        checks = {
            "accuracy_check": {
                "passed": model_version.accuracy >= config.accuracy_min,
                "value": model_version.accuracy,
                "threshold": config.accuracy_min,
            },
            "f1_check": {
                "passed": model_version.f1 >= 0.50,
                "value": model_version.f1,
                "threshold": 0.50,
            },
        }

        all_passed = all(c["passed"] for c in checks.values())

        if all_passed:
            self._index["versions"][version]["status"] = "validated"
            self._save_index()
            logger.info(f"Model {version} passed validation")
        else:
            failed = [k for k, v in checks.items() if not v["passed"]]
            logger.warning(f"Model {version} failed validation: {failed}")

        return {
            "version": version,
            "passed": all_passed,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def promote_to_champion(self, version: str) -> None:
        """
        Promote a validated model to champion (production).

        The previous champion is retired but preserved.
        """
        # Verify the model is validated
        status = self._index["versions"].get(version, {}).get("status")
        if status not in ("validated", "champion"):
            raise ValueError(
                f"Cannot promote {version}: status is '{status}', "
                f"must be 'validated'"
            )

        # Retire current champion
        current_champion = self._index.get("champion")
        if current_champion:
            self._index["versions"][current_champion]["status"] = "retired"
            logger.info(f"Retired previous champion: {current_champion}")

        # Copy to champion directory
        champion_dir = self.registry_dir / "champion"
        if champion_dir.exists():
            shutil.rmtree(champion_dir)

        source_dir = self.registry_dir / version
        shutil.copytree(source_dir, champion_dir)

        # Update index
        self._index["champion"] = version
        self._index["versions"][version]["status"] = "champion"
        self._index["promotion_history"] = self._index.get(
            "promotion_history", []
        )
        self._index["promotion_history"].append({
            "version": version,
            "promoted_at": datetime.utcnow().isoformat(),
            "previous_champion": current_champion,
        })
        self._save_index()

        logger.info(f"Promoted {version} to champion")

    def load_champion(self) -> Optional[ModelVersion]:
        """Load the current champion model."""
        champion = self._index.get("champion")
        if not champion:
            logger.warning("No champion model registered")
            return None
        return self.load_version("champion")

    def load_version(self, version: str) -> ModelVersion:
        """Load a specific model version."""
        version_dir = self.registry_dir / version

        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found")

        model = XGBClassifier()
        model.load_model(str(version_dir / "model.json"))
        scaler = joblib.load(version_dir / "scaler.pkl")

        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)

        return ModelVersion(version, model, scaler, metadata, version_dir)

    def list_versions(self) -> List[Dict]:
        """List all registered model versions."""
        versions = []
        for v, info in sorted(self._index.get("versions", {}).items()):
            versions.append({"version": v, **info})
        return versions

    def get_champion_version(self) -> Optional[str]:
        """Get the current champion version string."""
        return self._index.get("champion")

    # ── Internal ──────────────────────────────────────────────────────────

    def _next_version(self) -> str:
        """Generate the next version string."""
        versions = list(self._index.get("versions", {}).keys())
        if not versions:
            return "v001"
        nums = [int(v.lstrip("v")) for v in versions if v.startswith("v")]
        return f"v{max(nums) + 1:03d}"

    def _load_index(self) -> Dict:
        """Load the registry index."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {"versions": {}, "champion": None, "promotion_history": []}

    def _save_index(self) -> None:
        """Save the registry index."""
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)
