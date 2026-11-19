"""Flexible shape classifier supporting both geometric rules and sample-based calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from .config import ClassifierConfig, PreprocessingConfig
from .features import FEATURE_NAMES, ShapeFeatures, extract_features
from .preprocessor import FramePreprocessor


@dataclass
class ClassificationResult:
    """Single-frame shape prediction result."""

    label: str
    confidence: float
    scores: dict[str, float]
    features: ShapeFeatures


class ShapeClassifier:
    """
    Classifies extracted ShapeFeatures into shape categories.

    Supports three modes:
    - 'rules': Built-in geometric rules for Cube, Sphere, and Pyramid.
    - 'calibrated': Nearest-centroid classifier trained from sample image folders.
    - 'hybrid': Uses the calibrated model if loaded, blended with rule-based priors.
    """

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self.config = config or ClassifierConfig()
        self.calibrated_classes: dict[str, dict[str, list[float]]] = {}
        self._load_model_if_exists()

    def _load_model_if_exists(self) -> None:
        path = Path(self.config.model_path)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self.calibrated_classes = data.get("classes", {})
            except (OSError, json.JSONDecodeError):
                self.calibrated_classes = {}

    def classify(self, features: ShapeFeatures) -> ClassificationResult:
        """Predict the shape label and confidence from extracted features."""
        rule_scores = self._rule_scores(features)
        cal_scores = self._calibrated_scores(features) if self.calibrated_classes else {}

        if self.config.mode == "rules" or not cal_scores:
            final_scores = rule_scores
        elif self.config.mode == "calibrated":
            final_scores = cal_scores
        else:
            # Hybrid mode: combine calibrated scores with rule scores when available
            all_labels = set(cal_scores.keys()) | set(rule_scores.keys())
            final_scores = {}
            for lbl in all_labels:
                if lbl in cal_scores and lbl in rule_scores:
                    final_scores[lbl] = 0.65 * cal_scores[lbl] + 0.35 * rule_scores[lbl]
                elif lbl in cal_scores:
                    final_scores[lbl] = cal_scores[lbl]
                else:
                    final_scores[lbl] = rule_scores[lbl]

        if not final_scores:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                scores={},
                features=features,
            )

        best_label = max(final_scores, key=lambda k: final_scores[k])
        best_score = float(round(final_scores[best_label], 4))

        if best_score < self.config.confidence_threshold:
            return ClassificationResult(
                label="unknown",
                confidence=best_score,
                scores=final_scores,
                features=features,
            )

        return ClassificationResult(
            label=best_label,
            confidence=best_score,
            scores={k: round(v, 4) for k, v in final_scores.items()},
            features=features,
        )

    @staticmethod
    def _rule_scores(f: ShapeFeatures) -> dict[str, float]:
        """
        Compute normalized compatibility scores in [0, 1] for Cube, Sphere, and Pyramid
        based on scale-invariant side-profile geometry.
        """
        # --- Pyramid score ---
        # Pyramids taper toward the top (low taper_ratio & low top_to_mid_ratio),
        # fill ~50% of their bounding box (extent ~0.45-0.65), and have ~3 vertices.
        pyr_taper = float(np.clip((0.72 - f.taper_ratio) / 0.38, 0.0, 1.0))
        pyr_extent = float(np.clip(1.0 - abs(f.extent - 0.52) / 0.22, 0.0, 1.0))
        pyr_top = float(np.clip((0.68 - f.top_to_mid_ratio) / 0.35, 0.0, 1.0))
        pyr_vert = 1.0 if f.vertices == 3 else (0.5 if f.vertices == 4 else 0.0)
        pyramid_score = (
            0.40 * pyr_taper + 0.25 * pyr_extent + 0.20 * pyr_top + 0.15 * pyr_vert
        )

        # --- Cube score ---
        # Cubes fill most of their bounding box (extent > 0.80), have flat tops
        # (top_to_mid_ratio > 0.85, taper_ratio ~ 1.0), and ~4 vertices.
        cube_extent = float(np.clip((f.extent - 0.76) / 0.18, 0.0, 1.0))
        cube_top = float(np.clip((f.top_to_mid_ratio - 0.80) / 0.16, 0.0, 1.0))
        cube_taper = float(np.clip(1.0 - abs(f.taper_ratio - 1.0) / 0.25, 0.0, 1.0))
        cube_vert = 1.0 if f.vertices == 4 else (0.4 if f.vertices == 5 else 0.0)
        cube_score = (
            0.35 * cube_extent + 0.30 * cube_top + 0.20 * cube_taper + 0.15 * cube_vert
        )

        # --- Sphere score ---
        # Spheres have high circularity (> 0.82), medium extent (~0.785),
        # symmetric top/bottom (taper_ratio ~ 1.0), curved top (top_to_mid_ratio ~ 0.72),
        # and >= 6 polygon vertices.
        sph_circ = float(np.clip((f.circularity - 0.76) / 0.14, 0.0, 1.0))
        sph_extent = float(np.clip(1.0 - abs(f.extent - 0.78) / 0.10, 0.0, 1.0))
        sph_top = float(np.clip(1.0 - abs(f.top_to_mid_ratio - 0.73) / 0.15, 0.0, 1.0))
        sph_taper = float(np.clip(1.0 - abs(f.taper_ratio - 1.0) / 0.25, 0.0, 1.0))
        sph_vert = 1.0 if f.vertices >= 6 else (0.5 if f.vertices == 5 else 0.0)
        sphere_score = (
            0.35 * sph_circ
            + 0.20 * sph_extent
            + 0.20 * sph_top
            + 0.10 * sph_taper
            + 0.15 * sph_vert
        )

        return {
            "cube": float(np.clip(cube_score, 0.0, 1.0)),
            "sphere": float(np.clip(sphere_score, 0.0, 1.0)),
            "pyramid": float(np.clip(pyramid_score, 0.0, 1.0)),
        }

    def _calibrated_scores(self, f: ShapeFeatures) -> dict[str, float]:
        """Compute Gaussian kernel similarity to calibrated class centroids."""
        vec = f.to_vector()
        scores: dict[str, float] = {}
        for label, stats in self.calibrated_classes.items():
            mean = np.array(stats["mean"], dtype=np.float64)
            std = np.array(stats["std"], dtype=np.float64)
            # Floor std to prevent division by tiny variance on synthetic data
            safe_std = np.maximum(std, 0.04)
            z = (vec - mean) / safe_std
            rms_z = float(np.sqrt(np.mean(z * z)))
            # Map RMS z-score to a smooth [0, 1] confidence score
            scores[label] = float(np.exp(-0.25 * (rms_z ** 2)))
        return scores


def calibrate_from_directory(
    samples_dir: str | Path,
    output_model_path: str | Path,
    prep_config: PreprocessingConfig | None = None,
) -> dict[str, Any]:
    """
    Scan subdirectories of `samples_dir` (e.g. data/samples/cube, data/samples/sphere, ...),
    extract geometric features from every image, and save class centroids/stds to JSON.
    """
    samples_root = Path(samples_dir)
    preprocessor = FramePreprocessor(prep_config)
    classes_summary: dict[str, dict[str, Any]] = {}

    if not samples_root.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_root}")

    for class_dir in sorted(samples_root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        label = class_dir.name.lower()
        vectors: list[np.ndarray] = []

        image_files = (
            list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
        )
        for img_path in sorted(image_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            detected, _ = preprocessor.process(frame)
            if not detected:
                continue
            feat = extract_features(detected[0])
            vectors.append(feat.to_vector())

        if not vectors:
            continue

        mat = np.vstack(vectors)
        mean_vec = np.mean(mat, axis=0)
        std_vec = np.std(mat, axis=0)

        classes_summary[label] = {
            "count": len(vectors),
            "feature_names": list(FEATURE_NAMES),
            "mean": [round(float(x), 4) for x in mean_vec],
            "std": [round(float(x), 4) for x in std_vec],
        }

    model_payload = {
        "version": 1,
        "feature_names": list(FEATURE_NAMES),
        "classes": classes_summary,
    }

    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

    return model_payload

