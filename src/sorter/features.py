"""Scale-invariant geometric feature extraction for 2D/3D shape silhouettes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import cv2
import numpy as np

from .preprocessor import DetectedContour


FEATURE_NAMES: tuple[str, ...] = (
    "extent",
    "taper_ratio",
    "top_to_mid_ratio",
    "circularity",
    "solidity",
    "aspect_ratio",
    "vertices_norm",
)


@dataclass
class ShapeFeatures:
    """Normalized geometric descriptors for a single object contour."""

    extent: float            # Area / Bounding Box Area (~0.92 cube, ~0.78 sphere, ~0.50 pyramid)
    taper_ratio: float       # Top quartile width / Bottom quartile width (~0.33 pyramid, ~1.0 cube/sphere)
    top_to_mid_ratio: float  # Top band width / Middle band width (~0.96 cube, ~0.75 sphere, ~0.35 pyramid)
    circularity: float       # 4 * pi * Area / Perimeter^2 (~0.89 sphere, ~0.75 cube, ~0.60 pyramid)
    solidity: float          # Area / Convex Hull Area
    aspect_ratio: float      # Bounding Box Width / Height
    vertices: int            # Polygon approximation corner count (3 pyramid, 4 cube, >=6 sphere)
    vertices_norm: float     # Clamped vertices / 8.0 for continuous distance calculations

    def to_vector(self) -> np.ndarray:
        """Return the continuous feature vector used for distance/centroid classification."""
        return np.array(
            [
                self.extent,
                self.taper_ratio,
                self.top_to_mid_ratio,
                self.circularity,
                self.solidity,
                self.aspect_ratio,
                self.vertices_norm,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _band_width(mask: np.ndarray, y_start_frac: float, y_end_frac: float) -> float:
    """Compute the mean horizontal span of foreground pixels within a vertical slice."""
    h, w = mask.shape[:2]
    r0 = max(0, int(round(h * y_start_frac)))
    r1 = min(h, max(r0 + 1, int(round(h * y_end_frac))))
    band = mask[r0:r1, :]

    row_widths: list[int] = []
    for row in band:
        xs = np.flatnonzero(row > 0)
        if xs.size > 0:
            row_widths.append(int(xs[-1] - xs[0] + 1))

    if not row_widths:
        return 1.0
    return float(np.mean(row_widths))


def extract_features(detected: DetectedContour) -> ShapeFeatures:
    """Compute scale-invariant geometric features from a DetectedContour."""
    cnt = detected.contour
    _, _, w, h = detected.bbox
    area = max(1.0, detected.area)
    bbox_area = float(max(1, w * h))

    # 1. Extent: fraction of bounding box filled by contour
    extent = float(np.clip(area / bbox_area, 0.0, 1.0))

    # 2. Perimeter & Circularity: 4*pi*Area / Perimeter^2
    perimeter = float(cv2.arcLength(cnt, True))
    if perimeter > 1e-3:
        circularity = float(np.clip((4.0 * math.pi * area) / (perimeter * perimeter), 0.0, 1.0))
    else:
        circularity = 0.0

    # 3. Solidity: Area / Convex Hull Area
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(np.clip(area / max(1.0, hull_area), 0.0, 1.0))

    # 4. Aspect Ratio
    aspect_ratio = float(w) / float(max(1, h))

    # 5. Horizontal width ratios across vertical slices of the contour mask
    top_q_width = _band_width(detected.mask, 0.15, 0.35)
    bot_q_width = _band_width(detected.mask, 0.65, 0.85)
    taper_ratio = float(np.clip(top_q_width / max(1.0, bot_q_width), 0.0, 2.0))

    very_top_width = _band_width(detected.mask, 0.08, 0.22)
    mid_width = _band_width(detected.mask, 0.42, 0.58)
    top_to_mid_ratio = float(np.clip(very_top_width / max(1.0, mid_width), 0.0, 2.0))

    # 6. Polygon vertex approximation
    epsilon = 0.032 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    vertices = int(len(approx))
    vertices_norm = float(min(vertices, 8)) / 8.0

    return ShapeFeatures(
        extent=round(extent, 4),
        taper_ratio=round(taper_ratio, 4),
        top_to_mid_ratio=round(top_to_mid_ratio, 4),
        circularity=round(circularity, 4),
        solidity=round(solidity, 4),
        aspect_ratio=round(aspect_ratio, 4),
        vertices=vertices,
        vertices_norm=round(vertices_norm, 4),
    )

