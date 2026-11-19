"""Heads-Up Display (HUD) debug visualizer for macOS and Pi inspection."""

from __future__ import annotations

import cv2
import numpy as np

from .classifier import ClassificationResult
from .preprocessor import DetectedContour
from .tracker import ConveyorTracker, SortDecision


LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "cube": (255, 180, 50),      # Cyan-Blue
    "sphere": (80, 220, 100),    # Green
    "pyramid": (70, 140, 255),   # Orange
    "unknown": (140, 140, 140),  # Gray
}


def draw_debug_overlay(
    frame: np.ndarray,
    roi_bounds: tuple[int, int, int, int],
    tracker: ConveyorTracker,
    contour: DetectedContour | None,
    prediction: ClassificationResult | None,
    last_decision: SortDecision | None,
    fps: float = 0.0,
) -> np.ndarray:
    """Render ROI box, virtual tripwire lines, contour geometry, and classification HUD."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 1. Draw ROI rectangle (subtle gray dashed/thin border)
    rx1, ry1, rx2, ry2 = roi_bounds
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (90, 90, 90), 1)

    # 2. Draw Virtual Tripwire vertical boundaries
    tx0, tx1 = tracker.get_tripwire_bounds(w)
    cv2.line(vis, (tx0, ry1), (tx0, ry2), (0, 200, 200), 1)
    cv2.line(vis, (tx1, ry1), (tx1, ry2), (0, 200, 200), 1)
    cv2.putText(
        vis,
        "INSPECTION ZONE",
        (tx0 + 6, ry1 + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # 3. Draw detected object contour and live feature numbers
    if contour is not None:
        color = (
            LABEL_COLORS.get(prediction.label, (200, 200, 200))
            if prediction is not None
            else (200, 200, 200)
        )
        bx, by, bw, bh = contour.bbox
        cv2.drawContours(vis, [contour.contour], -1, color, 2)
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 1)
        cv2.circle(vis, contour.centroid, 3, (0, 0, 255), -1)

        if prediction is not None:
            f = prediction.features
            tag = f"{prediction.label.upper()} ({prediction.confidence:.2f})"
            cv2.putText(
                vis,
                tag,
                (bx, max(15, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            stats_line = (
                f"ext:{f.extent:.2f} tap:{f.taper_ratio:.2f} "
                f"cir:{f.circularity:.2f} v:{f.vertices}"
            )
            cv2.putText(
                vis,
                stats_line,
                (8, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    # 4. Draw top status bar (FPS & last finalized majority-vote decision)
    cv2.putText(
        vis,
        f"FPS: {fps:.1f}",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )

    if last_decision is not None:
        dec_color = LABEL_COLORS.get(last_decision.label, (255, 255, 255))
        summary = (
            f"SORTED #{last_decision.object_id}: {last_decision.label.upper()} "
            f"({last_decision.confidence:.2f})"
        )
        cv2.putText(
            vis,
            summary,
            (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            dec_color,
            1,
            cv2.LINE_AA,
        )

    return vis

