"""Image preprocessing, thresholding, and ROI contour extraction."""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .config import PreprocessingConfig


@dataclass
class DetectedContour:
    """Represents a segmented candidate object on the conveyor belt."""

    contour: np.ndarray          # Contour in full-frame coordinates
    roi_contour: np.ndarray      # Contour in ROI-local coordinates
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in full-frame coordinates
    centroid: tuple[int, int]    # (cx, cy) in full-frame coordinates
    area: float
    touches_roi_border: bool     # True if object is partially cut off by left/right ROI edge
    mask: np.ndarray             # Tight binary mask of the object inside its bounding box


class FramePreprocessor:
    """Segments foreground objects from the controlled conveyor backdrop."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def get_roi_bounds(self, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) pixel coordinates for the configured ROI."""
        h, w = frame_shape[:2]
        rx1, ry1, rx2, ry2 = self.config.roi
        return int(w * rx1), int(h * ry1), int(w * rx2), int(h * ry2)

    def process(self, frame: np.ndarray) -> tuple[list[DetectedContour], np.ndarray]:
        """
        Segment the frame and return candidate object contours sorted largest-first,
        along with the full-frame binary mask for visualization.
        """
        x1, y1, x2, y2 = self.get_roi_bounds(frame.shape)
        roi_frame = frame[y1:y2, x1:x2]

        if roi_frame.ndim == 3:
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_frame.copy()

        k = max(3, self.config.blur_kernel | 1)  # Ensure odd kernel size
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        thresh_type = (
            cv2.THRESH_BINARY_INV if self.config.invert_threshold else cv2.THRESH_BINARY
        )

        if self.config.threshold_mode == "adaptive":
            roi_mask = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresh_type,
                21,
                5,
            )
        elif self.config.threshold_mode == "fixed":
            _, roi_mask = cv2.threshold(
                blurred, self.config.fixed_threshold, 255, thresh_type
            )
        else:
            # Default: Otsu's automatic thresholding (ideal for high-contrast tunnel)
            # Check contrast first so an empty dark belt doesn't amplify sensor noise
            min_val, max_val, _, _ = cv2.minMaxLoc(blurred)
            if (max_val - min_val) < 30:
                roi_mask = np.zeros_like(blurred)
            else:
                _, roi_mask = cv2.threshold(
                    blurred, 0, 255, thresh_type | cv2.THRESH_OTSU
                )

        # Morphological open + close to remove noise specks and fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        roi_h, roi_w = roi_mask.shape[:2]
        margin = 3
        detected: list[DetectedContour] = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.config.min_contour_area or area > self.config.max_contour_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            touches_border = (bx <= margin) or ((bx + bw) >= (roi_w - margin))

            moments = cv2.moments(cnt)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"]) + x1
                cy = int(moments["m01"] / moments["m00"]) + y1
            else:
                cx = bx + bw // 2 + x1
                cy = by + bh // 2 + y1

            # Offset contour to full-frame coordinates
            full_cnt = cnt + np.array([[[x1, y1]]], dtype=cnt.dtype)

            # Build clean binary mask isolated to this contour's bounding box
            local_mask = np.zeros((bh, bw), dtype=np.uint8)
            shifted_cnt = cnt - np.array([[[bx, by]]], dtype=cnt.dtype)
            cv2.drawContours(local_mask, [shifted_cnt], -1, 255, thickness=cv2.FILLED)

            detected.append(
                DetectedContour(
                    contour=full_cnt,
                    roi_contour=cnt,
                    bbox=(bx + x1, by + y1, bw, bh),
                    centroid=(cx, cy),
                    area=area,
                    touches_roi_border=touches_border,
                    mask=local_mask,
                )
            )

        detected.sort(key=lambda d: d.area, reverse=True)

        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = roi_mask
        return detected, full_mask

