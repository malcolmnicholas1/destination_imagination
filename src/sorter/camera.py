"""Camera input adapters for macOS webcam, PiCamera2, image folders, and synthetic belt simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from pathlib import Path
import cv2
import numpy as np

from .config import CameraConfig


class CameraSource(ABC):
    """Abstract frame provider."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if the stream has ended."""

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""


class MacWebcamSource(CameraSource):
    """Standard OpenCV VideoCapture source for macOS webcams or video files."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap = cv2.VideoCapture(config.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)

    def read(self) -> np.ndarray | None:
        if not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        if frame.shape[1] != self.config.width or frame.shape[0] != self.config.height:
            frame = cv2.resize(frame, (self.config.width, self.config.height))
        return frame

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()


class PiCamera2Source(CameraSource):
    """Raspberry Pi Zero camera source using libcamera / Picamera2."""

    def __init__(self, config: CameraConfig) -> None:
        from picamera2 import Picamera2  # type: ignore[import-not-found]

        self.config = config
        self.picam2 = Picamera2()
        cam_cfg = self.picam2.create_video_configuration(
            main={"size": (config.width, config.height), "format": "RGB888"}
        )
        self.picam2.configure(cam_cfg)
        self.picam2.start()

    def read(self) -> np.ndarray | None:
        rgb = self.picam2.capture_array("main")
        if rgb is None:
            return None
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self.picam2.stop()
        self.picam2.close()


class ImageFolderSource(CameraSource):
    """Iterates through static images in a directory for batch testing and replay."""

    def __init__(self, folder: str | Path, width: int = 320, height: int = 240) -> None:
        self.width = width
        self.height = height
        folder_path = Path(folder)
        self.files = sorted(
            list(folder_path.rglob("*.png"))
            + list(folder_path.rglob("*.jpg"))
            + list(folder_path.rglob("*.jpeg"))
        )
        self.index = 0

    def read(self) -> np.ndarray | None:
        if self.index >= len(self.files):
            return None
        img = cv2.imread(str(self.files[self.index]))
        self.index += 1
        if img is None:
            return None
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height))
        return img

    def close(self) -> None:
        pass


def render_synthetic_shape_frame(
    shape_name: str,
    width: int = 320,
    height: int = 240,
    center_x: int | None = None,
    size: int = 76,
    angle_deg: float = 0.0,
    noise_sigma: float = 4.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Render a synthetic side-profile camera frame of a Cube, Sphere, or Pyramid
    on a conveyor belt inside a controlled lighting tunnel.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Dark matte tunnel background with subtle lighting gradient
    frame = np.full((height, width, 3), 22, dtype=np.uint8)
    belt_y = int(height * 0.84)
    # Draw conveyor belt surface below the object
    cv2.rectangle(frame, (0, belt_y), (width, height), (35, 35, 38), thickness=cv2.FILLED)

    cx = width // 2 if center_x is None else center_x
    base_y = belt_y - 4  # Object sits on top of the belt
    cy = base_y - size // 2

    color = (225, 230, 235)  # Bright illuminated prop

    if shape_name == "cube":
        # Side-profile of a cube sitting on the belt (with optional slight tilt)
        rect_center = (float(cx), float(cy))
        box_pts = cv2.boxPoints((rect_center, (float(size), float(size)), angle_deg))
        pts = np.int32(box_pts)
        cv2.fillPoly(frame, [pts], color)

    elif shape_name == "sphere":
        radius = size // 2
        cv2.circle(frame, (cx, base_y - radius), radius, color, thickness=cv2.FILLED)

    elif shape_name == "pyramid":
        # Triangular side profile sitting on the belt, with slight apex offset for rotation realism
        half_w = size // 2
        apex_offset = int(round(math.sin(math.radians(angle_deg)) * (size * 0.12)))
        pts = np.array(
            [
                [cx + apex_offset, base_y - size],  # Pointed top apex
                [cx - half_w, base_y],              # Bottom-left base corner
                [cx + half_w, base_y],              # Bottom-right base corner
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], color)

    else:
        raise ValueError(f"Unsupported synthetic shape: {shape_name}")

    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, frame.shape).astype(np.float32)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    return frame


class SyntheticConveyorSource(CameraSource):
    """
    Simulates a live moving conveyor belt carrying Cubes, Spheres, and Pyramids
    from left to right across the camera frame.
    """

    def __init__(
        self,
        config: CameraConfig | None = None,
        sequence: list[str] | None = None,
        step_px: int = 12,
        max_objects: int | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or CameraConfig()
        self.sequence = sequence or ["cube", "sphere", "pyramid"]
        self.step_px = step_px
        self.max_objects = max_objects
        self.rng = np.random.default_rng(seed)

        self.obj_idx = 0
        self.current_x = -40
        self.current_size = 76
        self.current_angle = 0.0

    def read(self) -> np.ndarray | None:
        if self.max_objects is not None and self.obj_idx >= self.max_objects:
            return None

        shape_name = self.sequence[self.obj_idx % len(self.sequence)]
        frame = render_synthetic_shape_frame(
            shape_name=shape_name,
            width=self.config.width,
            height=self.config.height,
            center_x=self.current_x,
            size=self.current_size,
            angle_deg=self.current_angle,
            noise_sigma=3.5,
            rng=self.rng,
        )

        self.current_x += self.step_px
        if self.current_x > self.config.width + 50:
            self.obj_idx += 1
            self.current_x = -40
            self.current_size = int(self.rng.integers(64, 88))
            self.current_angle = float(self.rng.uniform(-4.0, 4.0))

        return frame

    def close(self) -> None:
        pass

