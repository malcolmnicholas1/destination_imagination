"""Virtual tripwire object tracking and multi-frame majority voting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import time

from .classifier import ClassificationResult
from .config import TrackerConfig
from .preprocessor import DetectedContour


@dataclass
class SortDecision:
    """Finalized multi-frame classification decision emitted once per passing object."""

    object_id: int
    label: str
    confidence: float
    vote_counts: dict[str, int]
    total_frames: int
    timestamp: float = field(default_factory=time.time)


class ConveyorTracker:
    """
    Monitors a central horizontal 'Virtual Tripwire' zone on the conveyor belt.
    Accumulates frame predictions while an object passes through the zone and emits
    a single SortDecision via majority vote once the object completes its pass.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.next_object_id: int = 1
        self.active_votes: list[ClassificationResult] = []
        self.last_centroid_x: int | None = None
        self.frames_since_seen: int = 0
        self.current_object_dispatched: bool = False
        self.last_decision: SortDecision | None = None
        self.history: list[SortDecision] = []

    def get_tripwire_bounds(self, frame_width: int) -> tuple[int, int]:
        """Return (x_min, x_max) pixel coordinates of the central inspection zone."""
        x0_frac, x1_frac = self.config.tripwire_x_range
        return int(frame_width * x0_frac), int(frame_width * x1_frac)

    def is_in_tripwire(self, contour: DetectedContour, frame_width: int) -> bool:
        """Check whether the object's centroid is inside the tripwire zone and not clipped."""
        if contour.touches_roi_border:
            return False
        tx0, tx1 = self.get_tripwire_bounds(frame_width)
        cx, _ = contour.centroid
        return tx0 <= cx <= tx1

    def update(
        self,
        contour: DetectedContour | None,
        prediction: ClassificationResult | None,
        frame_width: int,
    ) -> SortDecision | None:
        """
        Update tracker state for the current frame.
        Returns a `SortDecision` exactly once when an object finishes enough votes.
        """
        if (
            contour is not None
            and prediction is not None
            and self.is_in_tripwire(contour, frame_width)
        ):
            cx, _ = contour.centroid
            # Detect if a brand-new object jumped into the zone far from the previous one
            if (
                self.last_centroid_x is not None
                and abs(cx - self.last_centroid_x) > frame_width * 0.35
            ):
                self._reset_track()

            self.last_centroid_x = cx
            self.frames_since_seen = 0
            self.active_votes.append(prediction)

            # Emit decision as soon as we reach the required vote count (low latency)
            if (
                not self.current_object_dispatched
                and len(self.active_votes) >= self.config.min_votes_required
            ):
                decision = self._compute_majority_decision()
                self.current_object_dispatched = True
                self.last_decision = decision
                self.history.append(decision)
                return decision

            return None

        # No valid object inside the tripwire on this frame
        if self.active_votes:
            self.frames_since_seen += 1
            if self.frames_since_seen >= self.config.max_frames_gap:
                decision = None
                if not self.current_object_dispatched and len(self.active_votes) > 0:
                    decision = self._compute_majority_decision()
                    self.last_decision = decision
                    self.history.append(decision)
                self._reset_track()
                return decision

        return None

    def _compute_majority_decision(self) -> SortDecision:
        counts = Counter(v.label for v in self.active_votes)
        # Prefer known labels over 'unknown' if tied
        best_label, _ = max(
            counts.items(),
            key=lambda kv: (kv[1], 0 if kv[0] == "unknown" else 1),
        )
        matching_confs = [
            v.confidence for v in self.active_votes if v.label == best_label
        ]
        mean_conf = float(sum(matching_confs) / max(1, len(matching_confs)))

        decision = SortDecision(
            object_id=self.next_object_id,
            label=best_label,
            confidence=round(mean_conf, 4),
            vote_counts=dict(counts),
            total_frames=len(self.active_votes),
        )
        self.next_object_id += 1
        return decision

    def _reset_track(self) -> None:
        self.active_votes.clear()
        self.last_centroid_x = None
        self.frames_since_seen = 0
        self.current_object_dispatched = False

