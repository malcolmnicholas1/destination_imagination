"""Sorting gate actuator abstractions for macOS simulation and Raspberry Pi Zero GPIO."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import threading

from .config import ActuationConfig
from .tracker import SortDecision

logger = logging.getLogger(__name__)


class SorterActuator(ABC):
    """Base interface for physical or simulated sorting mechanisms."""

    @abstractmethod
    def trigger_sort(self, decision: SortDecision) -> None:
        """Route the classified object to its corresponding bin."""

    @abstractmethod
    def close(self) -> None:
        """Release any hardware resources."""


class MockDisplayActuator(SorterActuator):
    """macOS / development actuator that logs sorting actions and tracks gate angle."""

    def __init__(self, config: ActuationConfig | None = None) -> None:
        self.config = config or ActuationConfig()
        self.current_angle: float = self.config.default_angle
        self.last_sorted_label: str = "none"
        self.sort_events: list[tuple[int, str, float]] = []

    def trigger_sort(self, decision: SortDecision) -> None:
        angle = self.config.shape_angles.get(
            decision.label, self.config.default_angle
        )
        self.current_angle = angle
        self.last_sorted_label = decision.label
        self.sort_events.append((decision.object_id, decision.label, angle))
        logger.info(
            "[ACTUATOR] Object #%d classified as %s (conf=%.2f, votes=%s) -> Gate angle %.1f°",
            decision.object_id,
            decision.label.upper(),
            decision.confidence,
            decision.vote_counts,
            angle,
        )

    def close(self) -> None:
        pass


class PiGPIOActuator(SorterActuator):
    """
    Raspberry Pi Zero hardware-PWM servo actuator using gpiozero + pigpio.
    Uses non-blocking background timers for conveyor travel delay and gate hold.
    """

    def __init__(self, config: ActuationConfig | None = None) -> None:
        self.config = config or ActuationConfig()
        self._servo = None
        self._timers: list[threading.Timer] = []

        try:
            from gpiozero import AngularServo
            from gpiozero.pins.pigpio import PiGPIOFactory

            factory = PiGPIOFactory()
            self._servo = AngularServo(
                self.config.servo_pin,
                min_angle=-90,
                max_angle=90,
                initial_angle=self.config.default_angle,
                pin_factory=factory,
            )
            logger.info(
                "Initialized PiGPIOActuator on BCM pin %d", self.config.servo_pin
            )
        except Exception as exc:
            logger.warning(
                "Hardware servo unavailable (%s); falling back to log-only mode.", exc
            )

    def trigger_sort(self, decision: SortDecision) -> None:
        target_angle = self.config.shape_angles.get(
            decision.label, self.config.default_angle
        )
        t_move = threading.Timer(
            self.config.travel_delay_sec,
            self._set_angle,
            args=(target_angle, decision),
        )
        t_reset = threading.Timer(
            self.config.travel_delay_sec + self.config.hold_duration_sec,
            self._set_angle,
            args=(self.config.default_angle, None),
        )
        t_move.daemon = True
        t_reset.daemon = True
        self._timers.extend([t_move, t_reset])
        t_move.start()
        t_reset.start()

    def _set_angle(self, angle: float, decision: SortDecision | None) -> None:
        if decision is not None:
            logger.info(
                "[PI SERVO] Routing Object #%d (%s) -> %.1f°",
                decision.object_id,
                decision.label.upper(),
                angle,
            )
        if self._servo is not None:
            self._servo.angle = angle

    def close(self) -> None:
        for t in self._timers:
            t.cancel()
        if self._servo is not None:
            self._servo.close()

