"""
PULSE Simulation — Centralised Error Handler

SimulationErrorHandler is the single point of contact for all runtime errors.
It:
  1. Logs technical details via Python logging.
  2. Emits a Qt signal so the UI can display an overlay.
  3. Provides numerical / matrix health-check utilities.
  4. Supports error-state reset for the Restart workflow.
"""

import logging
import traceback
import numpy as np
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal

from src.core.exceptions import (
    SimulationError,
    NumericalError,
    ConvergenceError,
    MatrixError,
)

logger = logging.getLogger("pulse.error_handler")


class SimulationErrorHandler(QObject):
    """Detects, logs, and communicates simulation errors to the UI."""

    # Signal payload: (category: str, user_message: str, technical_details: str)
    error_occurred = pyqtSignal(str, str, str)

    # Maximum number of log entries kept in memory
    _MAX_LOG = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self._error_log: list[dict] = []
        self._has_active_error = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def has_active_error(self) -> bool:
        return self._has_active_error

    def handle_error(self, error: SimulationError | Exception, context: str = ""):
        """Process an error: log it and emit a signal for the UI overlay."""
        if self._has_active_error:
            # Avoid cascading popups — just log silently
            self._log_entry(error, context, suppressed=True)
            return

        # Wrap non-SimulationError exceptions
        if not isinstance(error, SimulationError):
            error = SimulationError.from_exception(error, context)

        self._has_active_error = True
        self._log_entry(error, context)

        # Build technical details string for the optional collapsible section
        tech = self._format_technical_details(error)

        # Determine category label
        category = self._categorise(error)

        logger.error("Simulation error [%s]: %s", category, error.user_message)
        logger.debug("Technical details:\n%s", tech)

        # Notify UI
        self.error_occurred.emit(category, error.user_message, tech)

    def clear_error(self):
        """Reset the error state so the simulation can restart."""
        self._has_active_error = False

    def get_error_log(self) -> list[dict]:
        """Return a copy of the internal error log."""
        return list(self._error_log)

    # ------------------------------------------------------------------
    # Numerical health checks  (call from hot paths)
    # ------------------------------------------------------------------

    @staticmethod
    def check_numerical_health(value, label: str = "value"):
        """
        Validate that *value* (scalar or array) is finite.
        Raises NumericalError if NaN or Inf is found.
        """
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)):
            has_nan = bool(np.any(np.isnan(arr)))
            has_inf = bool(np.any(np.isinf(arr)))
            parts = []
            if has_nan:
                parts.append("NaN")
            if has_inf:
                parts.append("Infinity")
            problem = " and ".join(parts)
            raise NumericalError(
                user_message=(
                    f"The simulation produced invalid numbers ({problem}) "
                    f"in the {label}. This usually means the algorithm "
                    f"has become numerically unstable."
                ),
                details={"label": label, "has_nan": has_nan, "has_inf": has_inf},
            )

    @staticmethod
    def check_matrix_health(matrix, label: str = "matrix"):
        """
        Validate that a matrix is finite and reasonably conditioned.
        Raises MatrixError on failure.
        """
        arr = np.asarray(matrix, dtype=float)

        # Finiteness
        if not np.all(np.isfinite(arr)):
            raise MatrixError(
                user_message=(
                    f"The {label} matrix contains invalid numbers (NaN or Infinity). "
                    f"The algorithm state has become corrupted."
                ),
                details={"label": label},
            )

        # Symmetry + positive-definiteness (only for square matrices)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            try:
                eigenvalues = np.linalg.eigvalsh(arr)
                if np.any(eigenvalues < -1e-6):
                    raise MatrixError(
                        user_message=(
                            f"The {label} matrix is no longer positive-definite, "
                            f"which means the estimation uncertainty has become invalid. "
                            f"The filter needs to be restarted."
                        ),
                        details={"label": label, "min_eigenvalue": float(np.min(eigenvalues))},
                    )
            except np.linalg.LinAlgError:
                raise MatrixError(
                    user_message=f"The {label} matrix could not be analysed (singular matrix).",
                    details={"label": label},
                )

    @staticmethod
    def check_divergence(position, bounds: float = 1e4, label: str = "position"):
        """
        Check if a position estimate has diverged beyond reasonable bounds.
        Raises ConvergenceError if so.
        """
        x, y = float(position[0]), float(position[1])
        if abs(x) > bounds or abs(y) > bounds:
            raise ConvergenceError(
                user_message=(
                    f"The estimated {label} has diverged to extreme values "
                    f"({x:.1f}, {y:.1f}). The algorithm is not converging. "
                    f"Try restarting the simulation or changing algorithm parameters."
                ),
                details={"label": label, "x": x, "y": y, "bounds": bounds},
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _categorise(error: SimulationError) -> str:
        _map = {
            NumericalError: "Numerical Error",
            ConvergenceError: "Convergence Error",
            MatrixError: "Matrix Error",
        }
        for cls, label in _map.items():
            if isinstance(error, cls):
                return label
        return "Simulation Error"

    @staticmethod
    def _format_technical_details(error: SimulationError) -> str:
        lines = []
        if error.details:
            for k, v in error.details.items():
                lines.append(f"  {k}: {v}")
        # Include the original traceback if it was a wrapped exception
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        lines.append("")
        lines.append("Traceback (most recent call last):")
        lines.extend(["  " + l.rstrip() for l in tb])
        return "\n".join(lines)

    def _log_entry(self, error, context: str, suppressed: bool = False):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "category": self._categorise(error) if isinstance(error, SimulationError) else "Unknown",
            "message": getattr(error, "user_message", str(error)),
            "context": context,
            "suppressed": suppressed,
        }
        self._error_log.append(entry)
        if len(self._error_log) > self._MAX_LOG:
            self._error_log.pop(0)
