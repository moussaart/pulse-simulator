"""
PULSE Simulation — Custom Exception Hierarchy

All simulation-related errors inherit from SimulationError so they can be
caught uniformly at the top-level simulation loop.  Each exception carries:
  • user_message  – a friendly string suitable for display in the UI
  • details       – an optional dict with technical context for internal logging
"""


class SimulationError(Exception):
    """Base exception for all simulation runtime errors."""

    def __init__(self, user_message: str, details: dict | None = None, *args):
        self.user_message = user_message
        self.details = details or {}
        super().__init__(user_message, *args)

    @classmethod
    def from_exception(cls, exc: Exception, context: str = ""):
        """Wrap an arbitrary exception into a SimulationError."""
        user_msg = _friendly_message(exc, context)
        return cls(
            user_message=user_msg,
            details={
                "original_type": type(exc).__name__,
                "original_message": str(exc),
                "context": context,
            },
        )


class NumericalError(SimulationError):
    """Division by zero, NaN / Inf propagation, overflow / underflow."""
    pass


class ConvergenceError(SimulationError):
    """Algorithm failed to converge or produced an unstable estimate."""
    pass


class InputValidationError(SimulationError):
    """Invalid simulation parameters (negative distances, wrong dimensions, etc.)."""
    pass


class MeasurementError(SimulationError):
    """Sensor / measurement failures (all anchors returning invalid data, etc.)."""
    pass


class MatrixError(SimulationError):
    """Matrix singularity, non-positive-definite covariance, ill-conditioning."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ERROR_PATTERNS = {
    "ZeroDivisionError": "A division by zero occurred",
    "LinAlgError": "A matrix computation failed (singular or ill-conditioned matrix)",
    "OverflowError": "A numerical overflow occurred — values grew too large",
    "FloatingPointError": "A floating-point arithmetic error was detected",
    "ValueError": "An invalid value was encountered",
    "IndexError": "An unexpected data-size mismatch occurred",
    "TypeError": "An unexpected data type was encountered",
    "AttributeError": "A required simulation component was missing or not initialized",
    "KeyError": "A required configuration key was missing",
}


def _friendly_message(exc: Exception, context: str = "") -> str:
    """Convert an arbitrary exception into a human-readable sentence."""
    exc_type = type(exc).__name__
    base = _ERROR_PATTERNS.get(exc_type, f"An unexpected error occurred ({exc_type})")
    if context:
        base = f"{base} during {context}"
    return f"{base}."
