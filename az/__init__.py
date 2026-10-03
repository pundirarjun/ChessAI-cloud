"""Clean-run AlphaZero orchestration.

This package intentionally does not import the legacy RL iteration scripts.
Its checkpoint and replay readers require a clean-run manifest and refuse
legacy artifact names.
"""

from az.config import RunConfig

__all__ = ["RunConfig"]
