"""Compatibility entry point for held-out three-way calibration evaluation.

Run: python abyss/calibration_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "ml" / "shadow_confidence"))

from evaluate_calibration import main  # noqa: E402


if __name__ == "__main__":
    main()
