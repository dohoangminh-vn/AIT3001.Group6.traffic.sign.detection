"""Helpers for readable, filesystem-safe run names."""

import re
from datetime import datetime
from uuid import uuid4


def make_run_name(label=None, default_prefix="yolo26"):
    """Create a unique run name, optionally prefixed by a user label."""
    if label is None or not str(label).strip():
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(default_prefix)).strip(" .-") or "run"
    else:
        prefix = re.sub(r"\s+", "-", str(label).strip())
        prefix = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", prefix).strip(" .-")
        if not prefix:
            raise ValueError("tracking.name must contain at least one usable character")
    return f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
