#!/usr/bin/env python3
"""Download and verify the face-landmark model used by the evaluator."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

from src.metrics.face_landmarks import MODEL_FILENAME, MODEL_SHA256

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def main() -> None:
    destination = Path("models") / MODEL_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == MODEL_SHA256:
            print(f"Verified existing {destination}")
            return

    with tempfile.NamedTemporaryFile() as temporary:
        with urllib.request.urlopen(MODEL_URL) as response:
            shutil.copyfileobj(response, temporary)
        temporary.flush()
        temporary.seek(0)
        digest = hashlib.sha256(temporary.read()).hexdigest()
        if digest != MODEL_SHA256:
            raise SystemExit(
                f"Checksum mismatch: expected {MODEL_SHA256}, downloaded {digest}"
            )
        temporary.seek(0)
        with destination.open("wb") as output:
            shutil.copyfileobj(temporary, output)
    print(f"Downloaded and verified {destination}")


if __name__ == "__main__":
    main()
