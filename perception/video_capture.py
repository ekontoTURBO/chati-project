"""
Video Capture Pipeline
=======================
Captures VRChat frames directly from the screen using mss
(screen capture), downscales them, and base64-encodes them
for the multimodal AI model.

No OBS Virtual Camera needed — captures the monitor where
VRChat is running.

Requires:
  - mss installed
  - opencv-python installed
"""

import threading
import time
import base64
import logging
from typing import Optional
from io import BytesIO

import cv2
import numpy as np
import mss

logger = logging.getLogger("perception.video")

# Capture resolution — enough detail to read chatbox text
# above player heads while keeping base64 size reasonable
TARGET_WIDTH = 768
TARGET_HEIGHT = 432

# Capture interval in seconds — 1 FPS is the sweet spot for VRAM
CAPTURE_INTERVAL = 1.0

# JPEG quality for encoding — lower = smaller base64, less detail
JPEG_QUALITY = 70


class VideoCaptureProcessor:
    """Captures and preprocesses video frames via screen capture.

    Runs a background thread that captures the primary monitor
    at a configured interval, downscales frames, and stores the
    latest frame for retrieval by the agent controller.

    Attributes:
        monitor_index: mss monitor index (1 = primary)
        target_width: Downscale width in pixels
        target_height: Downscale height in pixels
        capture_interval: Seconds between frame captures
    """

    def __init__(
        self,
        camera_index: int = 0,
        monitor_index: int = 1,
        target_width: int = TARGET_WIDTH,
        target_height: int = TARGET_HEIGHT,
        capture_interval: float = CAPTURE_INTERVAL,
    ):
        # Monitor index for mss (1 = primary monitor)
        self.monitor_index = monitor_index
        # Target resolution for downscaled frames
        self.target_width = target_width
        self.target_height = target_height
        # Interval between captures in seconds
        self.capture_interval = capture_interval

        # Latest captured frame as base64 JPEG
        self._latest_frame_b64: Optional[str] = None
        # Latest raw frame as numpy array (BGR format)
        self._latest_frame_raw: Optional[np.ndarray] = None
        # Timestamp of the latest frame capture
        self._latest_timestamp: float = 0.0
        # Thread lock for frame access
        self._lock = threading.Lock()
        # Background capture thread
        self._thread: Optional[threading.Thread] = None
        # Flag to control the capture thread
        self._running = False

    def start(self) -> None:
        """Start the video capture background thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="video-capture",
        )
        self._thread.start()
        logger.info(
            f"Video capture started (monitor={self.monitor_index}, "
            f"{self.target_width}x{self.target_height}, "
            f"{1/self.capture_interval:.1f} FPS)"
        )

    def stop(self) -> None:
        """Stop video capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Video capture stopped.")

    def _capture_loop(self) -> None:
        """Background loop that captures screen at the configured interval."""
        with mss.mss() as sct:
            monitor = sct.monitors[self.monitor_index]
            logger.info(
                f"Screen capture: monitor {self.monitor_index} "
                f"({monitor['width']}x{monitor['height']})"
            )

            while self._running:
                try:
                    # Capture the screen
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    # Convert BGRA to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Downscale to target resolution
                    frame_resized = cv2.resize(
                        frame,
                        (self.target_width, self.target_height),
                        interpolation=cv2.INTER_AREA,
                    )

                    # Encode as JPEG for efficient base64
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    _, jpeg_bytes = cv2.imencode(".jpg", frame_resized, encode_params)
                    frame_b64 = base64.b64encode(jpeg_bytes.tobytes()).decode("utf-8")

                    # Store the latest frame (thread-safe)
                    with self._lock:
                        self._latest_frame_b64 = frame_b64
                        self._latest_frame_raw = frame_resized
                        self._latest_timestamp = time.time()

                except Exception as e:
                    logger.error(f"Screen capture error: {e}")

                # Wait for next capture interval
                time.sleep(self.capture_interval)

    def get_latest_frame_b64(self) -> Optional[str]:
        """Get the latest captured frame as a base64 JPEG string."""
        with self._lock:
            return self._latest_frame_b64

    def get_latest_frame_raw(self) -> Optional[np.ndarray]:
        """Get the latest captured frame as a raw numpy array (BGR)."""
        with self._lock:
            return self._latest_frame_raw.copy() if self._latest_frame_raw is not None else None

    @property
    def is_running(self) -> bool:
        """Check if video capture is currently active."""
        return self._running
