"""
Real-Time Scene Analyzer
==========================
Continuously analyzes the screen capture feed using:
- YOLO11 nano: Fast person/object detection
- EasyOCR: Read chatbox text above player heads
- Frame differencing: Detect scene changes

Runs in a background thread and produces a PerceptionState
that the agent controller reads for decision-making. This
replaces the slow "ask Gemma 4 what it sees" approach.
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("perception.scene")

# How often to run full analysis (seconds)
ANALYSIS_INTERVAL = 1.0

# Frame difference threshold to consider "scene changed"
CHANGE_THRESHOLD = 15.0

# Minimum YOLO confidence for detections
YOLO_CONFIDENCE = 0.35

# Person class ID in COCO dataset (used by YOLO)
COCO_PERSON_CLASS = 0


@dataclass
class DetectedObject:
    """A single detected object in the scene."""
    label: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (cx, cy) normalized 0-1


@dataclass
class PerceptionState:
    """Complete perception snapshot of the current scene."""
    # Timestamp of this snapshot
    timestamp: float = 0.0

    # Player detection
    players_visible: int = 0
    player_positions: list = field(default_factory=list)  # list of (x, y) normalized

    # Object detection
    objects: list = field(default_factory=list)  # list of DetectedObject

    # OCR — text visible on screen (chatbox messages)
    visible_text: list = field(default_factory=list)  # list of strings

    # Scene change detection
    scene_changed: bool = False
    scene_change_amount: float = 0.0

    # Is the view blocked (close-up wall, obstacle)
    view_blocked: bool = False

    def summary(self) -> str:
        """Human-readable summary for logging and prompts."""
        parts = []
        parts.append(f"Players: {self.players_visible}")
        if self.player_positions:
            positions = [f"({x:.1f},{y:.1f})" for x, y in self.player_positions]
            parts.append(f"at {', '.join(positions)}")
        if self.visible_text:
            parts.append(f"Text: {self.visible_text}")
        if self.scene_changed:
            parts.append(f"Scene changed ({self.scene_change_amount:.0f})")
        if self.view_blocked:
            parts.append("VIEW BLOCKED")
        obj_labels = [o.label for o in self.objects[:5]]
        if obj_labels:
            parts.append(f"Objects: {obj_labels}")
        return " | ".join(parts)

    def for_prompt(self) -> str:
        """Format perception data for inclusion in LLM prompts."""
        lines = []
        lines.append(f"Players visible: {self.players_visible}")
        if self.player_positions:
            for i, (x, y) in enumerate(self.player_positions):
                side = "left" if x < 0.4 else ("right" if x > 0.6 else "center")
                dist = "close" if y > 0.6 else ("far" if y < 0.3 else "medium distance")
                lines.append(f"  Player {i+1}: {side} side, {dist}")
        if self.visible_text:
            lines.append(f"Chatbox text on screen: {', '.join(self.visible_text)}")
        if self.view_blocked:
            lines.append("WARNING: View is blocked by a wall or obstacle")
        if self.objects:
            labels = list(set(o.label for o in self.objects))[:8]
            lines.append(f"Detected objects: {', '.join(labels)}")
        return "\n".join(lines)


class SceneAnalyzer:
    """Real-time scene analysis running in a background thread.

    Combines YOLO detection, OCR, and frame differencing into
    a unified PerceptionState updated every second.
    """

    def __init__(self, video_capture):
        self.video_capture = video_capture
        self._state = PerceptionState()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._prev_frame_gray: Optional[np.ndarray] = None

        # Models (loaded lazily on start)
        self._yolo = None
        self._ocr = None

    def start(self) -> None:
        """Start the scene analyzer background thread."""
        self._load_models()
        self._running = True
        self._thread = threading.Thread(
            target=self._analysis_loop,
            daemon=True,
            name="scene-analyzer",
        )
        self._thread.start()
        logger.info("Scene analyzer started.")

    def stop(self) -> None:
        """Stop the scene analyzer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Scene analyzer stopped.")

    def _load_models(self) -> None:
        """Load YOLO and OCR models."""
        # YOLO11 nano — tiny, fast, ~50MB VRAM
        logger.info("Loading YOLO11 nano model...")
        from ultralytics import YOLO
        self._yolo = YOLO("yolo11n.pt")
        logger.info("YOLO11 nano loaded.")

        # EasyOCR — text detection, ~1.2GB VRAM
        logger.info("Loading EasyOCR model...")
        import easyocr
        self._ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
        logger.info("EasyOCR loaded.")

    def get_state(self) -> PerceptionState:
        """Get the latest perception state (thread-safe)."""
        with self._lock:
            return self._state

    def _analysis_loop(self) -> None:
        """Background loop that continuously analyzes frames."""
        while self._running:
            try:
                frame = self.video_capture.get_latest_frame_raw()
                if frame is None:
                    time.sleep(0.5)
                    continue

                state = PerceptionState(timestamp=time.time())

                # 1. Frame differencing — fast, CPU only
                self._detect_scene_change(frame, state)

                # 2. YOLO detection — GPU, ~5ms for nano
                self._detect_objects(frame, state)

                # 3. OCR — GPU, only if players visible or text likely
                if state.players_visible > 0 or state.scene_changed:
                    self._detect_text(frame, state)

                # 4. Check if view is blocked
                self._check_view_blocked(frame, state)

                # Update shared state
                with self._lock:
                    self._state = state

            except Exception as e:
                logger.error(f"Scene analysis error: {e}")

            time.sleep(ANALYSIS_INTERVAL)

    def _detect_scene_change(self, frame: np.ndarray, state: PerceptionState) -> None:
        """Detect if the scene changed significantly using frame differencing."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_frame_gray is not None:
            diff = cv2.absdiff(self._prev_frame_gray, gray)
            change_amount = diff.mean()
            state.scene_change_amount = change_amount
            state.scene_changed = change_amount > CHANGE_THRESHOLD
        else:
            state.scene_changed = True

        self._prev_frame_gray = gray

    def _detect_objects(self, frame: np.ndarray, state: PerceptionState) -> None:
        """Run YOLO11 nano for object/person detection."""
        if self._yolo is None:
            return

        results = self._yolo(frame, conf=YOLO_CONFIDENCE, verbose=False)

        h, w = frame.shape[:2]
        players = []
        objects = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = result.names[cls_id]

                # Normalize center position to 0-1
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h

                obj = DetectedObject(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                )
                objects.append(obj)

                # COCO "person" class — likely a player avatar
                if cls_id == COCO_PERSON_CLASS:
                    players.append((cx, cy))

        state.players_visible = len(players)
        state.player_positions = players
        state.objects = objects

    def _detect_text(self, frame: np.ndarray, state: PerceptionState) -> None:
        """Run OCR to read chatbox text above player heads."""
        if self._ocr is None:
            return

        # Focus on upper half of frame where chatbox text appears
        h = frame.shape[0]
        upper_half = frame[:h // 2]

        try:
            results = self._ocr.readtext(upper_half, detail=0, paragraph=True)
            # Filter out very short results (noise)
            texts = [t.strip() for t in results if len(t.strip()) > 2]
            state.visible_text = texts
        except Exception as e:
            logger.debug(f"OCR error: {e}")

    def _check_view_blocked(self, frame: np.ndarray, state: PerceptionState) -> None:
        """Check if the view is blocked (wall, close-up obstacle).

        A blocked view typically has low color variance and few edges.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Check edge density — blocked views have very few edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean() / 255.0

        # Check color variance — blocked views are uniform
        color_std = frame.std()

        # Very low edge density + low color variance = blocked
        state.view_blocked = edge_density < 0.02 and color_std < 25
