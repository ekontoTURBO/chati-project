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
ANALYSIS_INTERVAL = 0.33  # ~3 FPS for perception

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

        # Background model for motion detection (MOG2)
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=40, detectShadows=False,
        )

        # Set by the controller when agent is moving/turning
        self.agent_is_moving = False

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
        """Load YOLO and OCR models.

        EasyOCR is loaded first (CPU mode) to avoid cuDNN symbol
        conflicts when loading after YOLO on a background thread.
        """
        # EasyOCR first — CPU mode avoids GPU conflicts
        logger.info("Loading EasyOCR model (CPU)...")
        import easyocr
        self._ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR loaded.")

        # YOLO11 nano — tiny, fast even on CPU
        logger.info("Loading YOLO11 nano model...")
        from ultralytics import YOLO
        self._yolo = YOLO("yolo11n.pt")
        # Force CPU to avoid cuDNN symbol mismatch with CUDA 13.2 driver
        self._yolo_device = "cpu"
        logger.info("YOLO11 nano loaded (CPU mode).")

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

                # 2. Motion-based player detection (only when standing still)
                if not self.agent_is_moving:
                    self._detect_players_motion(frame, state)
                else:
                    # Feed frame to bg model so it adapts, but don't detect
                    self._bg_subtractor.apply(frame, learningRate=0.1)

                # 3. YOLO detection — supplement with object detection
                self._detect_objects(frame, state)

                # 4. OCR — CPU, only if players visible
                if state.players_visible > 0:
                    self._detect_text(frame, state)

                # 5. Check if view is blocked
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

    def _detect_players_motion(self, frame: np.ndarray, state: PerceptionState) -> None:
        """Detect players using background subtraction (motion detection).

        Works with ANY avatar type — anime, furry, robot, anything.
        Anything that moves and is large enough is treated as a player.
        YOLO "person" detections are merged in as a bonus signal.
        """
        h, w = frame.shape[:2]
        min_area = (h * w) * 0.005  # At least 0.5% of frame

        # Apply background subtraction
        fg_mask = self._bg_subtractor.apply(frame)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_players = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            # Normalize center position to 0-1
            cx = (x + cw / 2) / w
            cy = (y + ch / 2) / h

            # Filter: player-sized objects are taller than wide (usually)
            # and not too wide (not the whole background shifting)
            aspect = ch / max(cw, 1)
            if aspect > 0.5 and cw < w * 0.6:
                motion_players.append((cx, cy))

        # Set player count from motion (YOLO will supplement later)
        if motion_players:
            state.players_visible = len(motion_players)
            state.player_positions = motion_players

    def _detect_objects(self, frame: np.ndarray, state: PerceptionState) -> None:
        """Run YOLO11 nano for object detection.

        YOLO "person" detections are merged with motion-based player
        detection. Motion detection is primary (works with any avatar),
        YOLO supplements with additional confidence.
        """
        if self._yolo is None:
            return

        results = self._yolo(frame, conf=YOLO_CONFIDENCE, verbose=False, device=self._yolo_device)

        h, w = frame.shape[:2]
        objects = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = result.names[cls_id]

                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h

                obj = DetectedObject(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                )
                objects.append(obj)

                # YOLO person detection — add to player list if not
                # already detected by motion
                if cls_id == COCO_PERSON_CLASS:
                    is_new = True
                    for px, py in state.player_positions:
                        if abs(px - cx) < 0.15 and abs(py - cy) < 0.15:
                            is_new = False
                            break
                    if is_new:
                        state.players_visible += 1
                        state.player_positions.append((cx, cy))

        state.objects = objects

    def _detect_text(self, frame: np.ndarray, state: PerceptionState) -> None:
        """Run OCR to read chatbox text above player heads.

        Focuses on the center of the frame and filters out VRChat
        UI menu text (Friends, Social, Explore, etc.) which appears
        on the edges.
        """
        if self._ocr is None:
            return

        h, w = frame.shape[:2]
        # Focus on center-upper area where chatbox text appears
        # Skip left 20% (VRChat menu sidebar) and bottom 50%
        roi = frame[:h // 2, int(w * 0.2):int(w * 0.9)]

        try:
            results = self._ocr.readtext(roi, detail=1, paragraph=False)
            texts = []
            for bbox, text, conf in results:
                text = text.strip()
                # Filter: must be 4+ chars, >50% confidence, not UI garbage
                if (len(text) >= 4
                        and conf > 0.5
                        and not self._is_ui_text(text)):
                    texts.append(text)
            state.visible_text = texts
        except Exception as e:
            logger.debug(f"OCR error: {e}")

    @staticmethod
    def _is_ui_text(text: str) -> bool:
        """Check if text is likely VRChat UI rather than player chatbox."""
        ui_keywords = {
            "friends", "social", "explore", "settings", "safety",
            "notifications", "avatar", "world", "menu", "camera",
            "emote", "quick", "options", "search", "favorites",
            "online", "offline", "status", "home", "invite",
            "join", "block", "mute", "report", "exit",
        }
        lower = text.lower().strip()
        # Exact match or very close to a UI keyword
        if lower in ui_keywords:
            return True
        # Very short garbled text is likely OCR noise from UI
        if len(text) < 5 and not text.replace(" ", "").isalpha():
            return True
        return False

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
