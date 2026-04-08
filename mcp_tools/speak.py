"""
MCP Tool: speak(text)
======================
Generates speech using Piper TTS and outputs audio to
VB-Audio Cable virtual microphone for VRChat.

Requires:
  - piper-tts installed (pip install piper-tts)
  - A Piper ONNX voice model downloaded
  - VB-Audio Cable installed and TTSAudioRouter running
"""

import io
import os
import wave
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mcp.speak")

# Directory where Piper voice models are stored
MODELS_DIR = Path(__file__).parent.parent / "models" / "piper"

# Default voice model — English US, medium quality, good speed/quality balance
DEFAULT_MODEL = "en_US-lessac-medium.onnx"


class SpeakTool:
    """MCP tool that converts text to speech via Piper TTS.

    The generated audio is routed through the TTSAudioRouter
    to VB-Audio Cable, which VRChat picks up as microphone input.

    Attributes:
        model_path: Path to the Piper ONNX voice model
        tts_router: TTSAudioRouter instance for audio output
    """

    def __init__(self, tts_router, model_path: Optional[str] = None):
        # Reference to the TTSAudioRouter that outputs to VB-Audio Cable
        self.tts_router = tts_router
        # Path to the Piper voice model (.onnx file)
        self.model_path = model_path or str(MODELS_DIR / DEFAULT_MODEL)
        # Lazy-loaded Piper voice instance (loaded on first use)
        self._voice = None

    def _load_voice(self):
        """Lazy-load the Piper TTS voice model.

        Deferred loading to avoid slow startup. The model is loaded
        once and reused for all subsequent speak() calls.
        """
        if self._voice is not None:
            return

        try:
            from piper import PiperVoice
        except ImportError:
            raise RuntimeError(
                "piper-tts not installed. Run: pip install piper-tts"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Piper voice model not found: {self.model_path}\n"
                f"Download from: https://github.com/rhasspy/piper/releases\n"
                f"Place .onnx + .json files in: {MODELS_DIR}"
            )

        logger.info(f"Loading Piper voice model: {self.model_path}")
        self._voice = PiperVoice.load(self.model_path)
        logger.info(
            f"Voice loaded. Sample rate: {self._voice.config.sample_rate} Hz"
        )

    def speak(self, text: str) -> dict:
        """Generate speech from text and play through virtual mic.

        This is the MCP tool entry point. It synthesizes the text
        using Piper TTS and streams the audio to VB-Audio Cable.

        Args:
            text: The text to speak aloud in VRChat

        Returns:
            Status dict with 'success' and 'text' fields
        """
        if not text or not text.strip():
            return {"success": False, "error": "Empty text provided"}

        self._load_voice()

        logger.info(f"Speaking: {text[:80]}...")

        try:
            # Synthesize audio — Piper returns raw PCM bytes
            # Use a BytesIO buffer to collect the full audio
            audio_buffer = io.BytesIO()

            # Create a WAV writer to properly frame the PCM data
            with wave.open(audio_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self._voice.config.sample_rate)

                # Synthesize and write to buffer
                self._voice.synthesize(text, wav_file)

            # Extract raw PCM from WAV (skip header)
            audio_buffer.seek(44)  # WAV header is 44 bytes
            raw_pcm = audio_buffer.read()

            # Update the router's sample rate to match this voice
            self.tts_router.sample_rate = self._voice.config.sample_rate

            # Send to VB-Audio Cable via the TTS router
            self.tts_router.play(raw_pcm)

            return {"success": True, "text": text}

        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return {"success": False, "error": str(e)}

    @property
    def tool_schema(self) -> dict:
        """Return the MCP tool definition for function calling.

        This schema is included in the system prompt so the model
        knows how to call this tool.
        """
        return {
            "name": "speak",
            "description": (
                "Speak text aloud in VRChat using text-to-speech. "
                "Use this to verbally respond to players."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak aloud",
                    }
                },
                "required": ["text"],
            },
        }
