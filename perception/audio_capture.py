"""
Audio Capture Pipeline with Whisper STT
========================================
Captures VRChat audio via WASAPI loopback on "CABLE In 16ch",
detects speech using energy-based VAD, then transcribes with
faster-whisper for accurate speech-to-text.

The agent receives text transcriptions instead of raw audio,
which Gemma 4 handles much better than raw waveforms.
"""

import threading
import queue
import time
import logging
from typing import Optional

import numpy as np
import pyaudiowpatch as pyaudio

logger = logging.getLogger("perception.audio")

# Duration of each audio chunk in seconds
CHUNK_DURATION = 2.0

# Minimum audio energy to consider a chunk as containing speech
SILENCE_THRESHOLD = 0.005

# Seconds of silence after speech to consider utterance complete
SILENCE_AFTER_SPEECH = 1.5


class AudioCaptureProcessor:
    """Captures VRChat audio and transcribes speech with Whisper.

    Uses pyaudiowpatch for WASAPI loopback capture from the
    CABLE In 16ch virtual cable, then faster-whisper for STT.
    Outputs text transcriptions instead of raw audio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = CHUNK_DURATION,
        device_index: Optional[int] = None,
        whisper_model: str = "base",
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self._device_index = device_index
        self._whisper_model_name = whisper_model

        # Queue of transcribed text chunks
        self._chunk_queue: queue.Queue[dict] = queue.Queue(maxsize=10)
        # Raw audio buffer for accumulation
        self._buffer: list[np.ndarray] = []
        # Speech state tracking
        self._is_speaking = False
        self._speech_buffer: list[np.ndarray] = []
        self._silence_start: float = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._whisper = None
        # Flag to mute capture while TTS is playing
        self.muted = False

    def _find_loopback_device(self) -> dict:
        """Find the best loopback device for capturing VRChat audio.

        Priority:
        1. Default output device loopback (headphones) — you hear VRChat
           AND the agent captures it via loopback
        2. CABLE In 16ch loopback — fallback if no default output
        """
        # Look for Realtek speakers loopback (main audio output)
        for i in range(self._pa.get_device_count()):
            d = self._pa.get_device_info_by_index(i)
            if (d.get("isLoopbackDevice", False)
                    and "speakers (realtek" in d["name"].lower()):
                logger.info(
                    f"Using Realtek loopback: [{i}] {d['name']} "
                    f"({d['maxInputChannels']}ch, {int(d['defaultSampleRate'])}Hz)"
                )
                return d

        # Fallback: CABLE In 16ch loopback
        for i in range(self._pa.get_device_count()):
            d = self._pa.get_device_info_by_index(i)
            if (d.get("isLoopbackDevice", False)
                    and "cable in 16ch" in d["name"].lower()):
                logger.info(
                    f"Using WASAPI loopback (cable): [{i}] {d['name']}"
                )
                return d

        raise RuntimeError("No WASAPI loopback device found for audio capture.")

    def _load_whisper(self):
        """Load the faster-whisper model."""
        from faster_whisper import WhisperModel
        logger.info(f"Loading Whisper '{self._whisper_model_name}' model...")
        self._whisper = WhisperModel(
            self._whisper_model_name,
            device="cuda",
            compute_type="float16",
        )
        logger.info("Whisper model loaded.")

    def start(self) -> None:
        """Start audio capture and Whisper STT."""
        self._pa = pyaudio.PyAudio()

        if self._device_index is None:
            device_info = self._find_loopback_device()
        else:
            device_info = self._pa.get_device_info_by_index(self._device_index)

        self._device_index = int(device_info["index"])

        # Load Whisper model
        self._load_whisper()

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(device_info,),
            daemon=True,
            name="audio-capture",
        )
        self._thread.start()
        logger.info("Audio capture + Whisper STT started.")

    def stop(self) -> None:
        """Stop audio capture."""
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Audio capture stopped.")

    def _capture_loop(self, device_info: dict) -> None:
        """Background loop: capture audio, detect speech, transcribe."""
        native_rate = int(device_info["defaultSampleRate"])
        channels = int(device_info["maxInputChannels"])

        logger.info(f"Opening loopback stream: {channels}ch, {native_rate}Hz")

        try:
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=native_rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=int(native_rate * 0.1),
            )

            while self._running:
                try:
                    data = self._stream.read(
                        int(native_rate * 0.1),
                        exception_on_overflow=False,
                    )
                    audio = np.frombuffer(data, dtype=np.float32)
                    if channels > 1:
                        audio = audio.reshape(-1, channels).mean(axis=1)

                    # Skip if muted (TTS is playing)
                    if self.muted:
                        continue

                    # Check energy
                    energy = np.sqrt(np.mean(audio ** 2))

                    if energy > SILENCE_THRESHOLD:
                        # Speech detected
                        if not self._is_speaking:
                            self._is_speaking = True
                            logger.info("* Speech detected")
                        self._speech_buffer.append(audio)
                        self._silence_start = 0.0
                    elif self._is_speaking:
                        # Silence after speech
                        if self._silence_start == 0.0:
                            self._silence_start = time.time()
                        self._speech_buffer.append(audio)

                        # Check if silence duration exceeded threshold
                        if time.time() - self._silence_start > SILENCE_AFTER_SPEECH:
                            # Utterance complete — transcribe it
                            self._transcribe_speech(native_rate)
                            self._is_speaking = False
                            self._speech_buffer.clear()
                            self._silence_start = 0.0

                except IOError as e:
                    logger.warning(f"Audio read warning: {e}")

        except Exception as e:
            logger.error(f"Audio capture error: {e}")

    def _transcribe_speech(self, native_rate: int) -> None:
        """Transcribe accumulated speech buffer with Whisper."""
        if not self._speech_buffer or not self._whisper:
            return

        # Concatenate all speech audio
        speech = np.concatenate(self._speech_buffer)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if native_rate != 16000:
            num_samples = int(len(speech) * 16000 / native_rate)
            indices = np.linspace(0, len(speech) - 1, num_samples)
            speech = np.interp(indices, np.arange(len(speech)), speech).astype(np.float32)

        duration = len(speech) / 16000
        if duration < 0.5:
            # Too short to be meaningful speech
            return

        logger.info(f"* Transcribing {duration:.1f}s of speech...")

        try:
            segments, info = self._whisper.transcribe(
                speech,
                language="en",
                beam_size=3,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

            if text and len(text) > 1:
                logger.info(f"* Heard: \"{text}\"")
                self._chunk_queue.put({
                    "text": text,
                    "duration": duration,
                    "timestamp": time.time(),
                })
            else:
                logger.debug("* Transcription empty, skipping")

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")

    def get_latest_chunk(self) -> Optional[dict]:
        """Get the most recent transcription."""
        try:
            return self._chunk_queue.get_nowait()
        except queue.Empty:
            return None

    def get_all_chunks(self) -> list[dict]:
        """Drain all available transcriptions."""
        chunks = []
        while not self._chunk_queue.empty():
            try:
                chunks.append(self._chunk_queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    @property
    def is_running(self) -> bool:
        return self._running
