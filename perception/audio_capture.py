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

# Minimum audio energy to consider speech (higher = only nearby/loud voices)
# 0.005 = hears everything, 0.03 = only nearby players, 0.06 = only close & loud
SILENCE_THRESHOLD = 0.03

# Seconds of silence after speech to consider utterance complete
SILENCE_AFTER_SPEECH = 1.5

# Minimum speech duration (seconds) to bother transcribing
# Filters out background snippets and noise bursts
MIN_SPEECH_DURATION = 1.0

# Cooldown after transcription — ignore audio for this long
# Prevents rapid-fire transcription in noisy lobbies
TRANSCRIPTION_COOLDOWN = 3.0

# Minimum word count to accept a transcription
# "You" or "The" alone is probably background noise
MIN_WORD_COUNT = 2

# When Chati's TTS is playing, use a much higher threshold so only LOUD
# barge-in speech triggers detection. This lets players interrupt Chati
# mid-sentence without the agent hearing its own TTS feedback.
BARGE_IN_THRESHOLD_MULTIPLIER = 4.0

# Force transcription after this many seconds of continuous speech.
# Critical for crowded lobbies where background chatter never dips
# below the silence threshold — otherwise the buffer grows forever
# and nothing ever gets transcribed.
MAX_UTTERANCE_SECONDS = 6.0


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
        self._speech_start: float = 0.0   # when current utterance began
        self._last_transcription_time: float = 0.0

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
                    raw = np.frombuffer(data, dtype=np.float32)

                    # Downmix to mono — DOA removed per AUDIT.md 1.3
                    if channels > 1:
                        mono = raw.reshape(-1, channels).mean(axis=1)
                    else:
                        mono = raw

                    # Cooldown still hard-skips (prevents hallucination spam)
                    if time.time() - self._last_transcription_time < TRANSCRIPTION_COOLDOWN:
                        continue

                    # Check energy
                    energy = np.sqrt(np.mean(mono ** 2))

                    # During TTS: only loud barge-in voices should trigger.
                    # Chati's own TTS via loopback is typically below this.
                    threshold = SILENCE_THRESHOLD
                    if self.muted:
                        threshold = SILENCE_THRESHOLD * BARGE_IN_THRESHOLD_MULTIPLIER

                    if energy > threshold:
                        if not self._is_speaking:
                            self._is_speaking = True
                            self._speech_start = time.time()
                            logger.info(f"* Speech detected (energy={energy:.3f})")
                        self._speech_buffer.append(mono)
                        self._silence_start = 0.0

                        # Force transcribe if the utterance has been
                        # going non-stop too long (e.g. noisy lobby).
                        if time.time() - self._speech_start > MAX_UTTERANCE_SECONDS:
                            logger.info("* Max utterance length reached — forcing transcribe")
                            self._transcribe_speech(native_rate)
                            self._is_speaking = False
                            self._speech_buffer.clear()
                            self._silence_start = 0.0
                            self._speech_start = 0.0
                    elif self._is_speaking:
                        if self._silence_start == 0.0:
                            self._silence_start = time.time()
                        self._speech_buffer.append(mono)

                        if time.time() - self._silence_start > SILENCE_AFTER_SPEECH:
                            self._transcribe_speech(native_rate)
                            self._is_speaking = False
                            self._speech_buffer.clear()
                            self._silence_start = 0.0
                            self._speech_start = 0.0

                except IOError as e:
                    logger.warning(f"Audio read warning: {e}")

        except Exception as e:
            logger.error(f"Audio capture error: {e}")

    def _transcribe_speech(self, native_rate: int) -> None:
        """Transcribe accumulated speech buffer with Whisper.

        Applies filters to avoid transcribing background noise.
        Cooldown is set on EVERY attempt (not just successful ones)
        to prevent burning GPU on rejected hallucinations in noisy
        lobbies (AUDIT.md section 3.4).
        """
        if not self._speech_buffer or not self._whisper:
            return

        # Set cooldown up-front so rejected chunks still back off
        self._last_transcription_time = time.time()

        speech = np.concatenate(self._speech_buffer)

        # Resample to 16kHz (Whisper expects 16kHz)
        if native_rate != 16000:
            num_samples = int(len(speech) * 16000 / native_rate)
            indices = np.linspace(0, len(speech) - 1, num_samples)
            speech = np.interp(indices, np.arange(len(speech)), speech).astype(np.float32)

        duration = len(speech) / 16000
        if duration < MIN_SPEECH_DURATION:
            logger.debug(f"* Speech too short ({duration:.1f}s), skipping")
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

            if not text:
                logger.debug("* Transcription empty, skipping")
                return

            word_count = len(text.split())
            if word_count < MIN_WORD_COUNT:
                logger.debug(f'* Too few words ({word_count}): "{text}", skipping')
                return

            # Filter common Whisper hallucinations on noise
            noise_phrases = {
                "thank you", "thanks for watching", "subscribe",
                "you", "the", "hmm", "uh", "um", "...",
            }
            if text.lower().strip(".,!? ") in noise_phrases:
                logger.debug(f'* Noise hallucination: "{text}", skipping')
                return

            logger.info(f'* Heard: "{text}"')

            # Clear old queued transcriptions — only keep the latest
            while not self._chunk_queue.empty():
                try:
                    self._chunk_queue.get_nowait()
                except queue.Empty:
                    break

            self._chunk_queue.put({
                "text": text,
                "duration": duration,
                "timestamp": time.time(),
            })

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
