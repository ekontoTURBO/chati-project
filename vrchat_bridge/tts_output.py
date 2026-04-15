"""
TTS Audio Output Router
========================
Routes Piper TTS audio to VB-Audio Cable virtual microphone
so VRChat picks it up as microphone input.

Uses pyaudiowpatch for reliable device access even when
VRChat or other apps share the same audio device.

Requires:
  - VB-Audio Cable installed (https://vb-audio.com/Cable/)
  - PyAudioWPatch + numpy installed
"""

import threading
import queue
import logging
import time
from typing import Optional

import numpy as np
import pyaudiowpatch as pyaudio

# Logger for this module
logger = logging.getLogger("vrchat.tts_output")


class TTSAudioRouter:
    """Routes TTS audio bytes to VB-Audio Cable virtual microphone.

    Attributes:
        sample_rate: Audio sample rate in Hz (must match Piper output)
        channels: Number of audio channels (1 = mono)
    """

    def __init__(self, sample_rate: int = 22050, channels: int = 1, osc_client=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self._osc = osc_client  # For push-to-talk control

        self._audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._device_index: Optional[int] = None
        self._player_thread: Optional[threading.Thread] = None
        self._running = False
        self._pa: Optional[pyaudio.PyAudio] = None

    def _find_cable_input(self) -> dict:
        """Find a writable VB-Audio Cable device via pyaudiowpatch.

        Tries CABLE In 16ch first (more likely to be available),
        then falls back to regular CABLE Input.

        Returns:
            Device info dict

        Raises:
            RuntimeError: If no VB-Audio Cable is found
        """
        # Prefer regular CABLE Input (VRChat can see CABLE Output as mic)
        for i in range(self._pa.get_device_count()):
            d = self._pa.get_device_info_by_index(i)
            name = d["name"].lower()
            if ("cable input" in name
                    and "16ch" not in name
                    and d["maxOutputChannels"] > 0):
                logger.info(f"Found VB-Audio Cable: [{i}] {d['name']}")
                return d

        # Fallback: CABLE In 16ch
        for i in range(self._pa.get_device_count()):
            d = self._pa.get_device_info_by_index(i)
            name = d["name"].lower()
            api = self._pa.get_host_api_info_by_index(d["hostApi"])["name"]
            if "cable in 16ch" in name and d["maxOutputChannels"] > 0 and api == "MME":
                logger.info(f"Found VB-Audio Cable 16ch: [{i}] {d['name']}")
                return d

        raise RuntimeError(
            "VB-Audio Cable not found. Install from: https://vb-audio.com/Cable/"
        )

    def start(self) -> None:
        """Start the audio router."""
        self._pa = pyaudio.PyAudio()
        device_info = self._find_cable_input()
        self._device_index = int(device_info["index"])

        self._running = True
        self._player_thread = threading.Thread(
            target=self._play_loop,
            daemon=True,
            name="tts-audio-player",
        )
        self._player_thread.start()
        logger.info("TTS audio router started.")

    def stop(self) -> None:
        """Stop the audio router."""
        self._running = False
        self._audio_queue.put(None)
        if self._player_thread:
            self._player_thread.join(timeout=5.0)
        if self._pa:
            self._pa.terminate()
        logger.info("TTS audio router stopped.")

    def play(self, audio_bytes: bytes) -> None:
        """Queue raw PCM audio bytes for playback."""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        self._audio_queue.put(audio_float)

    def play_numpy(self, audio_array: np.ndarray) -> None:
        """Queue a numpy audio array for playback."""
        self._audio_queue.put(audio_array)

    def _play_loop(self) -> None:
        """Background loop that plays audio through VB-Audio Cable."""
        logger.info("TTS play loop started.")
        while self._running:
            try:
                audio = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if audio is None:
                break

            # Convert float32 to int16 bytes for pyaudio
            audio_int16 = (audio * 32767).astype(np.int16).tobytes()
            logger.info(f"Playing {len(audio_int16)} bytes of audio...")

            try:
                # Hold push-to-talk down before speaking
                if self._osc:
                    logger.info("PTT ON")
                    for _ in range(3):
                        self._osc.voice(True)
                        time.sleep(0.05)

                stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=self._device_index,
                )
                # Write audio in chunks, keep PTT held
                chunk_size = 4096
                ptt_counter = 0
                for offset in range(0, len(audio_int16), chunk_size):
                    if not self._running:
                        break
                    stream.write(audio_int16[offset:offset + chunk_size])
                    # Re-send PTT every ~20 chunks to keep it held
                    ptt_counter += 1
                    if self._osc and ptt_counter % 20 == 0:
                        self._osc.voice(True)
                stream.stop_stream()
                stream.close()

                # Release push-to-talk after speaking
                if self._osc:
                    time.sleep(0.1)
                    self._osc.voice(False)
                    logger.info("PTT OFF")
                logger.info("Playback complete.")
            except Exception as e:
                if self._osc:
                    self._osc.voice(False)
                logger.error(f"Audio playback error: {e}")
                time.sleep(0.5)

    @property
    def is_running(self) -> bool:
        """Check if the audio router is currently active."""
        return self._running
