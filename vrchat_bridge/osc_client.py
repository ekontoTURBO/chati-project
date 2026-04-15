"""
VRChat OSC Client
=================
Persistent OSC connection to VRChat for avatar control,
movement, expressions, and chatbox messaging.

VRChat OSC docs: https://docs.vrchat.com/docs/osc-overview
Default ports: Send=9000, Receive=9001
"""

import threading
import time
import logging
from typing import Optional, Callable

from pythonosc import udp_client, dispatcher, osc_server

# Logger for this module — prefixed for easy filtering
logger = logging.getLogger("vrchat.osc")


class VRChatOSCClient:
    """Manages bidirectional OSC communication with VRChat.

    Attributes:
        send_ip: IP address to send OSC messages (default 127.0.0.1)
        send_port: Port VRChat listens on (default 9000)
        recv_port: Port to receive OSC messages from VRChat (default 9001)
    """

    def __init__(
        self,
        send_ip: str = "127.0.0.1",
        send_port: int = 9000,
        recv_port: int = 9001,
    ):
        # IP where VRChat is listening for OSC messages
        self.send_ip = send_ip
        # Port VRChat expects incoming OSC on (default 9000)
        self.send_port = send_port
        # Port we listen on for OSC responses from VRChat (default 9001)
        self.recv_port = recv_port

        # OSC sender — sends commands to VRChat
        self._client: Optional[udp_client.SimpleUDPClient] = None
        # OSC receiver thread — listens for avatar parameter changes
        self._server_thread: Optional[threading.Thread] = None
        # Flag to track connection state
        self._connected = False
        # Callbacks registered for incoming parameter changes
        self._param_callbacks: dict[str, list[Callable]] = {}

    def connect(self) -> None:
        """Establish OSC connection to VRChat.

        Creates the UDP sender and starts the receiver thread.
        """
        logger.info(
            f"Connecting to VRChat OSC at {self.send_ip}:{self.send_port}"
        )
        self._client = udp_client.SimpleUDPClient(
            self.send_ip, self.send_port
        )
        self._connected = True

        # Start receiver in background thread
        self._start_receiver()
        logger.info("VRChat OSC connected.")

    def disconnect(self) -> None:
        """Gracefully close the OSC connection."""
        self._connected = False
        if self._server_thread and self._server_thread.is_alive():
            self._osc_server.shutdown()
        logger.info("VRChat OSC disconnected.")

    def _start_receiver(self) -> None:
        """Start background thread to receive OSC messages from VRChat."""
        disp = dispatcher.Dispatcher()
        # Catch all avatar parameter changes
        disp.map("/avatar/parameters/*", self._handle_param_change)
        # Catch avatar change events
        disp.map("/avatar/change", self._handle_avatar_change)

        self._osc_server = osc_server.ThreadingOSCUDPServer(
            ("0.0.0.0", self.recv_port), disp
        )
        self._server_thread = threading.Thread(
            target=self._osc_server.serve_forever,
            daemon=True,  # Dies when main thread exits
            name="osc-receiver",
        )
        self._server_thread.start()
        logger.info(f"OSC receiver listening on port {self.recv_port}")

    def _handle_param_change(self, address: str, *args) -> None:
        """Handle incoming avatar parameter changes from VRChat.

        Args:
            address: OSC address (e.g., /avatar/parameters/VRCEmote)
            args: Parameter values
        """
        param_name = address.split("/")[-1]
        for callback in self._param_callbacks.get(param_name, []):
            callback(param_name, args)

    def _handle_avatar_change(self, address: str, *args) -> None:
        """Handle avatar change events (new avatar loaded)."""
        logger.info(f"Avatar changed: {args}")

    def on_param(self, param_name: str, callback: Callable) -> None:
        """Register a callback for avatar parameter changes.

        Args:
            param_name: Parameter name to watch
            callback: Function(param_name, values) to call
        """
        if param_name not in self._param_callbacks:
            self._param_callbacks[param_name] = []
        self._param_callbacks[param_name].append(callback)

    # ==========================================================
    # High-level API for VRChat control
    # ==========================================================

    def send(self, address: str, value) -> None:
        """Send a raw OSC message to VRChat.

        Args:
            address: OSC address path (e.g., /input/Jump)
            value: Value to send (int, float, bool, or str)
        """
        if not self._connected or not self._client:
            logger.warning("Cannot send — not connected to VRChat OSC")
            return
        self._client.send_message(address, value)

    def set_parameter(self, name: str, value) -> None:
        """Set an avatar parameter via OSC.

        Args:
            name: Parameter name (e.g., VRCEmote)
            value: Parameter value (int, float, or bool)
        """
        self.send(f"/avatar/parameters/{name}", value)

    def chatbox_message(self, text: str, direct: bool = True) -> None:
        """Send a message to VRChat's chatbox.

        Args:
            text: Message text to display
            direct: If True, display immediately; if False, use typing
        """
        # VRChat chatbox input: (message, send_immediately, play_sfx)
        self._client.send_message(
            "/chatbox/input", [text, direct, False]
        )

    def chatbox_typing(self, is_typing: bool = True) -> None:
        """Show/hide the typing indicator in VRChat chatbox.

        Args:
            is_typing: True to show typing, False to hide
        """
        self.send("/chatbox/typing", is_typing)

    def move(self, vertical: float = 0.0, horizontal: float = 0.0) -> None:
        """Send locomotion input to VRChat.

        Args:
            vertical: Forward/back axis (-1.0 to 1.0)
            horizontal: Left/right axis (-1.0 to 1.0)
        """
        self.send("/input/Vertical", vertical)
        self.send("/input/Horizontal", horizontal)

    def stop_moving(self) -> None:
        """Stop all movement by zeroing locomotion axes."""
        self.move(0.0, 0.0)

    def jump(self) -> None:
        """Make the avatar jump."""
        self.send("/input/Jump", 1)
        # VRChat needs a brief press then release
        time.sleep(0.1)
        self.send("/input/Jump", 0)

    def set_emote(self, emote_id: int) -> None:
        """Trigger an avatar emote/expression.

        Args:
            emote_id: Emote number (1-8 for standard VRChat emotes)
        """
        self.set_parameter("VRCEmote", emote_id)

    def look_horizontal(self, value: float) -> None:
        """Set horizontal look direction.

        Args:
            value: -1.0 (left) to 1.0 (right)
        """
        self.send("/input/LookHorizontal", value)

    def look_vertical(self, value: float) -> None:
        """Set vertical look direction.

        Args:
            value: -1.0 (down) to 1.0 (up)
        """
        self.send("/input/LookVertical", value)

    def voice(self, active: bool = True) -> None:
        """Toggle voice/microphone input (push-to-talk).

        Args:
            active: True to activate mic, False to deactivate
        """
        self.send("/input/Voice", 1 if active else 0)

    def run(self, running: bool = True) -> None:
        """Toggle running mode.

        Args:
            running: True to start running, False to walk
        """
        self.send("/input/Run", 1 if running else 0)

    @property
    def is_connected(self) -> bool:
        """Check if the OSC client is currently connected."""
        return self._connected
