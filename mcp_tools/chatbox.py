"""
MCP Tool: send_chatbox(text)
==============================
Sends text messages to VRChat's in-game chatbox display.
"""

import logging

logger = logging.getLogger("mcp.chatbox")


class ChatboxTool:
    """MCP tool for sending text to VRChat chatbox.

    Attributes:
        osc_client: VRChatOSCClient instance for sending messages
    """

    def __init__(self, osc_client, speak_tool=None):
        # OSC client for sending chatbox messages to VRChat
        self.osc_client = osc_client
        # NOTE: speak_tool param kept for backward compat but NOT used.
        # The controller owns speech decisions (AUDIT.md section 1.7).

    def send_chatbox(self, text: str) -> dict:
        """Send a text message to VRChat's chatbox.

        Appears above the avatar's head. VRChat caps chatbox at ~144
        chars so longer text is truncated at a sentence boundary when
        possible, with "..." appended.

        Args:
            text: Message to display

        Returns:
            Status dict with sent text
        """
        if not text or not text.strip():
            return {"success": False, "error": "Empty text"}

        display = text
        if len(display) > 144:
            # Try to cut at a sentence boundary
            cutoff = display[:140]
            for sep in (". ", "! ", "? ", "; "):
                idx = cutoff.rfind(sep)
                if idx > 60:
                    cutoff = cutoff[:idx + 1]
                    break
            display = cutoff.rstrip() + "..."

        logger.info(f"Chatbox: {display}")

        try:
            self.osc_client.chatbox_message(display, direct=True)
            return {"success": True, "text": display}
        except Exception as e:
            logger.error(f"Chatbox send failed: {e}")
            return {"success": False, "error": str(e)}

    @property
    def tool_schema(self) -> dict:
        """MCP tool definition for function calling."""
        return {
            "name": "send_chatbox",
            "description": (
                "Send a text message to VRChat's chatbox. The message "
                "appears above the avatar. Use for text communication "
                "in addition to or instead of speaking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Message to display (max 144 chars)",
                    }
                },
                "required": ["text"],
            },
        }
