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

    def __init__(self, osc_client):
        # OSC client for sending chatbox messages to VRChat
        self.osc_client = osc_client

    def send_chatbox(self, text: str) -> dict:
        """Send a text message to VRChat's chatbox.

        The message appears above the avatar's head in-game.

        Args:
            text: Message to display (max ~144 chars for VRChat)

        Returns:
            Status dict with sent text
        """
        if not text or not text.strip():
            return {"success": False, "error": "Empty text"}

        # VRChat chatbox has a ~144 character limit
        # Constraint: truncate to avoid OSC errors
        if len(text) > 144:
            text = text[:141] + "..."

        logger.info(f"Chatbox: {text}")

        try:
            self.osc_client.chatbox_message(text, direct=True)
            return {"success": True, "text": text}
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
