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
        # The controller owns the decision to speak; chatbox should not
        # have hidden TTS side-effects (see AUDIT.md section 1.7).
        self._speak_tool = None

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
            # Auto-speak the message (strip emojis for TTS)
            if self._speak_tool:
                import re
                # Remove emojis and other non-ASCII symbols
                clean = re.sub(
                    r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
                    r'\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
                    r'\U00002702-\U000027B0\U0000FE00-\U0000FE0F'
                    r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F'
                    r'\U0001FA70-\U0001FAFF\U00002600-\U000026FF'
                    r'\U0000200D\U00002764]+', '', text
                ).strip()
                if clean:
                    self._speak_tool.speak(clean)
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
