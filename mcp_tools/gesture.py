"""
MCP Tool: gesture(type)
========================
Sends emote/expression OSC commands to VRChat avatar.

VRChat emote IDs: 1=Wave, 2=Clap, 3=Point, 4=Cheer,
                   5=Dance, 6=Backflip, 7=Die, 8=Sadness
"""

import logging
import time

logger = logging.getLogger("mcp.gesture")

# Mapping of human-readable gesture names to VRChat emote IDs
# Constraint: IDs 1-8 are standard VRChat emotes
GESTURE_MAP = {
    "wave": 1,
    "clap": 2,
    "point": 3,
    "cheer": 4,
    "dance": 5,
    "backflip": 6,
    "die": 7,
    "sadness": 8,
    "sad": 8,
    "happy": 4,
    "thumbsup": 3,
    "bow": 7,
    "reset": 0,  # Reset to idle
}


class GestureTool:
    """MCP tool that triggers VRChat avatar emotes/gestures.

    Attributes:
        osc_client: VRChatOSCClient instance for sending commands
    """

    def __init__(self, osc_client):
        # OSC client used to send gesture commands to VRChat
        self.osc_client = osc_client

    def gesture(self, gesture_type: str = "", **kwargs) -> dict:
        """Trigger an avatar gesture/emote in VRChat.

        Args:
            gesture_type: Name of the gesture (e.g., 'wave', 'dance')

        Returns:
            Status dict with 'success' and 'gesture' fields
        """
        # The model sends "type" (from schema) but Python can't use
        # "type" as a param name, so accept it via kwargs
        if not gesture_type:
            gesture_type = kwargs.get("type", "")
        gesture_type = gesture_type.lower().strip()

        if gesture_type not in GESTURE_MAP:
            available = ", ".join(sorted(GESTURE_MAP.keys()))
            return {
                "success": False,
                "error": f"Unknown gesture: '{gesture_type}'. "
                         f"Available: {available}",
            }

        emote_id = GESTURE_MAP[gesture_type]
        logger.info(f"Performing gesture: {gesture_type} (emote #{emote_id})")

        try:
            self.osc_client.set_emote(emote_id)
            # Hold the emote briefly, then reset to idle
            time.sleep(2.0)
            self.osc_client.set_emote(0)
            return {"success": True, "gesture": gesture_type, "emote_id": emote_id}
        except Exception as e:
            logger.error(f"Gesture failed: {e}")
            return {"success": False, "error": str(e)}

    @property
    def tool_schema(self) -> dict:
        """MCP tool definition for function calling."""
        gestures = ", ".join(sorted(GESTURE_MAP.keys()))
        return {
            "name": "gesture",
            "description": (
                f"Perform an avatar gesture/emote in VRChat. "
                f"Available gestures: {gestures}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Gesture name to perform",
                        "enum": sorted(GESTURE_MAP.keys()),
                    }
                },
                "required": ["type"],
            },
        }
