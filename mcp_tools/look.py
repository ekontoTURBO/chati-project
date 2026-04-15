"""
MCP Tool: look_at(target)
==========================
Controls VRChat avatar head/eye tracking via OSC.

VRChat OSC look input is continuous — sending a value keeps
rotating. We send a brief impulse then reset to zero.
"""

import logging
import time

logger = logging.getLogger("mcp.look")

# Mapping of look targets to (horizontal, vertical) axis values
# Constraint: Both axes range from -1.0 to 1.0
LOOK_TARGETS = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, 1.0),
    "down": (0.0, -1.0),
    "up_left": (-0.7, 0.7),
    "up_right": (0.7, 0.7),
    "down_left": (-0.7, -0.7),
    "down_right": (0.7, -0.7),
    "center": (0.0, 0.0),
    "reset": (0.0, 0.0),
}


class LookTool:
    """MCP tool that controls avatar head/eye direction.

    Attributes:
        osc_client: VRChatOSCClient instance for sending commands
    """

    def __init__(self, osc_client):
        # OSC client for sending look direction commands
        self.osc_client = osc_client

    def turn(self, direction: str = "left", amount: str = "quarter") -> dict:
        """Rotate the avatar's body by turning in place.

        Args:
            direction: 'left' or 'right'
            amount: 'slight' (45deg), 'quarter' (90deg), 'half' (180deg)

        Returns:
            Status dict
        """
        direction = direction.lower().strip()
        amount = amount.lower().strip()

        if direction not in ("left", "right"):
            return {"success": False, "error": f"Invalid turn direction: {direction}"}

        # Duration determines how far we rotate
        durations = {"slight": 0.3, "quarter": 0.6, "half": 1.2}
        duration = durations.get(amount, 0.6)

        h_val = -1.0 if direction == "left" else 1.0
        logger.info(f"Turning {direction} ({amount}, {duration}s)")

        try:
            self.osc_client.look_horizontal(h_val)
            time.sleep(duration)
            self.osc_client.look_horizontal(0.0)
            return {"success": True, "direction": direction, "amount": amount}
        except Exception as e:
            self.osc_client.look_horizontal(0.0)
            logger.error(f"Turn failed: {e}")
            return {"success": False, "error": str(e)}

    @property
    def turn_schema(self) -> dict:
        """MCP tool definition for the turn tool."""
        return {
            "name": "turn",
            "description": (
                "Rotate the avatar's body left or right. "
                "Use this to look in a new direction or turn around. "
                "amount: slight (45deg), quarter (90deg), half (180deg)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "Turn direction",
                        "enum": ["left", "right"],
                    },
                    "amount": {
                        "type": "string",
                        "description": "How far to turn",
                        "enum": ["slight", "quarter", "half"],
                        "default": "quarter",
                    },
                },
                "required": ["direction"],
            },
        }

    def look_at(self, target: str) -> dict:
        """Set the avatar's head/eye look direction.

        Args:
            target: Direction to look ('left', 'right', 'up', etc.)

        Returns:
            Status dict with look target details
        """
        target = target.lower().strip()

        if target not in LOOK_TARGETS:
            available = ", ".join(sorted(LOOK_TARGETS.keys()))
            return {
                "success": False,
                "error": f"Unknown look target: '{target}'. "
                         f"Available: {available}",
            }

        h_val, v_val = LOOK_TARGETS[target]
        logger.info(f"Looking {target} (h={h_val:.1f}, v={v_val:.1f})")

        try:
            # Send the look direction as a brief impulse
            self.osc_client.look_horizontal(h_val)
            self.osc_client.look_vertical(v_val)
            # Hold for a short moment so VRChat registers it
            time.sleep(0.3)
            # Reset to center to stop continuous rotation
            self.osc_client.look_horizontal(0.0)
            self.osc_client.look_vertical(0.0)
            return {"success": True, "target": target}
        except Exception as e:
            logger.error(f"Look failed: {e}")
            return {"success": False, "error": str(e)}

    @property
    def tool_schema(self) -> dict:
        """MCP tool definition for function calling."""
        targets = ", ".join(sorted(LOOK_TARGETS.keys()))
        return {
            "name": "look_at",
            "description": (
                f"Control avatar head/eye direction. "
                f"Available targets: {targets}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Direction to look",
                        "enum": sorted(LOOK_TARGETS.keys()),
                    }
                },
                "required": ["target"],
            },
        }
