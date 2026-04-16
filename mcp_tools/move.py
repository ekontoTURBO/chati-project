"""
MCP Tool: move(direction, speed)
=================================
Sends locomotion OSC commands to control VRChat avatar movement.
"""

import asyncio
import logging
import time

logger = logging.getLogger("mcp.move")

# Mapping of direction names to (vertical_axis, horizontal_axis) values
# Constraint: Both axes range from -1.0 to 1.0
DIRECTION_MAP = {
    "forward": (1.0, 0.0),
    "backward": (-1.0, 0.0),
    "back": (-1.0, 0.0),
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
    "forward_left": (1.0, -0.7),
    "forward_right": (1.0, 0.7),
    "stop": (0.0, 0.0),
}


class MoveTool:
    """MCP tool that controls VRChat avatar locomotion.

    Attributes:
        osc_client: VRChatOSCClient instance for sending commands
    """

    def __init__(self, osc_client):
        # OSC client for sending movement commands to VRChat
        self.osc_client = osc_client

    def move(self, direction: str, speed: float = 1.0, duration: float = 1.0) -> dict:
        """Move the avatar in a specified direction.

        Args:
            direction: Movement direction ('forward', 'left', etc.)
            speed: Speed multiplier from 0.0 (stopped) to 1.0 (full)
            duration: How long to move in seconds (default 1.0)

        Returns:
            Status dict with movement details
        """
        direction = direction.lower().strip()

        if direction not in DIRECTION_MAP:
            available = ", ".join(sorted(DIRECTION_MAP.keys()))
            return {
                "success": False,
                "error": f"Unknown direction: '{direction}'. "
                         f"Available: {available}",
            }

        # Clamp speed between 0.0 and 1.0
        speed = max(0.0, min(1.0, speed))
        # Clamp duration between 0.1 and 10.0 seconds
        duration = max(0.1, min(10.0, duration))

        vertical, horizontal = DIRECTION_MAP[direction]
        # Apply speed multiplier to axis values
        vertical *= speed
        horizontal *= speed

        logger.info(
            f"Moving {direction} at speed {speed:.1f} for {duration:.1f}s"
        )

        try:
            # Enable running if speed is high
            if speed > 0.7:
                self.osc_client.run(True)

            # Send movement axes
            self.osc_client.move(vertical, horizontal)

            # Hold movement for the specified duration
            time.sleep(duration)

            # Stop movement after duration
            self.osc_client.stop_moving()
            self.osc_client.run(False)

            return {
                "success": True,
                "direction": direction,
                "speed": speed,
                "duration": duration,
            }
        except Exception as e:
            # Safety: always try to stop on error
            self.osc_client.stop_moving()
            logger.error(f"Movement failed: {e}")
            return {"success": False, "error": str(e)}

    def jump(self) -> dict:
        """Make the avatar jump.

        Returns:
            Status dict indicating success
        """
        logger.info("Jumping!")
        try:
            self.osc_client.jump()
            return {"success": True, "action": "jump"}
        except Exception as e:
            logger.error(f"Jump failed: {e}")
            return {"success": False, "error": str(e)}

    def approach(self, look_tool=None, spatial_memory=None) -> dict:
        """Approach the last known player/sound position.

        Uses spatial memory to determine which direction to face,
        then walks forward. Multi-step smart movement.

        Args:
            look_tool: LookTool for turning (optional)
            spatial_memory: SpatialMemory for direction lookup (optional)

        Returns:
            Status dict
        """
        if spatial_memory is None:
            # Fallback: just walk forward
            logger.info("Approach: no spatial memory, walking forward")
            return self.move(direction="forward", speed=0.8, duration=1.5)

        direction = spatial_memory.get_approach_direction()
        if direction is None:
            logger.info("Approach: no known target, looking around")
            if look_tool:
                look_tool.turn(direction="right", amount="quarter")
            return {"success": True, "action": "searching"}

        logger.info(f"Approach: target direction = {direction}")

        if direction == "left" and look_tool:
            look_tool.turn(direction="left", amount="quarter")
            time.sleep(0.2)
        elif direction == "right" and look_tool:
            look_tool.turn(direction="right", amount="quarter")
            time.sleep(0.2)

        # Walk forward regardless (after turning if needed)
        return self.move(direction="forward", speed=0.9, duration=2.0)

    @property
    def tool_schema(self) -> dict:
        """MCP tool definition for function calling."""
        directions = ", ".join(sorted(DIRECTION_MAP.keys()))
        return {
            "name": "move",
            "description": (
                f"Move the avatar in VRChat. Directions: {directions}. "
                "Speed 0.0-1.0, duration in seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "Movement direction",
                        "enum": sorted(DIRECTION_MAP.keys()),
                    },
                    "speed": {
                        "type": "number",
                        "description": "Speed from 0.0 to 1.0 (default 1.0)",
                        "default": 1.0,
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds (default 1.0)",
                        "default": 1.0,
                    },
                },
                "required": ["direction"],
            },
        }
