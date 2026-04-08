"""
MCP Tool: join_world()
========================
Joins a VRChat world using the vrchat:// URI deep link protocol.
This launches VRChat into the specified world instance.

Note: VRChat must already be running. The deep link triggers
a world change within the running client.
"""

import subprocess
import logging

logger = logging.getLogger("mcp.world")

# Popular public worlds for random exploration
KNOWN_WORLDS = {
    "the_black_cat": "wrld_4cf554b4-430c-4f8f-b53e-1f294eed230b",
    "just_b_club": "wrld_9caebc3b-9a7c-4b30-b3c4-6e5e0f1e3a5c",
    "movie_and_chill": "wrld_791ebf58-54ce-4d3a-a0a0-39571c46a3b4",
    "the_great_pug": "wrld_6caf5200-70e1-46c2-b043-e3c4abe69e0f",
    "midnight_rooftop": "wrld_ddf10a76-45e7-4781-8c62-0acec6a5e30d",
    "void_club": "wrld_12345678-void-club-placeholder-0000",
    "home": "wrld_0e3bb118-3261-4b03-aed2-f0c2f82e3cc9",
}


class WorldTool:
    """MCP tool for joining VRChat worlds via deep link.

    Uses the vrchat:// URI protocol to trigger world changes
    in the running VRChat client.
    """

    def join_world(self, world_id: str = "", world_name: str = "") -> dict:
        """Join a VRChat world by ID or known name.

        Args:
            world_id: VRChat world ID (wrld_xxxx format). Takes priority.
            world_name: Friendly name from known worlds list.

        Returns:
            Dict with success status and world info
        """
        # Resolve world name to ID if needed
        if not world_id and world_name:
            world_id = KNOWN_WORLDS.get(world_name.lower().replace(" ", "_"))
            if not world_id:
                return {
                    "success": False,
                    "error": f"Unknown world name: '{world_name}'",
                    "known_worlds": list(KNOWN_WORLDS.keys()),
                }

        if not world_id:
            return {
                "success": False,
                "error": "Provide either world_id or world_name",
                "known_worlds": list(KNOWN_WORLDS.keys()),
            }

        # Build the VRChat deep link URI
        uri = f"vrchat://launch?ref=vrchat.com&id={world_id}:~region(us)"

        logger.info(f"Joining world: {world_id}")
        logger.info(f"Deep link: {uri}")

        try:
            # Launch the deep link — Windows will route it to VRChat
            subprocess.Popen(
                ["cmd", "/c", "start", "", uri],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return {
                "success": True,
                "world_id": world_id,
                "world_name": world_name or "custom",
            }
        except Exception as e:
            logger.error(f"Failed to launch world: {e}")
            return {"success": False, "error": str(e)}

    @property
    def tool_schema(self) -> dict:
        """MCP tool definition for function calling."""
        return {
            "name": "join_world",
            "description": (
                "Join a VRChat world. Provide either a world_id "
                "(wrld_xxxx format) or a known world_name. "
                f"Known worlds: {', '.join(KNOWN_WORLDS.keys())}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "world_id": {
                        "type": "string",
                        "description": "VRChat world ID in wrld_xxxx format",
                    },
                    "world_name": {
                        "type": "string",
                        "description": (
                            "Known world name: "
                            + ", ".join(KNOWN_WORLDS.keys())
                        ),
                    },
                },
                "required": [],
            },
        }
