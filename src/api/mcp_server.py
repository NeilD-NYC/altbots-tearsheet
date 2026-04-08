import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=True)

mcp = FastMCP(
    name="altbots-tearsheet",
    instructions="AltBots institutional intelligence. Use get_manager_tearsheet for fund profiles.",
)

@mcp.tool()
async def get_manager_tearsheet(manager_name: str) -> str:
    """Retrieve an institutional tearsheet for a fund manager."""
    # Simplified for testing; your actual logic remains in core
    return json.dumps({"status": "success", "manager": manager_name, "message": "Backend Alive."})

if __name__ == "__main__":
    mcp.run()
