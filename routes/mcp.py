# import logging
from fastapi import APIRouter
from mcp.server.fastmcp import FastMCP
# mcp_logger = logging.getLogger("mcp")
# mcp_logger.setLevel(logging.WARNING)

router = APIRouter()
mcp = FastMCP("dummy-server")

@mcp.tool()
def dummy_mcp_tool():
    return {"status": "OK"}

@mcp.tool()
def greet(name: str, greeting: str = "Hello"):
    return {"message": f"{greeting}, {name}! Welcome to the MCP server."}
