"""Persistent MCP server connection pool.

Instead of spawning a fresh subprocess for every tool call, this module
starts each MCP server once and keeps the stdio connection alive for
reuse across multiple ``call_tool`` / ``list_tools`` invocations.

This eliminates the per-call subprocess overhead (~0.5-1s on Windows)
and makes parallel execution genuinely faster than sequential.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class MCPServerPool:
    """Pool of persistent MCP server connections.

    Usage::

        pool = MCPServerPool(server_paths)
        await pool.start_all()          # or start_servers({"iot", "fmsr"})
        result = await pool.call_tool("iot", "get_assets", {"site": "MAIN"})
        tools  = await pool.list_tools("iot")
        await pool.close()

    Or as an async context manager::

        async with MCPServerPool(server_paths) as pool:
            await pool.start_servers({"iot", "fmsr"})
            result = await pool.call_tool("iot", "get_assets", {"site": "MAIN"})
    """

    def __init__(self, server_paths: dict[str, Path | str]) -> None:
        self._server_paths = server_paths
        self._sessions: dict[str, Any] = {}        # name -> ClientSession
        self._locks: dict[str, asyncio.Lock] = {}   # name -> per-server lock
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "MCPServerPool":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # ── lifecycle ─────────────────────────────────────────────────────

    async def start_server(self, name: str) -> None:
        """Start a single MCP server and establish a persistent session."""
        if name in self._sessions:
            return  # already running

        path = self._server_paths.get(name)
        if path is None:
            _log.warning("Cannot start unknown server '%s'", name)
            return

        from .executor import _make_stdio_params
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        params = _make_stdio_params(path)
        read, write = await self._stack.enter_async_context(
            stdio_client(params)
        )
        session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()

        self._sessions[name] = session
        self._locks[name] = asyncio.Lock()
        _log.debug("Server '%s' started and connected.", name)

    async def start_servers(self, names: set[str]) -> None:
        """Start multiple servers (skips unknown names)."""
        for name in names:
            if name in self._server_paths:
                await self.start_server(name)

    async def start_all(self) -> None:
        """Start every registered server."""
        await self.start_servers(set(self._server_paths))

    async def close(self) -> None:
        """Shut down all server connections and subprocesses."""
        self._sessions.clear()
        self._locks.clear()
        await self._stack.aclose()

    # ── tool operations ───────────────────────────────────────────────

    async def call_tool(
        self, server_name: str, tool_name: str, args: dict
    ) -> str:
        """Call a tool on a running server (serialised per-server)."""
        session = self._sessions.get(server_name)
        if session is None:
            raise RuntimeError(
                f"Server '{server_name}' not started. "
                f"Running: {list(self._sessions)}"
            )
        async with self._locks[server_name]:
            result = await session.call_tool(tool_name, args)
            return _extract_content(result.content)

    async def list_tools(self, server_name: str) -> list[dict]:
        """List tools on a running server."""
        session = self._sessions.get(server_name)
        if session is None:
            raise RuntimeError(
                f"Server '{server_name}' not started. "
                f"Running: {list(self._sessions)}"
            )
        async with self._locks[server_name]:
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                schema = t.inputSchema or {}
                props = schema.get("properties", {})
                required = set(schema.get("required", []))
                parameters = [
                    {
                        "name": k,
                        "type": v.get("type", "any"),
                        "required": k in required,
                    }
                    for k, v in props.items()
                ]
                tools.append(
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": parameters,
                    }
                )
            return tools

    def has_server(self, name: str) -> bool:
        """Check if a server is currently connected."""
        return name in self._sessions


def _extract_content(content: list[Any]) -> str:
    """Extract text from MCP tool call result content."""
    return "\n".join(getattr(item, "text", str(item)) for item in content)
