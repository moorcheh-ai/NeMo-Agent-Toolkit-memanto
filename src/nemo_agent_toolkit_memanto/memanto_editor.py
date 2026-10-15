# SPDX-License-Identifier: Apache-2.0
"""MemoryEditor backed by Memanto, with one Memanto agent per user."""

import asyncio
import hashlib
from typing import Any

from memanto import Memanto
from memanto import NothingToExtractError
from memanto.app.utils.errors import AgentNotFoundError
from nat.plugin_api import MemoryEditor
from nat.plugin_api import MemoryItem

# Recall fields copied into MemoryItem.metadata; "id" is exposed as "memory_id".
_METADATA_FIELDS = ("type", "title", "confidence", "provenance", "source", "created_at", "updated_at")


class MemantoEditor(MemoryEditor):
    """Store and recall each user's memories in that user's own Memanto agent.

    The agent id is ``<agent_prefix>_<first 32 hex chars of sha256(user_id)>``, so
    users are isolated by namespace and a user id never appears in an agent name.

    ``add_items`` stores ``MemoryItem.memory`` as written. An item with only a
    ``conversation`` is passed to Memanto's extraction, which stores the durable
    facts, preferences and decisions it finds; small talk stores nothing.
    ``MemoryItem.metadata`` is not stored.

    Memanto clients are blocking and not thread-safe, so every call runs in a
    worker thread while holding a per-user lock.
    """

    def __init__(self, *, api_key: str | None = None, agent_prefix: str = "nat", max_memories: int = 20) -> None:
        self._api_key = api_key
        self._agent_prefix = agent_prefix
        self._max_memories = max_memories
        self._clients: dict[str, Memanto] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def agent_id_for(self, user_id: str) -> str:
        """Return the Memanto agent id that holds *user_id*'s memories."""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]
        return f"{self._agent_prefix}_{digest}"

    async def _client(self, agent_id: str, *, create: bool) -> Memanto | None:
        """Return the cached client for *agent_id*; call with its lock held.

        Returns None when the agent does not exist and *create* is False, so
        reads and deletes for an unknown user never create an agent.
        """
        client = self._clients.get(agent_id)
        if client is None:
            try:
                client = await asyncio.to_thread(Memanto, agent_id, api_key=self._api_key, auto_create=create)
            except AgentNotFoundError:
                return None
            self._clients[agent_id] = client
        return client

    def _lock(self, agent_id: str) -> asyncio.Lock:
        return self._locks.setdefault(agent_id, asyncio.Lock())

    async def add_items(self, items: list[MemoryItem]) -> None:
        """Store *items*, each in its user's agent."""
        await asyncio.gather(*(self._add_item(item) for item in items))

    async def _add_item(self, item: MemoryItem) -> None:
        if item.memory is None and item.conversation is None:
            raise ValueError("A MemoryItem needs a memory or a conversation to store")

        memory = item.memory if item.memory is not None and item.memory.strip() else None
        # Blank turns (an empty chat message) carry nothing to extract.
        messages = [message for message in item.conversation or [] if (message.get("content") or "").strip()]
        if memory is None and not messages:
            return

        agent_id = self.agent_id_for(item.user_id)
        async with self._lock(agent_id):
            client = await self._client(agent_id, create=True)
            assert client is not None  # create=True never returns None

            if memory is not None:
                await asyncio.to_thread(client.remember, memory, tags=item.tags or None)
                return
            try:
                await asyncio.to_thread(client.extract, messages, max_memories=self._max_memories)
            except NothingToExtractError:
                pass  # small talk; nothing worth remembering

    async def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[MemoryItem]:
        """Recall up to *top_k* of the user's memories most relevant to *query*.

        Requires ``user_id``. Also accepts Memanto recall filters: ``type``,
        ``tags`` and ``min_similarity``.
        """
        user_id = kwargs.pop("user_id")
        agent_id = self.agent_id_for(user_id)
        async with self._lock(agent_id):
            client = await self._client(agent_id, create=False)
            if client is None:
                return []
            result = await asyncio.to_thread(client.recall, query, limit=top_k, **kwargs)

        return [
            MemoryItem(
                user_id=user_id,
                memory=memory.get("content"),
                tags=memory.get("tags") or [],
                metadata={"memory_id": memory.get("id"), **{field: memory.get(field) for field in _METADATA_FIELDS}},
                similarity_score=memory.get("score"),
            )
            for memory in result.get("memories", [])
        ]

    async def remove_items(self, **kwargs: Any) -> None:
        """Delete one memory (``user_id`` and ``memory_id``) or all of a user's memories (``user_id``)."""
        user_id = kwargs.pop("user_id", None)
        memory_id = kwargs.pop("memory_id", None)
        if kwargs:
            raise TypeError(f"Unsupported remove_items arguments: {sorted(kwargs)}")
        if user_id is None:
            raise ValueError("remove_items requires user_id: each user's memories live in their own Memanto agent")

        agent_id = self.agent_id_for(user_id)
        async with self._lock(agent_id):
            client = await self._client(agent_id, create=False)
            if client is None:
                return
            if memory_id is not None:
                await asyncio.to_thread(client.delete_memory, memory_id)
                return
            await asyncio.to_thread(client.delete_agent, delete_memories=True)
            del self._clients[agent_id]
