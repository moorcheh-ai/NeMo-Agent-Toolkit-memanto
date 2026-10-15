# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MemantoEditor against a fake Memanto client."""

import asyncio
import hashlib
import threading
import time
from typing import Any

import pytest
from memanto import NothingToExtractError
from memanto.app.utils.errors import AgentNotFoundError
from nat.plugin_api import MemoryItem

from nemo_agent_toolkit_memanto import memanto_editor
from nemo_agent_toolkit_memanto.memanto_editor import MemantoEditor


class FakeMemanto:
    """Stand-in for memanto.Memanto that records calls per agent."""

    agents: dict[str, list[tuple[str, tuple, dict]]] = {}
    constructed: list[dict[str, Any]] = []
    extract_error: Exception | None = None
    recall_result: dict[str, Any] = {"memories": []}
    active_calls = 0
    max_active_calls = 0
    lock = threading.Lock()

    def __init__(self, agent_id: str, *, api_key: str | None = None, auto_create: bool = True) -> None:
        FakeMemanto.constructed.append({"agent_id": agent_id, "api_key": api_key, "auto_create": auto_create})
        if agent_id not in FakeMemanto.agents:
            if not auto_create:
                raise AgentNotFoundError(agent_id)
            FakeMemanto.agents[agent_id] = []
        self.agent_id = agent_id

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        with FakeMemanto.lock:
            FakeMemanto.active_calls += 1
            FakeMemanto.max_active_calls = max(FakeMemanto.max_active_calls, FakeMemanto.active_calls)
        time.sleep(0.01)
        FakeMemanto.agents[self.agent_id].append((name, args, kwargs))
        with FakeMemanto.lock:
            FakeMemanto.active_calls -= 1

    def remember(self, content: str, **kwargs: Any) -> dict[str, Any]:
        self._record("remember", content, **kwargs)
        return {"memory_id": "m1"}

    def extract(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self._record("extract", messages, **kwargs)
        if FakeMemanto.extract_error is not None:
            raise FakeMemanto.extract_error
        return {"successful": 1}

    def recall(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self._record("recall", query, **kwargs)
        return FakeMemanto.recall_result

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        self._record("delete_memory", memory_id)
        return {"status": "deleted"}

    def delete_agent(self, *, delete_memories: bool = False) -> dict[str, Any]:
        self._record("delete_agent", delete_memories=delete_memories)
        del FakeMemanto.agents[self.agent_id]
        return {"status": "deleted"}


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    FakeMemanto.agents = {}
    FakeMemanto.constructed = []
    FakeMemanto.extract_error = None
    FakeMemanto.recall_result = {"memories": []}
    FakeMemanto.active_calls = 0
    FakeMemanto.max_active_calls = 0
    monkeypatch.setattr(memanto_editor, "Memanto", FakeMemanto)
    return FakeMemanto


@pytest.fixture
def editor():
    return MemantoEditor(api_key="key", agent_prefix="app", max_memories=7)


def calls(editor: MemantoEditor, user_id: str) -> list[tuple[str, tuple, dict]]:
    return FakeMemanto.agents[editor.agent_id_for(user_id)]


def test_agent_id_is_prefixed_sha256_of_user_id(editor):
    expected = "app_" + hashlib.sha256(b"alex@example.com").hexdigest()[:32]
    assert editor.agent_id_for("alex@example.com") == expected
    assert editor.agent_id_for("alex") != editor.agent_id_for("Alex")


@pytest.mark.parametrize("user_id", ["", "   ", None])
def test_agent_id_rejects_missing_user_id(editor, user_id):
    with pytest.raises(ValueError, match="user_id"):
        editor.agent_id_for(user_id)


async def test_memory_text_is_remembered_with_tags(editor):
    await editor.add_items(
        [
            MemoryItem(
                user_id="alex",
                memory="Alex is vegetarian.",
                conversation=[{"role": "user", "content": "Alex is vegetarian."}],
                tags=["diet"],
                metadata={"ignored": True},
            )
        ]
    )

    assert calls(editor, "alex") == [("remember", ("Alex is vegetarian.",), {"tags": ["diet"]})]
    assert FakeMemanto.constructed == [{"agent_id": editor.agent_id_for("alex"), "api_key": "key", "auto_create": True}]


async def test_memory_without_tags_passes_none(editor):
    await editor.add_items([MemoryItem(user_id="alex", memory="Likes tea.")])
    assert calls(editor, "alex") == [("remember", ("Likes tea.",), {"tags": None})]


async def test_conversation_is_extracted(editor):
    conversation = [
        {"role": "user", "content": "I'm moving to Toronto."},
        {"role": "assistant", "content": ""},
    ]
    await editor.add_items([MemoryItem(user_id="alex", conversation=conversation)])

    assert calls(editor, "alex") == [
        ("extract", ([{"role": "user", "content": "I'm moving to Toronto."}],), {"max_memories": 7})
    ]


async def test_nothing_to_extract_is_not_an_error(editor):
    FakeMemanto.extract_error = NothingToExtractError("nothing worth remembering")
    await editor.add_items([MemoryItem(user_id="alex", conversation=[{"role": "user", "content": "hi"}])])
    assert [name for name, _, _ in calls(editor, "alex")] == ["extract"]


async def test_other_extraction_failures_propagate(editor):
    FakeMemanto.extract_error = ValueError("Memory extraction did not return valid JSON")
    with pytest.raises(ValueError, match="valid JSON"):
        await editor.add_items([MemoryItem(user_id="alex", conversation=[{"role": "user", "content": "hi"}])])


async def test_blank_conversation_stores_nothing_and_creates_no_agent(editor):
    await editor.add_items([MemoryItem(user_id="alex", conversation=[{"role": "user", "content": "  "}])])
    await editor.add_items([MemoryItem(user_id="alex", memory=" ")])
    assert FakeMemanto.agents == {}
    assert FakeMemanto.constructed == []


async def test_blank_memory_falls_back_to_conversation(editor):
    await editor.add_items(
        [MemoryItem(user_id="alex", memory="  ", conversation=[{"role": "user", "content": "I use vim."}])]
    )
    assert [name for name, _, _ in calls(editor, "alex")] == ["extract"]


async def test_item_without_memory_or_conversation_is_rejected(editor):
    with pytest.raises(ValueError, match="memory or a conversation"):
        await editor.add_items([MemoryItem(user_id="alex")])


async def test_unsupported_add_params_are_rejected(editor):
    with pytest.raises(TypeError):
        await editor.add_items([MemoryItem(user_id="alex", memory="x")], infer=True)


async def test_users_are_isolated_in_separate_agents(editor):
    await editor.add_items(
        [
            MemoryItem(user_id="alex", memory="Alex fact."),
            MemoryItem(user_id="sam", memory="Sam fact."),
        ]
    )
    assert calls(editor, "alex") == [("remember", ("Alex fact.",), {"tags": None})]
    assert calls(editor, "sam") == [("remember", ("Sam fact.",), {"tags": None})]


async def test_client_is_created_once_per_user(editor):
    await editor.add_items([MemoryItem(user_id="alex", memory="a")])
    await editor.add_items([MemoryItem(user_id="alex", memory="b")])
    await editor.search("q", user_id="alex")
    assert len(FakeMemanto.constructed) == 1


async def test_calls_for_one_user_are_serialized(editor):
    await editor.add_items([MemoryItem(user_id="alex", memory=f"fact {i}") for i in range(5)])
    assert FakeMemanto.max_active_calls == 1
    assert len(calls(editor, "alex")) == 5


async def test_calls_for_different_users_run_concurrently(editor):
    await editor.add_items([MemoryItem(user_id=f"user{i}", memory="fact") for i in range(5)])
    assert FakeMemanto.max_active_calls > 1


async def test_search_maps_memories_to_items(editor):
    await editor.add_items([MemoryItem(user_id="alex", memory="seed")])
    FakeMemanto.recall_result = {
        "memories": [
            {
                "id": "m1",
                "title": "Diet",
                "content": "Alex is vegetarian.",
                "type": "preference",
                "confidence": 0.9,
                "tags": ["diet"],
                "provenance": "explicit_statement",
                "source": "user",
                "created_at": "2026-10-09T00:00:00+00:00",
                "updated_at": "2026-10-09T00:00:00+00:00",
                "score": 0.42,
                "text": "not copied",
            }
        ]
    }

    items = await editor.search("what does Alex eat?", top_k=3, user_id="alex", type=["preference"])

    assert calls(editor, "alex")[-1] == ("recall", ("what does Alex eat?",), {"limit": 3, "type": ["preference"]})
    assert items == [
        MemoryItem(
            user_id="alex",
            memory="Alex is vegetarian.",
            tags=["diet"],
            metadata={
                "memory_id": "m1",
                "type": "preference",
                "title": "Diet",
                "confidence": 0.9,
                "provenance": "explicit_statement",
                "source": "user",
                "created_at": "2026-10-09T00:00:00+00:00",
                "updated_at": "2026-10-09T00:00:00+00:00",
            },
            similarity_score=0.42,
        )
    ]


async def test_search_for_unknown_user_returns_nothing_and_creates_nothing(editor):
    assert await editor.search("q", user_id="stranger") == []
    assert FakeMemanto.agents == {}
    assert FakeMemanto.constructed[0]["auto_create"] is False


async def test_search_requires_user_id(editor):
    with pytest.raises(KeyError):
        await editor.search("q")


async def test_remove_one_memory(editor):
    await editor.add_items([MemoryItem(user_id="alex", memory="seed")])
    await editor.remove_items(user_id="alex", memory_id="m1")
    assert calls(editor, "alex")[-1] == ("delete_memory", ("m1",), {})


async def test_remove_all_memories_deletes_the_agent(editor):
    await editor.add_items([MemoryItem(user_id="alex", memory="seed")])
    agent_id = editor.agent_id_for("alex")

    await editor.remove_items(user_id="alex")

    assert agent_id not in FakeMemanto.agents
    # The deleted client is dropped, so the next write starts a fresh agent.
    await editor.add_items([MemoryItem(user_id="alex", memory="again")])
    assert calls(editor, "alex") == [("remember", ("again",), {"tags": None})]


async def test_remove_for_unknown_user_is_a_no_op(editor):
    await editor.remove_items(user_id="stranger")
    assert FakeMemanto.agents == {}


async def test_remove_requires_user_id(editor):
    with pytest.raises(ValueError, match="requires user_id"):
        await editor.remove_items(memory_id="m1")


async def test_remove_rejects_unknown_arguments(editor):
    with pytest.raises(TypeError, match="run_id"):
        await editor.remove_items(user_id="alex", run_id="r1")


async def test_concurrent_first_use_creates_one_client(editor):
    await asyncio.gather(
        *(editor.search("q", user_id="alex") for _ in range(3)),
        editor.add_items([MemoryItem(user_id="alex", memory="x")]),
    )
    assert len([c for c in FakeMemanto.constructed if c["auto_create"]]) == 1
