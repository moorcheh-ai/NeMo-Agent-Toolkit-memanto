# SPDX-License-Identifier: Apache-2.0
"""Live tests against the Memanto backend. Skipped unless MOORCHEH_API_KEY is set.

Each test uses a random agent prefix and deletes the agents it created.
"""

import asyncio
import os
import uuid

import pytest
import yaml
from nat.data_models.api_server import ChatRequest
from nat.plugin_api import Builder
from nat.plugin_api import FunctionBaseConfig
from nat.plugin_api import FunctionInfo
from nat.plugin_api import MemoryItem
from nat.plugin_api import register_function

from nemo_agent_toolkit_memanto.memanto_editor import MemantoEditor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("MOORCHEH_API_KEY"), reason="MOORCHEH_API_KEY is not set"),
]

# Memanto indexes writes asynchronously; recall sees them shortly after.
INDEX_DELAY_S = 3


class CaptureAgentConfig(FunctionBaseConfig, name="memanto_test_capture_agent"):
    """Inner agent that records the messages it receives."""


CAPTURED: list[list[str]] = []


@register_function(config_type=CaptureAgentConfig)
async def capture_agent(_config: CaptureAgentConfig, _builder: Builder):

    async def _respond(chat_request: ChatRequest) -> str:
        CAPTURED.append([str(message.content) for message in chat_request.messages])
        return "Noted."

    yield FunctionInfo.from_fn(_respond, description="Records the conversation it is given.")


@pytest.fixture
def prefix():
    return f"natit{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def editor(prefix):
    editor = MemantoEditor(agent_prefix=prefix)
    yield editor
    for user_id in ("alice", "bob"):
        await editor.remove_items(user_id=user_id)


async def test_store_recall_isolate_and_delete(editor):
    await editor.add_items(
        [
            MemoryItem(user_id="alice", memory="Alice is allergic to peanuts.", tags=["food-allergy", "health"]),
            MemoryItem(user_id="bob", memory="Bob prefers aisle seats on flights.", tags=["travel"]),
            MemoryItem(
                user_id="alice",
                conversation=[{"role": "user", "content": "I just moved to Toronto and I work as a nurse."}],
            ),
            MemoryItem(user_id="alice", conversation=[{"role": "user", "content": "hi there"}]),
        ]
    )
    await asyncio.sleep(INDEX_DELAY_S)

    alice = await editor.search("Where does Alice live and what can't she eat?", top_k=10, user_id="alice")
    alice_text = " ".join(item.memory or "" for item in alice).lower()
    assert "peanut" in alice_text
    assert "toronto" in alice_text
    assert "aisle" not in alice_text  # Bob's memory stays in Bob's agent
    assert all(item.user_id == "alice" for item in alice)
    assert all(item.metadata["memory_id"] for item in alice)
    allergy = next(item for item in alice if "peanut" in (item.memory or "").lower())
    assert allergy.tags == ["food-allergy", "health"]

    bob = await editor.search("seat preference", top_k=10, user_id="bob")
    assert [item.memory for item in bob] == ["Bob prefers aisle seats on flights."]

    await editor.remove_items(user_id="bob", memory_id=bob[0].metadata["memory_id"])
    await asyncio.sleep(INDEX_DELAY_S)
    assert await editor.search("seat preference", top_k=10, user_id="bob") == []

    await editor.remove_items(user_id="alice")
    assert await editor.search("peanuts", user_id="alice") == []


async def test_unknown_user_reads_nothing(editor):
    assert await editor.search("anything", user_id="alice") == []
    await editor.remove_items(user_id="alice")


async def test_auto_memory_workflow_recalls_per_user(prefix, tmp_path):
    from nat.runtime.loader import load_workflow

    config = {
        "memory": {"user_memory": {"_type": "memanto", "agent_prefix": prefix}},
        "llms": {"unused_llm": {"_type": "nat_test_llm"}},
        "functions": {"inner": {"_type": "memanto_test_capture_agent"}},
        "workflow": {
            "_type": "auto_memory_agent",
            "description": "Assistant with Memanto memory",
            "inner_agent_name": "inner",
            "memory_name": "user_memory",
            "llm_name": "unused_llm",
        },
    }
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.safe_dump(config))

    async def chat(session_manager, user_id: str, message: str) -> None:
        async with session_manager.session(user_id=user_id) as session:
            async with session.run(message) as runner:
                await runner.result()

    CAPTURED.clear()
    async with load_workflow(config_file) as session_manager:
        try:
            await chat(session_manager, "alice", "I'm a vegetarian and my dog is called Biscuit.")
            await asyncio.sleep(INDEX_DELAY_S)

            await chat(session_manager, "alice", "What is my dog's name?")
            await chat(session_manager, "bob", "What is my dog's name?")
        finally:
            editor = await session_manager._shared_builder.get_memory_client("user_memory")
            for user_id in ("alice", "bob"):
                await editor.remove_items(user_id=user_id)

    first, alice_followup, bob_followup = CAPTURED
    assert not any("Relevant context from memory" in message for message in first)
    alice_context = " ".join(alice_followup)
    assert "Relevant context from memory" in alice_context
    assert "Biscuit" in alice_context
    assert not any("Biscuit" in message for message in bob_followup)


async def test_memory_tools_round_trip(prefix):
    """The toolkit's add_memory, get_memory and delete_memory tools work against Memanto."""
    import json

    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.tool.memory_tools.add_memory_tool import AddToolConfig
    from nat.tool.memory_tools.delete_memory_tool import DeleteToolConfig
    from nat.tool.memory_tools.get_memory_tool import GetToolConfig

    from nemo_agent_toolkit_memanto.memory import MemantoMemoryConfig

    # Toolkit 1.9 takes the identity from tool config; 1.8 takes it from the tool input.
    identity_in_config = "user_id" in AddToolConfig.model_fields
    tool_identity = {} if identity_in_config else {"user_id": "alice"}
    config_identity = {"user_id": "alice"} if identity_in_config else {}

    async with WorkflowBuilder() as builder:
        await builder.add_memory_client("user_memory", MemantoMemoryConfig(agent_prefix=prefix))
        tools = {}
        for name, config_cls in (("add", AddToolConfig), ("get", GetToolConfig), ("delete", DeleteToolConfig)):
            tools[name] = await builder.add_function(name, config_cls(memory="user_memory", **config_identity))

        try:
            added = await tools["add"].ainvoke(
                {"memory": "Alice's favourite colour is teal.", "tags": ["profile"], **tool_identity}
            )
            assert "Memory added successfully" in added
            await asyncio.sleep(INDEX_DELAY_S)

            found = await tools["get"].ainvoke(
                {"query": "What is Alice's favourite colour?", "top_k": 3, **tool_identity}
            )
            memories = json.loads(found.split("\n", 1)[1])
            assert any("teal" in memory["memory"] for memory in memories)
            assert all(memory["user_id"] == "alice" for memory in memories)

            await tools["delete"].ainvoke(tool_identity)
            found = await tools["get"].ainvoke({"query": "favourite colour", "top_k": 3, **tool_identity})
            assert json.loads(found.split("\n", 1)[1]) == []
        finally:
            editor = await builder.get_memory_client("user_memory")
            await editor.remove_items(user_id="alice")
