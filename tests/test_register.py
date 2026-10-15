# SPDX-License-Identifier: Apache-2.0
"""Loader smoke tests: the toolkit discovers the plugin and builds the backend."""

from importlib.metadata import entry_points

import pytest
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.config import Config
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from pydantic import ValidationError

from nemo_agent_toolkit_memanto import memanto_editor
from nemo_agent_toolkit_memanto.memanto_editor import MemantoEditor
from nemo_agent_toolkit_memanto.memory import MemantoMemoryConfig


def test_entry_point_is_declared():
    (entry_point,) = [ep for ep in entry_points(group="nat.plugins") if ep.name == "nat_memanto"]
    assert entry_point.value == "nemo_agent_toolkit_memanto.register"
    entry_point.load()


def test_toolkit_discovers_memanto_memory():
    discover_and_register_plugins(PluginTypes.ALL)
    registered = GlobalTypeRegistry.get().get_memory(MemantoMemoryConfig)
    assert registered.config_type is MemantoMemoryConfig
    assert registered.local_name == "memanto"


def test_yaml_config_selects_memanto():
    discover_and_register_plugins(PluginTypes.ALL)
    config = Config.model_validate(
        {
            "memory": {"user_memory": {"_type": "memanto", "agent_prefix": "support"}},
            "workflow": {"_type": "test_echo"},
        }
    )
    memory_config = config.memory["user_memory"]
    assert isinstance(memory_config, MemantoMemoryConfig)
    assert memory_config.api_key is None
    assert memory_config.agent_prefix == "support"
    assert memory_config.max_memories == 20


@pytest.mark.parametrize("field", [{"agent_prefix": "bad prefix"}, {"agent_prefix": ""}, {"max_memories": 0}])
def test_invalid_config_is_rejected(field):
    with pytest.raises(ValidationError):
        MemantoMemoryConfig(**field)


def test_api_key_is_secret():
    config = MemantoMemoryConfig(api_key="secret-key")
    assert "secret-key" not in repr(config)


async def test_builder_returns_configured_editor(monkeypatch):
    monkeypatch.setattr(memanto_editor, "Memanto", object)  # never contacted here
    async with WorkflowBuilder() as builder:
        await builder.add_memory_client(
            "user_memory", MemantoMemoryConfig(api_key="secret-key", agent_prefix="app", max_memories=5)
        )
        editor = await builder.get_memory_client("user_memory")

    assert isinstance(editor, MemantoEditor)
    assert editor._api_key == "secret-key"
    assert editor._max_memories == 5
    assert editor.agent_id_for("alex").startswith("app_")
