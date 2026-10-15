# SPDX-License-Identifier: Apache-2.0
"""Memanto memory backend registration for NeMo Agent Toolkit."""

from collections.abc import AsyncGenerator

from nat.plugin_api import Builder
from nat.plugin_api import MemoryBaseConfig
from nat.plugin_api import MemoryEditor
from nat.plugin_api import OptionalSecretStr
from nat.plugin_api import get_secret_value
from nat.plugin_api import register_memory
from pydantic import Field


class MemantoMemoryConfig(MemoryBaseConfig, name="memanto"):
    """Memanto long-term memory. Each user gets their own Memanto agent."""

    api_key: OptionalSecretStr = Field(
        default=None,
        description=(
            "Moorcheh API key. Defaults to MOORCHEH_API_KEY, then the key saved by "
            "`memanto` setup. Not needed on the on-prem backend."
        ),
    )
    agent_prefix: str = Field(
        default="nat",
        pattern=r"^[A-Za-z0-9_-]{1,32}$",
        description=(
            "Prefix of the per-user agent ids, `<agent_prefix>_<hash of user_id>`. "
            "Use a distinct prefix per application so they never share memories."
        ),
    )
    max_memories: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum memories extracted from one conversation passed to add_items.",
    )


@register_memory(config_type=MemantoMemoryConfig)
async def memanto_memory_client(config: MemantoMemoryConfig, _builder: Builder) -> AsyncGenerator[MemoryEditor, None]:
    from nemo_agent_toolkit_memanto.memanto_editor import MemantoEditor

    yield MemantoEditor(
        api_key=get_secret_value(config.api_key),
        agent_prefix=config.agent_prefix,
        max_memories=config.max_memories,
    )
