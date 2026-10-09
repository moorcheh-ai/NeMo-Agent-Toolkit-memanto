# NeMo Agent Toolkit Memanto Plugin

Long-term memory for [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) workflows, backed by [Memanto](https://github.com/moorcheh-ai/memanto).

The plugin registers a memory backend with `_type: memanto`. It works with the toolkit's built-in memory tools (`add_memory`, `get_memory`, `delete_memory`) and with the automatic memory wrapper (`auto_memory_agent`).

## How it stores memory

- **One Memanto agent per user.** Each `user_id` maps to its own agent, `<agent_prefix>_<first 32 hex characters of sha256(user_id)>`. Users are isolated by namespace, and user ids never appear in agent names.
- **Explicit memories are stored as written.** When a `MemoryItem` has `memory` text, as the `add_memory` tool usually produces, that text is stored with the item's `tags`.
- **Conversations are distilled.** When an item has only a `conversation`, as the automatic memory wrapper produces, Memanto extracts the durable facts, preferences and decisions in it and stores those as typed memories. Small talk stores nothing.
- **Reads never create agents.** Searching or deleting for a user with no memories returns nothing and creates nothing.

`MemoryItem.metadata` is not stored. Search results carry each memory's id, type, title, confidence, provenance, source and timestamps in `metadata`, and its relevance in `similarity_score`, where higher is more similar.

## Install

```bash
uv add nemo-agent-toolkit-memanto
```

or:

```bash
pip install nemo-agent-toolkit-memanto
```

The package must be installed in the same Python environment as `nvidia-nat-core`. Confirm the toolkit discovers it:

```bash
nat info components -t memory -q memanto
```

## Credentials

Set your Moorcheh API key, from [console.moorcheh.ai](https://console.moorcheh.ai):

```bash
export MOORCHEH_API_KEY=...
```

If it is not set, the key saved by `memanto` setup is used. The on-prem backend needs no key.

## Configuration

```yaml
memory:
  user_memory:
    _type: memanto
```

| Field | Default | Description |
| --- | --- | --- |
| `api_key` | `MOORCHEH_API_KEY` | Moorcheh API key. |
| `agent_prefix` | `nat` | Prefix of the per-user agent ids, matching `[A-Za-z0-9_-]{1,32}`. Use a different prefix for each application so they never share memories. |
| `max_memories` | `20` | Maximum memories extracted from one conversation, from 1 to 100. |

### Automatic memory

The wrapper saves every user message, recalls relevant memories before the inner agent answers, and saves the agent's replies. The user identity comes from the toolkit runtime.

```yaml
memory:
  user_memory:
    _type: memanto

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.3-70b-instruct

functions:
  current_datetime:
    _type: current_datetime
  my_react_agent:
    _type: react_agent
    llm_name: nim_llm
    tool_names: [current_datetime]

workflow:
  _type: auto_memory_agent
  description: Assistant that remembers each user
  inner_agent_name: my_react_agent
  memory_name: user_memory
  llm_name: nim_llm
  search_params:
    top_k: 5
```

Automatic memory needs the LangChain integration, `nvidia-nat-langchain`. This example is in [`examples/auto_memory.yml`](examples/auto_memory.yml); it uses an NVIDIA NIM model, so set `NVIDIA_API_KEY` as well:

```bash
nat run --config_file examples/auto_memory.yml --input "I'm vegetarian and I live in Toronto."
nat run --config_file examples/auto_memory.yml --input "Suggest a restaurant for me."
```

### Memory tools

```yaml
memory:
  user_memory:
    _type: memanto

functions:
  add_memory:
    _type: add_memory
    memory: user_memory
    user_id: alex
  get_memory:
    _type: get_memory
    memory: user_memory
    user_id: alex
  delete_memory:
    _type: delete_memory
    memory: user_memory
    user_id: alex
```

A fixed `user_id` suits a single-user deployment. For multiple authenticated users, set a trusted `user_id_resolver` instead, as described in the toolkit's guide to authenticating memory tool users.

Tool-level `user_id` and `user_id_resolver` need NeMo Agent Toolkit 1.9 or later. On 1.8, leave them out: the agent passes `user_id` in each tool call.

## Backend API

| `MemoryEditor` call | Memanto operation |
| --- | --- |
| `add_items(items)` | `remember` for items with `memory`; conversation extraction for items with only `conversation` |
| `search(query, top_k, user_id=...)` | `recall`. Also accepts the Memanto filters `type`, `tags` and `min_similarity`, for example through `search_params` |
| `remove_items(user_id=...)` | Deletes the user's agent and all of its memories |
| `remove_items(user_id=..., memory_id=...)` | Deletes one memory |

`MemantoEditor.agent_id_for(user_id)` returns the agent that holds a user's memories, so you can inspect them with the `memanto` CLI or UI.

## Supported versions

| Package | Versions |
| --- | --- |
| NeMo Agent Toolkit (`nvidia-nat-core`) | 1.8, 1.9 |
| Python | 3.11, 3.12, 3.13 |
| Memanto | 0.2.28 or later |

## Development

```bash
uv sync --extra test
uv run pytest
```

The integration tests call the live Memanto backend, including a full `auto_memory_agent` workflow with two users. They need `MOORCHEH_API_KEY` and the toolkit's `--run_integration` flag:

```bash
MOORCHEH_API_KEY=... uv run pytest --run_integration -m integration
```

Each run uses a random agent prefix and deletes the agents it creates.

## Reporting bugs

- Bugs in this plugin, or in how Memanto stores and recalls memory: [moorcheh-ai/memanto issues](https://github.com/moorcheh-ai/memanto/issues).
- Bugs in the toolkit itself, such as memory tools, the automatic memory wrapper, or plugin discovery: [NVIDIA/NeMo-Agent-Toolkit issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues).

## License

Apache-2.0. See [LICENSE](LICENSE).
