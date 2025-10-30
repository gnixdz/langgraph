from typing import List, Dict, Any, Callable
from typing_extensions import TypedDict

class NodeSpec(TypedDict, total=False):
    id: str
    tags: List[str]
    capabilities: List[str]
    inputs: Dict[str, Any]       # e.g., {"requires": ["table","column"]}
    outputs: Dict[str, Any]      # e.g., {"adds": ["confidence"]}
    cost: Dict[str, float]       # e.g., {"llm_calls":1,"api_calls":0,"expected_latency_ms":900}
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]

# --- in your NodeRegistry class ---
class NodeRegistry:
    def __init__(self):
        self._nodes = {}  # id -> callable

    def register(self, node_id: str, fn):
        self._nodes[node_id] = fn

    # NEW: public helpers
    def ids(self):
        return list(self._nodes.keys())

    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get(self, node_id: str):
        return self._nodes[node_id]

    # Optional Python protocol sugar
    def __contains__(self, node_id: str):
        return node_id in self._nodes

    def __len__(self):
        return len(self._nodes)

# Optional decorator to register nodes from user code
_GLOBAL_REGISTRY = NodeRegistry()

def node_meta(*, id: str, tags=None, capabilities=None, inputs=None, outputs=None, cost=None):
    tags = tags or []; capabilities = capabilities or []
    inputs = inputs or {}; outputs = outputs or {}; cost = cost or {}
    def _wrap(fn):
        _GLOBAL_REGISTRY.add({
            "id": id, "tags": tags, "capabilities": capabilities,
            "inputs": inputs, "outputs": outputs, "cost": cost, "fn": fn
        })
        return fn
    return _wrap

def get_global_registry() -> NodeRegistry:
    return _GLOBAL_REGISTRY
