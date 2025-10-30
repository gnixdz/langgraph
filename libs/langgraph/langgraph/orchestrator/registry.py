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

class NodeRegistry:
    def __init__(self, specs: List[NodeSpec] | None = None):
        self._by_id: Dict[str, NodeSpec] = {}
        if specs:
            for s in specs:
                self.add(s)

    def add(self, spec: NodeSpec):
        assert "id" in spec and "fn" in spec
        self._by_id[spec["id"]] = spec

    def get(self, nid: str) -> NodeSpec:
        return self._by_id[nid]

    def all(self) -> List[NodeSpec]:
        return list(self._by_id.values())

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
