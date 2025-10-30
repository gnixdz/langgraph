from typing import Dict, Any, Optional, Callable
from .controller import Controller
from .registry import NodeRegistry

class Orchestrator:
    """Executes a dynamic micro-plan by calling node fns selected by Controller.
    Keeps LangGraph intact; no core patching required."""
    def __init__(self, registry: NodeRegistry, controller: Controller, *, max_steps: int = 64):
        self.registry = registry
        self.controller = controller
        self.max_steps = max_steps
        self.trace: list[dict] = []

    def run(self, state: Dict[str, Any], *, entry: Optional[str] = None) -> Dict[str, Any]:
        steps = 0
        last = None
        # optional entry node
        if entry:
            out = self._exec(entry, state)
            state.update(out or {})
            last = entry
            steps += 1

        while steps < self.max_steps:
            if self.controller.terminal(state):
                break
            action = self.controller.decide(state, last)
            nid = action.get("dispatch")
            if not nid:
                break
            out = self._exec(nid, state)
            if isinstance(out, dict):
                state.update(out)
            last = nid
            steps += 1
        return state

    def _exec(self, node_id: str, state: Dict[str, Any]) -> Dict[str, Any] | None:
        spec = self.registry.get(node_id)
        fn: Callable[[Dict[str, Any]], Dict[str, Any]] = spec["fn"]  # type: ignore
        try:
            res = fn(state)
            self.trace.append({"node": node_id, "ok": True})
            return res
        except Exception as e:
            self.trace.append({"node": node_id, "ok": False, "error": repr(e)})
            # Bubble error or swallow; for now bubble
            raise
