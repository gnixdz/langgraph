from typing import Dict, Any, List, Optional, Tuple
from .policy import eval_pred
from .registry import NodeRegistry, NodeSpec

def _has_caps(spec: NodeSpec, required: List[str]) -> bool:
    return all(c in spec.get("capabilities", []) for c in required)

def _io_ok(spec: NodeSpec, state: Dict[str, Any]) -> bool:
    reqs = spec.get("inputs", {}).get("requires", [])
    return all(k in state for k in reqs)

def _budget_ok(spec: NodeSpec, state: Dict[str, Any], goal_budget: Dict[str, Any] | None) -> bool:
    if not goal_budget:
        return True
    used_llm = state.get("budget", {}).get("llm", 0)
    used_api = state.get("budget", {}).get("api", 0)
    llm_cost = spec.get("cost", {}).get("llm_calls", 0)
    api_cost = spec.get("cost", {}).get("api_calls", 0)
    return (used_llm + llm_cost) <= goal_budget.get("llm_calls_max", float("inf")) and \
           (used_api + api_cost) <= goal_budget.get("api_calls_max", float("inf"))

def _rank(cands: List[NodeSpec], prefer_tags: List[str] | None) -> List[Tuple[float, NodeSpec]]:
    if not prefer_tags:
        return [(0.0, c) for c in cands]
    out: List[Tuple[float, NodeSpec]] = []
    for c in cands:
        score = sum(1.0 for t in c.get("tags", []) if t in prefer_tags)
        # tie-break on latency if available
        lat = float(c.get("cost", {}).get("expected_latency_ms", 0))
        out.append((-score + lat/1e6, c))
    return sorted(out, key=lambda x: x[0])

class Controller:
    """Pure decision logic. No execution. Returns {'dispatch': node_id} and optional {'plan': [node_ids]}."""
    def __init__(self, policy: Dict[str, Any], registry: NodeRegistry):
        self.policy = policy
        self.registry = registry

    def terminal(self, state: Dict[str, Any]) -> bool:
        term = self.policy.get("routing", {}).get("terminal_when")
        return eval_pred(term, state, self.policy) if term else False

    def decide(self, state: Dict[str, Any], last_node: Optional[str] = None) -> Dict[str, Any]:
        goals = self.policy.get("goals", [])
        active = [g for g in goals if eval_pred(g["trigger"], state, self.policy)]
        if not active:
            return {"dispatch": None}

        goal = active[0]  # simple priority
        req_caps = goal.get("require", {}).get("all_caps", [])
        prefer = goal.get("prefer", {}).get("tags_any", [])
        budget = goal.get("budget")

        cands = [s for s in self.registry.all() if _has_caps(s, req_caps) and _io_ok(s, state) and _budget_ok(s, state, budget)]
        ranked = _rank(cands, prefer)
        if not ranked:
            fb = goal.get("fallback", {})
            forb = set(fb.get("forbid_tags", []))
            cands_fb = [s for s in self.registry.all() if _has_caps(s, req_caps) and _io_ok(s, state)
                        and forb.isdisjoint(set(s.get("tags", [])))]
            ranked = _rank(cands_fb, fb.get("prefer", []))
            if not ranked:
                return {"dispatch": None}

        chosen = ranked[0][1]
        return {"dispatch": chosen["id"]}
