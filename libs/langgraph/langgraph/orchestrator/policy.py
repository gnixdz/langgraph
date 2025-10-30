from typing import Any, Dict
import yaml

def load_policy(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        doc = yaml.safe_load(f) or {}
    # allow access to policy.tau etc.
    if "policy" in doc:
        doc.update(doc["policy"])
    return doc

def _get_path(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _val(x, state, policy):
    if isinstance(x, str):
        if x.startswith("policy."):
            return _get_path(policy, x)
        return _get_path(state, x, x)
    return x

def eval_pred(pred: Dict[str, Any], state: Dict[str, Any], policy: Dict[str, Any]) -> bool:
    if not pred:
        return False
    if "all" in pred:
        return all(eval_pred(p, state, policy) for p in pred["all"])
    if "any" in pred:
        return any(eval_pred(p, state, policy) for p in pred["any"])
    if "not" in pred:
        return not eval_pred(pred["not"], state, policy)
    if "eq" in pred:
        a, b = pred["eq"]; return _val(a, state, policy) == _val(b, state, policy)
    if "lt" in pred:
        a, b = pred["lt"]; return float(_val(a, state, policy)) < float(_val(b, state, policy))
    if "lte" in pred:
        a, b = pred["lte"]; return float(_val(a, state, policy)) <= float(_val(b, state, policy))
    if "gte" in pred:
        a, b = pred["gte"]; return float(_val(a, state, policy)) >= float(_val(b, state, policy))
    if "exists" in pred:
        return _get_path(state, pred["exists"]) is not None
    if "true" in pred:
        return bool(_get_path(state, pred["true"]))
    return False
