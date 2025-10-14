from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

# simple weights; you can make them config-driven later
W = {"range": 0.4, "domain": 0.3, "functional": 0.1, "format": 0.2}

class SymbolicVerifierNode:
    """
    Deterministic verifier for candidate mappings.
    Expects in state:
      - column_meta: {"name","dtype","table_class", "not_null", "unique"}
      - ontology: object with helpers: has_path(src, dst, max_hops=1), range_of(id), domain_of(id),
                  label_of(id), functional(id), expected_regex(id) (optional)
      - candidates: list of {"id": <ontology_property_id>, "property_meta": {...}}  (property_meta optional)
      - sample_values: optional list[str]
    Produces in state:
      - verifier_scores: list[float]
      - verifier_flags:  list[dict]
    """
    def __init__(self,
                 timeout_ms: int = 200,
                 weights: Optional[Dict[str, float]] = None,
                 semantic_penalty_pairs: Optional[List[Tuple[str, str]]] = None):
        self.timeout_ms = timeout_ms
        self.W = weights or dict(W)
        self.semantic_pairs = semantic_penalty_pairs or [
            ("birth","address"),
            ("age","birth"),
            ("postal","city"),
        ]

    # LangGraph calls .invoke(state) on Python nodes
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        col = state["column_meta"]
        onto = state["ontology"]
        cands = state.get("candidates", [])
        samples = state.get("sample_values", [])

        v_scores, v_flags = [], []
        for cand in cands:
            prop_id = cand["id"]
            prop_meta = cand.get("property_meta", {})

            rng = prop_meta.get("range") or getattr(onto, "range_of")(prop_id)
            dom = prop_meta.get("domain") or getattr(onto, "domain_of")(prop_id)
            label = prop_meta.get("label")  or getattr(onto, "label_of")(prop_id) or prop_id
            functional = prop_meta.get("functional")
            if functional is None and hasattr(onto, "functional"):
                functional = getattr(onto, "functional")(prop_id)
            exp_regex = prop_meta.get("expected_regex")
            if exp_regex is None and hasattr(onto, "expected_regex"):
                exp_regex = getattr(onto, "expected_regex")(prop_id)

            r = _range_check(col.get("dtype"), rng)
            d = _domain_check(col.get("table_class"), dom, onto)
            f = _functional_check(col, functional)
            m = _format_check(samples, exp_regex)

            v_hat = (self.W["range"]*r +
                     self.W["domain"]*d +
                     self.W["functional"]*f +
                     self.W["format"]*m)

            if _semantic_mismatch(col.get("name",""), str(label), self.semantic_pairs):
                v_hat *= 0.8

            v_hat = max(0.0, min(1.0, v_hat))
            v_scores.append(round(v_hat, 3))
            v_flags.append({"range": r, "domain": d, "functional": f, "format": m})

        state["verifier_scores"] = v_scores
        state["verifier_flags"] = v_flags
        return state


# ---- helpers (deterministic) ----

def _range_check(dtype: Optional[str], on_range: Optional[str]) -> float:
    if not dtype or not on_range: return 0.0
    dtype = dtype.lower()
    on_range = on_range.lower()
    exact = {("int","xsd:int"), ("integer","xsd:int"),
             ("bigint","xsd:int"), ("float","xsd:float"),
             ("double","xsd:double"), ("varchar","xsd:string"),
             ("text","xsd:string"), ("date","xsd:date"), ("datetime","xsd:dateTime")}
    convertible = {("int","xsd:string"), ("integer","xsd:string"),
                   ("varchar","xsd:anyuri"), ("text","xsd:anyuri")}
    if (dtype, on_range) in exact: return 1.0
    if (dtype, on_range) in convertible: return 0.5
    return 0.0

def _domain_check(table_class: Optional[str], on_domain: Optional[str], ontology) -> float:
    if not table_class or not on_domain: return 0.0
    if table_class == on_domain: return 1.0
    if hasattr(ontology, "has_path") and ontology.has_path(table_class, on_domain, max_hops=1):
        return 0.5
    return 0.0

def _functional_check(col_meta: Dict[str, Any], prop_functional: Optional[bool]) -> float:
    if prop_functional is None: return 0.5  # neutral
    not_null = bool(col_meta.get("not_null"))
    unique = bool(col_meta.get("unique"))
    if prop_functional and (unique or not_null): return 1.0
    if not prop_functional and not unique: return 0.5
    return 0.5

def _format_check(values: List[Any], regex: Optional[str]) -> float:
    if not regex or not values: return 0.5
    import re
    p = re.compile(regex)
    n = len(values)
    ok = 0
    for v in values:
        s = str(v) if v is not None else ""
        if p.fullmatch(s): ok += 1
    ratio = ok / max(1, n)
    if ratio >= 0.9: return 1.0
    if ratio >= 0.5: return 0.5
    return 0.0

def _semantic_mismatch(col_name: str, prop_label: str, pairs: List[tuple[str,str]]) -> bool:
    cn, pl = col_name.lower(), prop_label.lower()
    return any((a in cn and b in pl) or (a in pl and b in cn) for a,b in pairs)
