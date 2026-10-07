"""Deterministic v5 evidence context; no execution, API or semantic scoring."""
from copy import deepcopy
import hashlib
import json
import re

from . import artifact_facts, judge, judge_v5, provenance_v5, transport_v3

VERSION = "prm-typed-context-v5-1"


def _same_json_readbacks(packet, artifacts):
    """Narrow policy guard: pure readback of exact already-supplied JSON.

    No path resolution or prior filesystem-existence claim. Do not guard
    observed changed content, an observed earlier differing artifact, additional
    tool observations, or an action with other public reasoning/work.
    """
    records = artifacts["records"]
    rows = {r["turn"]: r for r in packet["history"] + packet["targets"]}
    guards = []
    pure_call = re.compile(r"\s*<tool_call>\s*<function=bash>\s*<parameter=(?:cmd|command)>.*?</parameter>\s*</function>\s*</tool_call>\s*\Z", re.S)
    for record in records:
        n = record["turn"]
        if record["basis"] != "tool_observation" or record["path_binding"] != "direct_cat_read":
            continue
        action = rows[n]["action"]
        if not isinstance(action, str) or not pure_call.fullmatch(action) or action.count("<function=") != 1:
            continue
        current = [ev for ev in packet["evidence"] if ev["turn"] == n and ev["kind"] not in {"action_record", "action_ledger"}]
        if len(current) != 1 or current[0]["id"] != record["source_evidence_id"]:
            continue
        prior = [r for r in records if r["turn"] < n and r["artifact_path"] == record["artifact_path"]]
        matching = [r for r in prior if r["normalized_json"] == record["normalized_json"]]
        differing_observed = [r for r in prior if r["basis"] == "tool_observation" and r["normalized_json"] != record["normalized_json"]]
        if matching and not differing_observed:
            guards.append({"turn": n, "current_record_id": record["id"],
                "matching_prior_record_ids": [r["id"] for r in matching],
                "scope": "Pure direct readback of exact previously supplied JSON at same literal path; no prior file existence asserted. Under v5 routine-I/O policy this alone cannot earn +1."})
    return guards


def prepare(packet, image_receipts, *, execution_receipts=None):
    item = transport_v3.prepare_context(packet, image_receipts)
    p, context = item["packet"], item["context"]
    capabilities = provenance_v5.prepare(p, context, execution_receipts=execution_receipts)
    artifacts = artifact_facts.prepare(p)
    facts = {ident: deepcopy(fact) for ident, fact in artifacts["fact_index"].items()
             if fact.get("source_evidence_id") is not None}
    for card in context.get("typed_numeric_observations", {}).get("cards", []):
        for measurement in card.get("current_values", []):
            key = {k: measurement[k] for k in ("source_evidence_id", "asset", "field")}
            digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
            ident = "nf5:" + digest
            fact = {"id": ident, "turn": measurement["source_action_turn"],
                    "source_evidence_id": measurement["source_evidence_id"],
                    "basis": "tool_observation", "kind": "ffprobe_measurement",
                    "field": measurement["field"], "asset": measurement["asset"],
                    "derived_decimal": measurement["value"], "unit": measurement["unit"],
                    "scope": "Reported ffprobe value at this turn, not unchanged state or creative quality."}
            if ident in facts and facts[ident] != fact:
                raise judge.ProtocolError("conflicting deterministic fact identity")
            facts[ident] = fact
    index = capabilities["evidence_index"]
    typed_support = {
        "visual": [ident for ident, ev in index.items() if ev["availability"] == "available" and "visible_content" in ev["can_support"]],
        # The archived bundle has no authenticated audio attachment adapter.
        "audio": [],
        "execution": [ident for ident, ev in index.items() if "command_exit_zero" in ev["can_support"]],
        "execution_error": [ident for ident, ev in index.items() if any(c in ev["can_support"] for c in ("command_exit_nonzero", "command_not_executed"))],
    }
    known = {e["id"]: e for e in p["evidence"]}
    for ident, fact in facts.items():
        source = fact["source_evidence_id"]
        if source not in known or fact["turn"] != known[source]["turn"]:
            raise judge.ProtocolError("derived fact does not bind its source turn")
    context.update(typed_context_version=VERSION, capabilities_v5=capabilities,
                   artifact_facts_v5=artifacts, fact_index_v5=facts,
                   typed_support_v5=typed_support,
                   same_json_readback_guards_v5=_same_json_readbacks(p, artifacts))
    return item


def render_decision(assessment, packet, context):
    """Validate then expose fixed labels and exact fact records, no LLM prose."""
    accepted = judge_v5.parse_assessment(assessment, packet, context)
    result = judge_v5.canonical_result(accepted, context)
    judge.parse_assessment(result, packet)
    return {"canonical": result, "decisions": [
        {"turn": row["turn"], "event": row["event"], "scope": row["scope"],
         "support_evidence_ids": deepcopy(row["support_evidence_ids"]),
         "facts": [deepcopy(context["fact_index_v5"][ref]) for ref in row["fact_refs"]],
         "limitations": deepcopy(row["limitations"])} for row in accepted["assessments"]]}
