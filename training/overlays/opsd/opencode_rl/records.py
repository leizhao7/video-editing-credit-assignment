"""Capture actual engine inputs/outputs and admit only complete, exact turns."""

from __future__ import annotations

import math

from .contract import Contract, ParityError, digest, require_ids


class EngineCapture:
    def __init__(self, journal, contract: Contract, weight_version: str):
        self.journal, self.contract, self.weight_version = journal, contract, weight_version
        self.input = None
        self.ids, self.logprobs = [], []
        self.finished = False

    def start(self, prompt_ids, features, sampling, *, engine_request_id, logprobs_mode):
        if self.input is not None:
            raise ParityError("One HTTP request must map to exactly one engine request")
        require_ids(prompt_ids, "engine prompt IDs")
        if len(prompt_ids) >= self.contract.max_context:
            raise ParityError("Prompt exceeds contract")
        if logprobs_mode != "raw_logprobs":
            raise ParityError("Unreviewed behavior logprob definition")
        if sampling.get("n") != 1:
            raise ParityError("Multiple completions are not supported")
        for name in ("temperature", "top_p", "top_k"):
            if sampling.get(name) != getattr(self.contract, name):
                raise ParityError(f"Effective sampling differs from contract: {name}: "
                                  f"actual={sampling.get(name)!r}, expected={getattr(self.contract, name)!r}")
        if not 0 < sampling.get("max_tokens", 0) <= self.contract.max_output:
            raise ParityError("Effective output budget differs from contract")
        self.input = {"prompt_token_ids": list(prompt_ids), "features": features,
                      "sampling": sampling, "engine_request_id": engine_request_id,
                      "weight_version": self.weight_version, "logprobs_mode": logprobs_mode,
                      "contract_sha256": self.contract.sha256}
        self.input["input_sha256"] = digest(self.input)
        self.journal.event("engine_input", self.input)

    def output(self, ids, logprobs, *, cumulative, finished, finish_reason, weight_version):
        if self.input is None or self.finished:
            raise ParityError("Engine output outside an active request")
        if weight_version != self.weight_version:
            raise ParityError("Weights changed inside a generation")
        if type(ids) is not list or any(type(i) is not int or i < 0 for i in ids):
            raise ParityError("Invalid raw completion IDs")
        if not isinstance(logprobs, list) or len(ids) != len(logprobs):
            raise ParityError("Missing/misaligned behavior logprobs")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= -9999 or v > 1e-5 for v in logprobs):
            raise ParityError("Invalid behavior logprob; no padding or sentinels allowed")
        if cumulative:
            if ids[:len(self.ids)] != self.ids or logprobs[:len(self.logprobs)] != self.logprobs:
                raise ParityError("Cumulative engine output rewrote previous tokens")
            new_ids, new_lps = ids[len(self.ids):], logprobs[len(self.logprobs):]
        else:
            new_ids, new_lps = ids, logprobs
        self.ids.extend(new_ids)
        self.logprobs.extend(new_lps)
        if len(self.ids) > self.contract.max_output or len(self.input["prompt_token_ids"]) + len(self.ids) > self.contract.max_context:
            raise ParityError("Generated sequence exceeds contract")
        self.finished = bool(finished)
        self.journal.event("engine_output", {"token_ids": new_ids, "logprobs": new_lps,
            "finished": self.finished, "finish_reason": finish_reason, "weight_version": weight_version})


def load_turn(store, request_id, contract):
    events, sealed = store.read_request(request_id)
    return turn_from_events(store, request_id, contract, events, sealed)


def turn_from_events(store, request_id, contract, events, sealed):
    """Validate the same sealed bytes without another remote filesystem read."""
    if sealed.get('request_id') != request_id or sealed.get('events_sha256') != digest(events):
        raise ParityError('Request journal identity/hash mismatch')
    if sealed["status"] != "complete":
        raise ParityError("Incomplete/failed request cannot enter training")
    requests = [e["data"] for e in events if e["kind"] == "request"]
    inputs = [e["data"] for e in events if e["kind"] == "engine_input"]
    outputs = [e["data"] for e in events if e["kind"] == "engine_output"]
    wire = [e["data"] for e in events if e["kind"] == "wire_complete"]
    if len(requests) != 1 or len(inputs) != 1 or not outputs or len(wire) != 1 or not wire[0].get("done"):
        raise ParityError("Missing actual request/input/output/stream completion")
    meta = requests[0]["metadata"]
    if meta.get("contract_sha256") != contract.sha256 or inputs[0]["contract_sha256"] != contract.sha256:
        raise ParityError("Runtime/training contract mismatch")
    if meta.get("agent") != "build" or meta.get("primary") is not True:
        raise ParityError("Auxiliary requests are archived but are not policy turns")
    if any(o["finished"] for o in outputs[:-1]) or not outputs[-1]["finished"]:
        raise ParityError("Engine termination mismatch")
    if outputs[-1]["finish_reason"] not in ("stop", "length"):
        raise ParityError("Aborted engine output is not a completed action")
    input_data = dict(inputs[0])
    expected = input_data.pop("input_sha256")
    if digest(input_data) != expected:
        raise ParityError("Input digest mismatch")
    if any(o["weight_version"] != inputs[0]["weight_version"] for o in outputs):
        raise ParityError("Mixed behavior weights inside one request")
    ids = [tid for output in outputs for tid in output["token_ids"]]
    lps = [lp for output in outputs for lp in output["logprobs"]]
    require_ids(ids, "raw completion IDs")
    require_ids(inputs[0]["prompt_token_ids"], "raw prompt IDs")
    if len(ids) != len(lps) or any(not math.isfinite(lp) or lp <= -9999 or lp > 1e-5 for lp in lps):
        raise ParityError("Bad behavior probability alignment")
    for feature in inputs[0]["features"]:
        if feature["modality"] != "image":
            raise ParityError("Native audio/video tensor training not implemented")
        for tensor in feature["data"].values():
            store.load_tensor(tensor)
    return {"request_id": request_id, "metadata": meta, "input": inputs[0],
            "response_ids": ids, "rollout_logprobs": lps, "loss_mask": [1] * len(ids),
            "finish_reason": outputs[-1]["finish_reason"]}
