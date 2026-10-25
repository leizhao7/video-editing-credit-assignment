"""Source-checked additions to the VERIFIED frozen worker and fused VLM head.

Never copy this worktree's older worker over production source. Insert opt-in
branches ahead of sampled-token consumers, retaining all existing fixes.
"""
import ast

from .answers import require

MARKER = "# ANSWER_OPSD_TOPK_DISTRIBUTION_V1"


def _insert_method(source, class_name, method, body):
    tree = ast.parse(source)
    cls = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    require(len(cls) == 1, "Unknown worker class anchor")
    matches = [node for node in cls[0].body if isinstance(node, ast.FunctionDef) and node.name == method]
    require(len(matches) == 1, "Unknown worker method anchor")
    node = matches[0]
    require(any(arg.arg == "data" for arg in node.args.args), "Worker data transport changed")
    index = 1 if ast.get_docstring(node, clean=False) is not None else 0
    at = node.body[index].lineno - 1
    lines = source.splitlines(keepends=True)
    return "".join(lines[:at]) + body + "".join(lines[at:])


def extend_topk_source(kind, source):
    require(MARKER not in source, "Already extended top-K source")
    require("OCRL_EXACT_MEDIA_V7" in source, "Top-K requires the existing exact-media production fixes")
    if kind == "worker":
        for cls, method, target in (("PolicyWorkerBase", "forward", "score"), ("RefWorkerBase", "forward", "teacher")):
            source = _insert_method(source, cls, method,
                '        if (data.metadata or {}).get("opsd_topk") is not None:\n'
                '            from examples.train.answer_self_distillation.topk_worker import forward\n'
                f'            return forward(self, data, "{target}")\n')
        source = _insert_method(source, "PolicyWorkerBase", "forward_backward",
            '        if self.cfg.algorithm.policy_loss_type == "answer_opsd_topk":\n'
            '            from examples.train.answer_self_distillation.topk_worker import forward_backward\n'
            '            return forward_backward(self, data, loss_fn, loss_fn_config, return_per_token_outputs)\n')
    elif kind == "fused":
        signature = "    mm_token_type_ids=None,\n):"
        anchor = "    weight = _full_weight(head.weight).to(hidden.dtype)\n"
        require(source.count(signature) == source.count(anchor) == 1, "Unreviewed fused VLM forward anchors")
        source = source.replace(signature, "    mm_token_type_ids=None,\n    opsd_request=None,\n):")
        source = source.replace(anchor, anchor +
            '    if opsd_request is not None:\n'
            '        from examples.train.answer_self_distillation.topk import project_response\n'
            '        return project_response(hidden, weight, sequences, num_actions, opsd_request)\n')
    else:
        raise ValueError("Unknown top-K source target")
    source += "\n" + MARKER + "\n"
    compile(source, f"<answer-opsd-{kind}>", "exec")
    return source
