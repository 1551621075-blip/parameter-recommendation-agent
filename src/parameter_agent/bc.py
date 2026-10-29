"""NumPy behavioral cloning, explicit held-out calibration, bounded JSON artifacts.

An MLP predicts a combination mask and four signed normalized increments. A mask
zeros unselected regression heads; no defect-specific label overrides or clamps
are applied after inference. Hard domain gates remain outside learned weights.
Content hashes detect corruption, not authorship: storage must authorize models.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
from pathlib import Path

import numpy as np

from .domain import (
    CONSTRAINT_VERSION,
    PARAMETER_NAMES,
    PROCESS_VERSION,
    ParameterValues,
    ProcessContext,
    canonical,
    check_candidate,
    digest,
    validate_context,
)
from .errors import DomainError
from .fixtures import ZERO, _expected_quality

FEATURE_ORDER = [*PARAMETER_NAMES, "material_viscosity", "ambient_c", "severity",
                 "defect_short_shot", "defect_flash", "defect_warpage", "defect_normal"]
MASKS = [0, 1, 2, 4, 8, 3, 5, 9, 6, 10, 12]
MODEL_VERSION = "numpy-bc-mlp-v1"
SCALES = np.array([6.0, 10.0, 4.0, 3.0], dtype=np.float64)
CONFIDENCE_THRESHOLD = 0.55  # fixed in advance, never optimized on held-out test
MAX_MODEL_BYTES = 1_048_576
SHAPES = {"w1": (11, 48), "b1": (48,), "w2": (48, 32), "b2": (32,),
          "wc": (32, 11), "bc": (11,), "wr": (32, 4), "br": (4,)}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


def _fail(message):
    raise DomainError("invalid_model", message, 422)


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        _fail(f"invalid {name} fields")


def _number(value, name, low=-1e6, high=1e6):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        _fail(f"invalid finite {name}")


def _array(value, shape, name):
    def walk(items, dims):
        if type(items) is not list or len(items) != dims[0]:
            _fail(f"invalid {name} shape")
        if len(dims) == 1:
            for item in items:
                _number(item, name)
        else:
            for item in items:
                walk(item, dims[1:])
    walk(value, shape)
    return np.asarray(value, dtype=np.float64)


def _features(context):
    return [*(float(getattr(context.current, name)) for name in PARAMETER_NAMES),
            context.material_viscosity, context.ambient_c, context.severity,
            *(float(context.defect == defect) for defect in ("short_shot", "flash", "warpage", "normal"))]


def _mask(delta):
    return sum(1 << index for index, name in enumerate(PARAMETER_NAMES) if float(delta[name]) != 0)


def _softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _forward(weights, features):
    h1 = np.tanh(features @ weights["w1"] + weights["b1"])
    h2 = np.tanh(h1 @ weights["w2"] + weights["b2"])
    return h1, h2, h2 @ weights["wc"] + weights["bc"], np.tanh(h2 @ weights["wr"] + weights["br"])


def _loss(weights, x, labels, targets, active):
    _, _, logits, regression = _forward(weights, x)
    probabilities = _softmax(logits)
    classification = -float(np.log(np.maximum(probabilities[np.arange(len(x)), labels], 1e-300)).mean())
    regression_loss = float(np.sum((regression - targets) ** 2 * active) / max(1, active.sum()))
    return classification + 2 * regression_loss, classification, regression_loss


def _calibrate(logits, labels):
    """Only calibration logits enter this fixed one-dimensional NLL search."""
    temperatures = np.unique(np.append(np.geomspace(0.5, 3.0, 61), 1.0))
    losses = []
    for temperature in temperatures:
        probabilities = _softmax(logits / temperature)
        losses.append(-float(np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-300)).mean()))
    best = int(np.argmin(losses))
    before = -float(np.log(np.maximum(_softmax(logits)[np.arange(len(labels)), labels], 1e-300)).mean())
    return {"temperature": float(temperatures[best]), "nll_before": before, "nll_after": losses[best],
            "method": "calibration-only-temperature-grid-nll", "rows": len(labels),
            "confidence_threshold": CONFIDENCE_THRESHOLD}


def _prepare(records, seed):
    if type(records) is not list or not 40 <= len(records) <= 10_000:
        raise DomainError("invalid_demonstrations", "requires 40..10000 original demonstration records", 422)
    by_group, seen_contexts, machines = {}, set(), {}
    for record in records:
        if type(record) is not dict or set(record) != {"group_id", "context", "delta"}:
            raise DomainError("invalid_demonstrations", "invalid demonstration fields", 422)
        group = record["group_id"]
        if type(group) is not str or not _ID.fullmatch(group):
            raise DomainError("invalid_demonstrations", "invalid group ID", 422)
        try:
            context = ProcessContext.model_validate(record["context"])
            values = ParameterValues.model_validate(record["delta"]).model_dump(mode="json")
            issues = validate_context(context, context.observed_at_ms)
        except (ValueError, TypeError) as exc:
            raise DomainError("invalid_demonstrations", "invalid demonstration schema", 422) from exc
        if issues or context.context_id in seen_contexts:
            raise DomainError("invalid_demonstrations", "invalid/duplicate demonstration context", 422)
        seen_contexts.add(context.context_id)
        lineage = (context.plant_id, context.machine_id)
        if lineage in machines and machines[lineage] != group:
            raise DomainError("split_lineage", "one device lineage may not span group IDs", 422)
        machines[lineage] = group
        candidate = check_candidate(context, values, context.observed_at_ms)
        if not candidate["eligible"] and candidate["issues"] != ["no_change"]:
            raise DomainError("invalid_demonstrations", "expert action violates hard gates", 422)
        # Labels must already be lattice values; don't silently change supervision.
        if any(float(values[name]) != float(candidate["delta"][name]) for name in PARAMETER_NAMES):
            raise DomainError("invalid_demonstrations", "expert labels must be grid-aligned", 422)
        by_group.setdefault(group, []).append((context, values, digest(record)))
    groups = sorted(by_group)
    if not 10 <= len(groups) <= 500 or len(groups) % 5:
        raise DomainError("invalid_split", "10..500 groups in multiples of five give exact 60/20/20", 422)
    rng = np.random.default_rng(seed)
    shuffled = [groups[index] for index in rng.permutation(len(groups))]
    cut1, cut2 = len(groups) * 3 // 5, len(groups) * 4 // 5
    split_ids = {"train": sorted(shuffled[:cut1]), "calibration": sorted(shuffled[cut1:cut2]),
                 "test": sorted(shuffled[cut2:])}
    partitions, lineage = {}, {}
    for part, ids in split_ids.items():
        rows = [row for group in ids for row in sorted(by_group[group], key=lambda row: row[0].context_id)]
        partitions[part] = rows
        lineage[part] = {"groups": ids, "rows": len(rows), "records_hash": digest([row[2] for row in rows])}
    return partitions, lineage


def train_model(records: list[dict], *, seed: int = 41723) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise DomainError("invalid_seed", "seed must be a bounded nonnegative integer", 422)
    partitions, lineage = _prepare(records, seed)
    train = partitions["train"]
    raw = np.asarray([_features(row[0]) for row in train], dtype=np.float64)
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    std = np.maximum(std, 1e-6)
    x = (raw - mean) / std
    labels = np.array([MASKS.index(_mask(row[1])) for row in train])
    targets = np.array([[float(row[1][name]) for name in PARAMETER_NAMES] for row in train]) / SCALES
    active = (targets != 0).astype(np.float64)
    rng = np.random.default_rng(seed)
    weights = {name: (rng.normal(0, math.sqrt(1 / shape[0]), shape) if len(shape) == 2
                      else np.zeros(shape, dtype=np.float64)) for name, shape in SHAPES.items()}
    initial_loss = _loss(weights, x, labels, targets, active)[0]
    first = {name: np.zeros_like(value) for name, value in weights.items()}
    second = {name: np.zeros_like(value) for name, value in weights.items()}
    history, step = [], 0
    # Fixed training schedule: no test peeking, no test-based early stopping.
    epochs, batch_size = 240, 128
    for epoch in range(epochs):
        permutation = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            indexes = permutation[start:start + batch_size]
            batch, lab, target, enabled = x[indexes], labels[indexes], targets[indexes], active[indexes]
            h1, h2, logits, regression = _forward(weights, batch)
            dc = _softmax(logits)
            dc[np.arange(len(batch)), lab] -= 1
            dc /= len(batch)
            dr = 4 * (regression - target) * enabled / max(1, enabled.sum())
            dr *= 1 - regression ** 2
            gradients = {"wc": h2.T @ dc, "bc": dc.sum(axis=0), "wr": h2.T @ dr, "br": dr.sum(axis=0)}
            dh2 = (dc @ weights["wc"].T + dr @ weights["wr"].T) * (1 - h2 ** 2)
            gradients.update(w2=h1.T @ dh2, b2=dh2.sum(axis=0))
            dh1 = (dh2 @ weights["w2"].T) * (1 - h1 ** 2)
            gradients.update(w1=batch.T @ dh1, b1=dh1.sum(axis=0))
            step += 1
            for name in weights:
                gradient = gradients[name] + (1e-4 * weights[name] if name.startswith("w") else 0)
                first[name] = 0.9 * first[name] + 0.1 * gradient
                second[name] = 0.999 * second[name] + 0.001 * gradient ** 2
                weights[name] -= 0.003 * (first[name] / (1 - 0.9 ** step)) / (np.sqrt(second[name] / (1 - 0.999 ** step)) + 1e-8)
        if epoch == 0 or (epoch + 1) % 40 == 0:
            loss, classification, regression = _loss(weights, x, labels, targets, active)
            history.append({"epoch": epoch + 1, "loss": loss, "classification_loss": classification,
                            "regression_loss": regression})
    calib = partitions["calibration"]
    calib_x = (np.array([_features(row[0]) for row in calib]) - mean) / std
    calib_labels = np.array([MASKS.index(_mask(row[1])) for row in calib])
    calibration = _calibrate(_forward(weights, calib_x)[2], calib_labels)
    support = {"min": raw[:, :7].min(axis=0).tolist(), "max": raw[:, :7].max(axis=0).tolist(),
               "margin_fraction": 0.15,
               "plant_ids": sorted({row[0].plant_id for row in train}),
               "material_ids": sorted({row[0].material_id for row in train}),
               "mold_ids": sorted({row[0].mold_id for row in train})}
    model = {"model_version": MODEL_VERSION, "process_version": PROCESS_VERSION,
             "constraint_version": CONSTRAINT_VERSION, "feature_order": list(FEATURE_ORDER),
             "parameter_names": list(PARAMETER_NAMES), "masks": list(MASKS),
             "architecture": [11, 48, 32, 11, 4], "seed": seed,
             "weights": {name: value.tolist() for name, value in weights.items()},
             "normalization": {"mean": mean.tolist(), "std": std.tolist(), "source": "train"},
             "support": support, "calibration": calibration, "lineage": lineage,
             "training": {"epochs": epochs, "batch_size": batch_size, "initial_loss": initial_loss,
                          "final_loss": history[-1]["loss"], "normalization_source": "train",
                          "test_used_for_tuning": False}}
    model["model_hash"] = digest(model)
    validate_model(model)
    # The single held-out evaluation happens only AFTER all weights/calibration
    # and artifact hashes are frozen. No result flows back into training.
    test_metrics = _evaluate(model, partitions["test"])
    return {"model": model, "metrics": {
        "synthetic": True, "split": lineage, "training": {**model["training"], "history": history},
        "calibration": calibration, "test": test_metrics,
        "interpretation": {
            "expert_objective": "Enumerated safe action minimizing toy expected defect rate plus intervention cost.",
            "bc": "Combination agreement and delta error measure imitation, not quality or causal benefit.",
            "effect": "Expected rates are from an original synthetic simulator; no real-world effect was measured.",
        }}}


def validate_model(model: dict) -> None:
    _keys(model, ("model_version", "process_version", "constraint_version", "feature_order", "parameter_names",
                  "masks", "architecture", "seed", "weights", "normalization", "support", "calibration",
                  "lineage", "training", "model_hash"), "model")
    encoded = canonical(model)
    if len(encoded) > MAX_MODEL_BYTES:
        _fail("artifact exceeds byte limit")
    for key, expected in (("model_version", MODEL_VERSION), ("process_version", PROCESS_VERSION),
                          ("constraint_version", CONSTRAINT_VERSION), ("feature_order", FEATURE_ORDER),
                          ("parameter_names", list(PARAMETER_NAMES)), ("masks", MASKS),
                          ("architecture", [11, 48, 32, 11, 4])):
        if model[key] != expected or type(model[key]) is not type(expected):
            _fail(f"unsupported {key}")
    if any(type(item) is not int for item in model["masks"] + model["architecture"]):
        _fail("integer architecture/masks required")
    if type(model["seed"]) is not int or not 0 <= model["seed"] < 2**63:
        _fail("invalid seed")
    _keys(model["weights"], SHAPES, "weights")
    for name, shape in SHAPES.items():
        _array(model["weights"][name], shape, name)
    norm = model["normalization"]
    _keys(norm, ("mean", "std", "source"), "normalization")
    _array(norm["mean"], (11,), "normalization mean")
    std = _array(norm["std"], (11,), "normalization std")
    if np.any(std < 1e-6) or norm["source"] != "train":
        _fail("invalid train-only normalization")
    support = model["support"]
    _keys(support, ("min", "max", "margin_fraction", "plant_ids", "material_ids", "mold_ids"), "support")
    low, high = _array(support["min"], (7,), "support min"), _array(support["max"], (7,), "support max")
    if np.any(low > high) or type(support["margin_fraction"]) is not float or support["margin_fraction"] != 0.15:
        _fail("invalid support interval")
    for field in ("plant_ids", "material_ids", "mold_ids"):
        ids = support[field]
        if type(ids) is not list or not 1 <= len(ids) <= 500 or any(type(item) is not str or not _ID.fullmatch(item) for item in ids) or ids != sorted(set(ids)):
            _fail("invalid support identifiers")
    calibration = model["calibration"]
    _keys(calibration, ("temperature", "nll_before", "nll_after", "method", "rows", "confidence_threshold"), "calibration")
    _number(calibration["temperature"], "calibration temperature", 0.5, 3.0)
    for field in ("nll_before", "nll_after"):
        _number(calibration[field], field, 0, 1000)
    if calibration["method"] != "calibration-only-temperature-grid-nll" or type(calibration["confidence_threshold"]) is not float or calibration["confidence_threshold"] != CONFIDENCE_THRESHOLD:
        _fail("invalid calibration policy")
    lineage = model["lineage"]
    _keys(lineage, ("train", "calibration", "test"), "lineage")
    seen = set()
    for part in ("train", "calibration", "test"):
        entry = lineage[part]
        _keys(entry, ("groups", "rows", "records_hash"), "partition")
        groups = entry["groups"]
        if type(groups) is not list or not 1 <= len(groups) <= 500 or any(type(item) is not str or not _ID.fullmatch(item) for item in groups) or groups != sorted(set(groups)) or seen.intersection(groups):
            _fail("invalid/disjoint group lineage")
        seen.update(groups)
        if type(entry["rows"]) is not int or not len(groups) <= entry["rows"] <= 10_000:
            _fail("invalid partition row count")
        if type(entry["records_hash"]) is not str or not _HEX.fullmatch(entry["records_hash"]):
            _fail("invalid partition hash")
    if not 10 <= len(seen) <= 500 or len(seen) % 5 or len(lineage["train"]["groups"]) * 5 != len(seen) * 3 or len(lineage["calibration"]["groups"]) * 5 != len(seen) or len(lineage["test"]["groups"]) * 5 != len(seen):
        _fail("expected exact 60/20/20 groups")
    if type(calibration["rows"]) is not int or calibration["rows"] != lineage["calibration"]["rows"] or sum(entry["rows"] for entry in lineage.values()) > 10_000:
        _fail("invalid calibration/total rows")
    training = model["training"]
    _keys(training, ("epochs", "batch_size", "initial_loss", "final_loss", "normalization_source", "test_used_for_tuning"), "training")
    if type(training["epochs"]) is not int or training["epochs"] != 240 or type(training["batch_size"]) is not int or training["batch_size"] != 128 or training["normalization_source"] != "train" or training["test_used_for_tuning"] is not False:
        _fail("unsupported training provenance")
    for field in ("initial_loss", "final_loss"):
        _number(training[field], field, 0, 1000)
    if type(model["model_hash"]) is not str or not _HEX.fullmatch(model["model_hash"]) or model["model_hash"] != digest({key: value for key, value in model.items() if key != "model_hash"}):
        _fail("artifact hash mismatch")


def _predict_validated(model, context, now_ms):
    issues = validate_context(context, now_ms)
    result = {"status": "abstain", "raw_delta": dict(ZERO), "combination": [], "confidence": 0.0,
              "issues": issues, "model_hash": model["model_hash"],
              "context_hash": digest(context.model_dump(mode="json"))}
    if issues:
        return result
    raw = np.asarray(_features(context), dtype=np.float64)
    support = model["support"]
    low, high = np.array(support["min"]), np.array(support["max"])
    margin = 0.15 * np.maximum(high - low, np.array([1, 5, 1, 1, 0.05, 1, 0.05]))
    for index in np.flatnonzero((raw[:7] < low - margin) | (raw[:7] > high + margin)):
        issues.append(f"ood_{FEATURE_ORDER[index]}")
    for attr, field in (("plant_id", "plant_ids"), ("material_id", "material_ids"), ("mold_id", "mold_ids")):
        if getattr(context, attr) not in support[field]:
            issues.append(f"ood_{attr}")
    if issues:
        return result
    weights = {name: np.asarray(value, dtype=np.float64) for name, value in model["weights"].items()}
    x = ((raw - np.array(model["normalization"]["mean"])) / np.array(model["normalization"]["std"]))[None, :]
    _, _, logits, regression = _forward(weights, x)
    probabilities = _softmax(logits / model["calibration"]["temperature"])[0]
    index = int(np.argmax(probabilities))
    combination = [name for axis, name in enumerate(PARAMETER_NAMES) if MASKS[index] & (1 << axis)]
    # Signed bounded regression is an architectural output (tanh), not repair.
    # Formatting keeps six decimal places; lattice quantization is domain-only.
    delta = {name: (f"{regression[0, axis] * SCALES[axis]:.6f}" if name in combination else "0")
             for axis, name in enumerate(PARAMETER_NAMES)}
    result.update(raw_delta=delta, combination=combination, confidence=float(probabilities[index]))
    if result["confidence"] < CONFIDENCE_THRESHOLD:
        issues.append("low_confidence")
    candidate = check_candidate(context, delta, now_ms)
    if not candidate["eligible"]:
        issues.extend(candidate["issues"])
    result["status"] = "abstain" if issues else "ok"
    return result


def predict(model: dict, context: ProcessContext, now_ms: int) -> dict:
    validate_model(model)
    return _predict_validated(model, context, now_ms)


def _evaluate(model, rows):
    """One final test pass. Baseline actions obey the SAME hard gates."""
    x = np.array([_features(row[0]) for row in rows])
    x = (x - np.array(model["normalization"]["mean"])) / np.array(model["normalization"]["std"])
    weights = {name: np.array(value) for name, value in model["weights"].items()}
    _, _, logits, regression = _forward(weights, x)
    predicted_masks = np.array(MASKS)[np.argmax(logits, axis=1)]
    truth_masks = np.array([_mask(row[1]) for row in rows])
    enabled = (predicted_masks[:, None] & (1 << np.arange(4))) != 0
    raw_delta = regression * SCALES * enabled
    truth = np.array([[float(row[1][name]) for name in PARAMETER_NAMES] for row in rows])
    errors = np.abs(raw_delta - truth)
    rates = {key: [] for key in ("no_action", "simple_rule", "expert", "bc_policy")}
    accepted, correct_accepted = 0, 0
    for index, (context, expert, _) in enumerate(rows):
        current = context.current.model_dump(mode="json")
        rates["no_action"].append(_expected_quality(context, current))
        rule = {"short_shot": (4, 0, 2, 0), "flash": (0, -5, -2, 0),
                "warpage": (-4, 0, 0, 2), "normal": (0, 0, 0, 0)}[context.defect]
        simple = dict(zip(PARAMETER_NAMES, map(str, rule), strict=True))
        prediction = _predict_validated(model, context, context.observed_at_ms)
        for key, proposal in (("expert", expert), ("simple_rule", simple), ("bc_policy", prediction["raw_delta"])):
            candidate = check_candidate(context, proposal, context.observed_at_ms)
            use = candidate["eligible"] and (key != "bc_policy" or prediction["status"] == "ok")
            rates[key].append(_expected_quality(context, candidate["target"] if use else current))
        if prediction["status"] == "ok":
            accepted += 1
            correct_accepted += int(predicted_masks[index] == truth_masks[index])
    return {"rows": len(rows), "evaluation_passes": 1,
            "combination_accuracy": float(np.mean(predicted_masks == truth_masks)),
            "delta_mae": {name: float(errors[:, index].mean()) for index, name in enumerate(PARAMETER_NAMES)},
            "normalized_delta_mae": float((errors / SCALES).mean()),
            "coverage": accepted / len(rows),
            "accepted_combination_accuracy": correct_accepted / accepted if accepted else None,
            "expert_no_action_fraction": float(np.mean(truth_masks == 0)),
            "predicted_no_action_fraction": float(np.mean(predicted_masks == 0)),
            "no_action_combination_accuracy": float(np.mean(truth_masks == 0)),
            "synthetic_expected_defect_rate": {key: float(np.mean(values)) for key, values in rates.items()},
            "synthetic_expected_improvement": {key: float(np.mean(rates["no_action"]) - np.mean(values))
                                               for key, values in rates.items()},
            "real_world_effect": "not measured"}


def _open_parent(path):
    """Walk an existing absolute directory using dirfds; reject symlink ancestors."""
    path = Path(path)
    if ".." in path.parts or not path.name:
        raise DomainError("unsafe_model_path", "path traversal is not allowed", 422)
    absolute = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:-1]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor, absolute.name
    except BaseException:
        os.close(descriptor)
        raise


def save_model(path: Path, model: dict):
    validate_model(model)
    payload = canonical(model)
    parent = None
    created = False
    try:
        parent, name = _open_parent(path)
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, ValueError) as exc:
        if created and parent is not None:
            os.unlink(name, dir_fd=parent)
        raise DomainError("model_io", "model save requires a new nonsymlink local path", 422) from exc
    finally:
        if parent is not None:
            os.close(parent)


def load_model(path: Path):
    parent = None
    try:
        parent, name = _open_parent(path)
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= MAX_MODEL_BYTES:
                raise ValueError("artifact size/type exceeds limit")
            payload = handle.read(MAX_MODEL_BYTES + 1)
            if len(payload) > MAX_MODEL_BYTES:
                raise ValueError("artifact exceeds byte limit")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        def constant(value):
            raise ValueError(f"nonfinite JSON constant: {value}")
        model = json.loads(payload.decode("utf-8", errors="strict"), object_pairs_hook=pairs, parse_constant=constant)
        validate_model(model)
        return model
    except (OSError, ValueError, UnicodeError, RecursionError, OverflowError) as exc:
        raise DomainError("model_io", "invalid bounded UTF-8 JSON model or unsafe path", 422) from exc
    finally:
        if parent is not None:
            os.close(parent)
