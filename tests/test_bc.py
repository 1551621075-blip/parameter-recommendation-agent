import hashlib
import json
import os
from copy import deepcopy

import numpy as np
import pytest

from parameter_agent import bc
from parameter_agent.domain import PARAMETER_NAMES, ProcessContext, check_candidate, digest
from parameter_agent.errors import DomainError
from parameter_agent.fixtures import demo_contexts, generate_demonstrations

NOW = 1_750_010_000_000
ZERO = dict.fromkeys(PARAMETER_NAMES, "0")


@pytest.fixture(scope="module")
def records():
    return generate_demonstrations()


@pytest.fixture(scope="module")
def trained(records):
    return bc.train_model(records)


def rehash(model):
    model["model_hash"] = digest({key: value for key, value in model.items() if key != "model_hash"})
    return model


def independent_features(context):
    return [*(float(context["current"][name]) for name in PARAMETER_NAMES),
            context["material_viscosity"], context["ambient_c"], context["severity"],
            *(float(context["defect"] == name) for name in ("short_shot", "flash", "warpage", "normal"))]


def independent_forward(model, contexts):
    x = np.array([independent_features(ctx) for ctx in contexts])
    x = (x - np.array(model["normalization"]["mean"])) / np.array(model["normalization"]["std"])
    w = {name: np.array(values) for name, values in model["weights"].items()}
    h1 = np.tanh(x @ w["w1"] + w["b1"])
    h2 = np.tanh(h1 @ w["w2"] + w["b2"])
    return h2 @ w["wc"] + w["bc"], np.tanh(h2 @ w["wr"] + w["br"])


def test_actual_training_is_meaningful_and_heldout_metrics_honest(trained):
    metrics = trained["metrics"]
    assert metrics["training"]["final_loss"] < metrics["training"]["initial_loss"] * 0.2
    assert len(metrics["training"]["history"]) == 7
    assert metrics["training"]["history"][-1]["epoch"] == 240
    test = metrics["test"]
    assert test["combination_accuracy"] > 0.8
    assert test["normalized_delta_mae"] < 0.12
    assert test["combination_accuracy"] > test["no_action_combination_accuracy"]
    assert test["evaluation_passes"] == 1
    assert test["real_world_effect"] == "not measured"
    assert set(test["synthetic_expected_defect_rate"]) == {"expert", "bc_policy", "simple_rule", "no_action"}
    assert metrics["training"]["test_used_for_tuning"] is False
    assert set(metrics["interpretation"]) == {"expert_objective", "bc", "effect"}


def test_exact_group_split_lineage_and_train_only_statistics(records, trained):
    model = trained["model"]
    lineage = model["lineage"]
    groups = {part: set(entry["groups"]) for part, entry in lineage.items()}
    assert [len(groups[part]) for part in ("train", "calibration", "test")] == [72, 24, 24]
    assert not (groups["train"] & groups["calibration"] or groups["train"] & groups["test"] or groups["test"] & groups["calibration"])
    assert set.union(*groups.values()) == {row["group_id"] for row in records}
    train_rows = [row for row in records if row["group_id"] in groups["train"]]
    features = np.array([independent_features(row["context"]) for row in train_rows])
    np.testing.assert_allclose(model["normalization"]["mean"], features.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(model["normalization"]["std"], features.std(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model["support"]["min"], features[:, :7].min(axis=0))
    np.testing.assert_array_equal(model["support"]["max"], features[:, :7].max(axis=0))
    devices = {}
    for row in records:
        part = next(part for part, ids in groups.items() if row["group_id"] in ids)
        machine = row["context"]["machine_id"]
        assert machine not in devices or devices[machine] == part
        devices[machine] = part
    for part, ids in groups.items():
        rows = sorted((row for row in records if row["group_id"] in ids), key=lambda row: (row["group_id"], row["context"]["context_id"]))
        assert lineage[part]["rows"] == len(rows)
        assert lineage[part]["records_hash"] == digest([digest(row) for row in rows])


def test_calibration_and_test_arithmetic_independent_oracle(records, trained):
    model = trained["model"]
    rows = sorted((row for row in records if row["group_id"] in model["lineage"]["calibration"]["groups"]),
                  key=lambda row: (row["group_id"], row["context"]["context_id"]))
    logits, _ = independent_forward(model, [row["context"] for row in rows])
    labels = [model["masks"].index(sum(1 << index for index, name in enumerate(PARAMETER_NAMES) if float(row["delta"][name]) != 0)) for row in rows]
    def nll(temperature):
        scaled = logits / temperature
        log_normalizer = np.log(np.exp(scaled - scaled.max(axis=1, keepdims=True)).sum(axis=1)) + scaled.max(axis=1)
        return float(np.mean(log_normalizer - scaled[np.arange(len(rows)), labels]))
    temperatures = sorted(set(np.geomspace(0.5, 3.0, 61).tolist() + [1.0]))
    losses = [nll(temperature) for temperature in temperatures]
    expected = temperatures[int(np.argmin(losses))]
    assert model["calibration"]["temperature"] == expected
    assert model["calibration"]["nll_after"] == pytest.approx(min(losses), abs=1e-12)
    assert model["calibration"]["nll_after"] <= model["calibration"]["nll_before"] + 1e-12
    test = [row for row in records if row["group_id"] in model["lineage"]["test"]["groups"]]
    logits, regression = independent_forward(model, [row["context"] for row in test])
    masks = np.array(model["masks"])[logits.argmax(axis=1)]
    truth = np.array([[float(row["delta"][name]) for name in PARAMETER_NAMES] for row in test])
    true_masks = ((truth != 0) * (1 << np.arange(4))).sum(axis=1)
    error = np.abs(regression * [6, 10, 4, 3] * ((masks[:, None] & (1 << np.arange(4))) != 0) - truth)
    assert trained["metrics"]["test"]["combination_accuracy"] == float(np.mean(masks == true_masks))
    for index, name in enumerate(PARAMETER_NAMES):
        assert trained["metrics"]["test"]["delta_mae"][name] == pytest.approx(error[:, index].mean(), abs=1e-12)


def test_counterfactual_test_never_tunes_weights_calibration_or_support(records, trained, monkeypatch):
    changed = deepcopy(records)
    test_groups = set(trained["model"]["lineage"]["test"]["groups"])
    for row in changed:
        if row["group_id"] in test_groups:
            row["delta"] = dict(ZERO)
            row["context"]["ambient_c"] = 44.0
    calls = []
    original = bc._evaluate
    def spy(model, rows):
        calls.append((model["model_hash"], len(rows)))
        return original(model, rows)
    monkeypatch.setattr(bc, "_evaluate", spy)
    alternate = bc.train_model(changed)
    assert len(calls) == 1
    original_model, alternate_model = trained["model"], alternate["model"]
    for field in ("weights", "normalization", "calibration", "support", "training"):
        assert original_model[field] == alternate_model[field]
    assert original_model["lineage"]["test"]["records_hash"] != alternate_model["lineage"]["test"]["records_hash"]
    assert alternate["metrics"]["test"]["combination_accuracy"] != trained["metrics"]["test"]["combination_accuracy"]


def test_calibration_labels_only_affect_calibration_not_training(records, trained):
    changed = deepcopy(records)
    groups = set(trained["model"]["lineage"]["calibration"]["groups"])
    for row in changed:
        if row["group_id"] in groups:
            row["delta"] = dict(ZERO)
    alternate = bc.train_model(changed)["model"]
    for field in ("weights", "normalization", "support", "training"):
        assert alternate[field] == trained["model"][field]
    assert alternate["calibration"]["temperature"] != trained["model"]["calibration"]["temperature"]
    assert alternate["lineage"]["test"] == trained["model"]["lineage"]["test"]


def test_training_deterministic_order_independent(records, trained):
    alternate = bc.train_model(list(reversed(records)))
    assert alternate == trained


def test_bad_records_and_split_lineage_rejected(records):
    changed = deepcopy(records[:120])
    changed[12]["context"]["machine_id"] = changed[0]["context"]["machine_id"]
    with pytest.raises(DomainError, match="lineage"):
        bc.train_model(changed)
    changed = deepcopy(records[:120])
    changed[1]["context"]["context_id"] = changed[0]["context"]["context_id"]
    with pytest.raises(DomainError):
        bc.train_model(changed)
    changed = deepcopy(records[:120])
    changed[0]["delta"]["temperature"] = "-1"
    with pytest.raises(DomainError):
        bc.train_model(changed)
    changed = deepcopy(records[:120])
    changed[0]["delta"] = {**ZERO, "temperature": "0.5"}
    with pytest.raises(DomainError):
        bc.train_model(changed)
    with pytest.raises(DomainError):
        bc.train_model(records[:132])  # 11 groups cannot give exact ratio
    with pytest.raises(DomainError):
        bc.train_model([])
    with pytest.raises(DomainError):
        bc.train_model(records, seed=True)


def test_ordinary_demos_safe_and_predictions_are_actual_neural_outputs(trained):
    model = trained["model"]
    for defect, context in demo_contexts(NOW).items():
        prediction = bc.predict(model, context, NOW)
        if defect in {"short_shot", "flash", "warpage"}:
            assert prediction["status"] == "ok"
            assert check_candidate(context, prediction["raw_delta"], NOW)["eligible"]
        else:
            assert prediction["status"] == "abstain"
        assert prediction["context_hash"] == digest(context.model_dump(mode="json"))
        assert prediction["model_hash"] == model["model_hash"]
        assert json.loads(json.dumps(prediction)) == prediction
        if defect in {"missing", "ood"}:
            continue
        logits, regression = independent_forward(model, [context.model_dump(mode="json")])
        mask = model["masks"][int(np.argmax(logits[0]))]
        expected = {name: f"{regression[0, axis] * [6, 10, 4, 3][axis]:.6f}" if mask & (1 << axis) else "0"
                    for axis, name in enumerate(PARAMETER_NAMES)}
        assert prediction["raw_delta"] == expected
        assert prediction["combination"] == [name for axis, name in enumerate(PARAMETER_NAMES) if mask & (1 << axis)]


def test_abstains_missing_units_ood_expired_and_unknown_source(trained):
    model = trained["model"]
    demos = demo_contexts(NOW)
    assert "missing_material_viscosity" in bc.predict(model, demos["missing"], NOW)["issues"]
    assert "ood_material_viscosity" in bc.predict(model, demos["ood"], NOW)["issues"]
    for field, value, issue in (("units", {"temperature": "F"}, "units_mismatch"),
                                 ("observed_at_ms", NOW - 86_400_001, "context_expired"),
                                 ("observed_at_ms", NOW + 1, "context_from_future"),
                                 ("plant_id", "synthetic-unknown", "ood_plant_id"),
                                 ("material_id", "synthetic-other", "ood_material_id"),
                                 ("process_version", "v2", "unsupported_process_version")):
        data = demos["short_shot"].model_dump(mode="json")
        data[field] = value
        prediction = bc.predict(model, ProcessContext.model_validate(data), NOW)
        assert prediction["status"] == "abstain" and issue in prediction["issues"]
    with pytest.raises(DomainError):
        bc.predict(None, demos["short_shot"], NOW)


def test_low_confidence_and_invalid_neural_sign_are_not_repaired(trained):
    model = deepcopy(trained["model"])
    model["weights"]["wc"] = np.zeros((32, 11)).tolist()
    model["weights"]["bc"] = np.zeros(11).tolist()
    rehash(model)
    prediction = bc.predict(model, demo_contexts(NOW)["short_shot"], NOW)
    assert prediction["status"] == "abstain"
    assert "low_confidence" in prediction["issues"]
    assert prediction["confidence"] == pytest.approx(1 / 11)
    model["weights"]["bc"][model["masks"].index(1)] = 100
    model["weights"]["wr"] = np.zeros((32, 4)).tolist()
    model["weights"]["br"] = [-1.0, 0.0, 0.0, 0.0]
    rehash(model)
    prediction = bc.predict(model, demo_contexts(NOW)["short_shot"], NOW)
    assert prediction["confidence"] > 0.99
    assert prediction["status"] == "abstain"
    assert prediction["combination"] == ["temperature"]
    assert float(prediction["raw_delta"]["temperature"]) < 0
    assert "raw_temperature_direction" in prediction["issues"]
    # NORMAL is not silently patched to no-action either: it must hard-abstain.
    prediction = bc.predict(model, demo_contexts(NOW)["normal"], NOW)
    assert prediction["raw_delta"]["temperature"] != "0"
    assert prediction["status"] == "abstain"


def test_hash_excludes_self_with_independent_json_oracle(trained):
    model = trained["model"]
    payload = {key: value for key, value in model.items() if key != "model_hash"}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    assert model["model_hash"] == hashlib.sha256(encoded).hexdigest()
    assert model["model_hash"] != digest(model)
    bc.validate_model(model)


@pytest.mark.parametrize("attack", [
    "hash", "shape", "ragged", "bool", "string", "huge", "nan", "inf", "version", "features", "masks",
    "extra", "normalization", "std", "lineage", "ratio", "calibration_rows", "temperature", "threshold", "support", "tuning",
])
def test_strict_model_attacks_even_when_rehashed(trained, attack):
    model = deepcopy(trained["model"])
    if attack == "hash":
        model["weights"]["w1"][0][0] += 0.01
    elif attack == "shape":
        model["weights"]["w1"].pop()
    elif attack == "ragged":
        model["weights"]["w1"][0].pop()
    elif attack in {"bool", "string", "huge", "nan", "inf"}:
        model["weights"]["w1"][0][0] = {"bool": True, "string": "0", "huge": 1e7, "nan": float("nan"), "inf": float("inf")}[attack]
    elif attack == "version":
        model["process_version"] = "synthetic-molding-v0"
    elif attack == "features":
        model["feature_order"] = model["feature_order"][::-1]
    elif attack == "masks":
        model["masks"][1] = True
    elif attack == "extra":
        model["execute"] = True
    elif attack == "normalization":
        model["normalization"]["source"] = "test"
    elif attack == "std":
        model["normalization"]["std"][0] = 0
    elif attack == "lineage":
        model["lineage"]["test"]["groups"][0] = model["lineage"]["train"]["groups"][0]
    elif attack == "ratio":
        model["lineage"]["train"]["groups"].pop()
    elif attack == "calibration_rows":
        model["calibration"]["rows"] += 1
    elif attack == "temperature":
        model["calibration"]["temperature"] = -1
    elif attack == "threshold":
        model["calibration"]["confidence_threshold"] = 0.0
    elif attack == "support":
        model["support"]["margin_fraction"] = 100.0
    elif attack == "tuning":
        model["training"]["test_used_for_tuning"] = True
    if attack not in {"hash", "nan", "inf"}:
        rehash(model)
    with pytest.raises(DomainError):
        bc.validate_model(model)


def test_model_io_roundtrip_new_path_only_and_permissions(trained, tmp_path):
    path = tmp_path / "model.json"
    bc.save_model(path, trained["model"])
    assert bc.load_model(path) == trained["model"]
    assert path.stat().st_mode & 0o777 == 0o600
    original = path.read_bytes()
    with pytest.raises(DomainError):
        bc.save_model(path, trained["model"])
    assert path.read_bytes() == original
    assert b"pickle" not in path.read_bytes()


def test_io_rejects_symlinks_including_ancestor(trained, tmp_path):
    target = tmp_path / "safe.json"
    bc.save_model(target, trained["model"])
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(DomainError):
        bc.load_model(link)
    with pytest.raises(DomainError):
        bc.save_model(link, trained["model"])
    folder = tmp_path / "real"
    folder.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(folder, target_is_directory=True)
    with pytest.raises(DomainError):
        bc.save_model(alias / "new.json", trained["model"])
    assert not (folder / "new.json").exists()
    with pytest.raises(DomainError):
        bc.load_model(tmp_path / "real" / ".." / "safe.json")


@pytest.mark.parametrize("payload", [b'{"model_hash":"a","model_hash":"b"}', b'{"value":NaN}', b'{"value":Infinity}',
                                       b'{"value":1e999}', b'\xff\xfe', b'\x80\x04pickle', b'{} trailing', b'[]',
                                       b'[' * 2000 + b']' * 2000, b'x' * (bc.MAX_MODEL_BYTES + 1)])
def test_bounded_strict_json_load_attacks(payload, tmp_path):
    path = tmp_path / "attack.json"
    path.write_bytes(payload)
    with pytest.raises(DomainError):
        bc.load_model(path)


def test_io_rejects_nonregular_files_without_blocking(tmp_path):
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    with pytest.raises(DomainError):
        bc.load_model(pipe)
    with pytest.raises(DomainError):
        bc.load_model(tmp_path)
