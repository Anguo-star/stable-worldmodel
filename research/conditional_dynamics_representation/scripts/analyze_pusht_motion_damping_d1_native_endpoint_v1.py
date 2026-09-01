#!/usr/bin/env python3
"""Compare the frozen D0, COJA, and D1 native Motion endpoints.

This is a read-only, Development-only endpoint comparison.  It intentionally
does not load a checkpoint, access a dataset, run a model, run CEM, or mutate
an existing result.  The formal D1 artifacts are accepted only when their
training, schedule, and Development identities form the pre-registered
single-data-change contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "research/conditional_dynamics_representation"
FORMAL_RUN = BASE / (
    "artifacts/pusht_motion_damping_d1_energy_stratified_native_v1/"
    "s14321_step8192_v2_final"
)
DEFAULT_OUTPUT = FORMAL_RUN / "endpoint_comparison_v1.json"

EXPECTED_INPUT_SHA256 = {
    "d0_diagnostic": "72f38a5c21d0f6d66d097cd8d1796ddbdbc581df234c3c569c49885c014f418f",
    "d0_result": "40fbd373491461caa153200f40d3e885cd8f302bdb356c3778a88fb5c012ee5e",
    "coja_diagnostic": "be54afff824248ffb1c8b50f62cc434b6f2a5b528c5b6169be0286767204f017",
    "coja_result": "a02ee3102e825e9021b86ae90015fe66d0b1c0f94da8c03a1c37aa63c5800fcb",
    "d1_training_report": "54625ddb4f603fc9bd0120593ec34421904bd3e699c37587a72ef17aba71e4ed",
    "d1_sidecar": "4b75c10fb5533f4d09c2b5f120fe96e796b76c7170b3d65c43fd81b1d713f043",
    "d1_result": "f9a2f3ad54cab8f14aa5127abed40f3e6f418feb1c38091b6c1b08af8b73dc75",
    "d1_diagnostic": "6cdd2d19d4c431f1dbc125ab85de1f905546cc9c60c69fbc8defbdf58695ca94",
}

INPUT_PATHS = {
    "d0_diagnostic": BASE / (
        "artifacts/native_conditional_signal_root_cause_v1/motion_damping/"
        "native_s14321_step8192_development_v1.json"
    ),
    "d0_result": BASE / (
        "artifacts/pusht_motion_damping_full_release_visible_joint_absolute_"
        "single_stage_native_control_step8192_v1/s14321_step8192_v1/"
        "development_response_analysis_v1.json"
    ),
    "coja_diagnostic": BASE / (
        "artifacts/native_conditional_signal_root_cause_v1/motion_damping/"
        "coja_s14321_step8192_development_v1.json"
    ),
    "coja_result": BASE / (
        "artifacts/pusht_motion_damping_full_release_visible_joint_absolute_"
        "single_stage_step8192_v1/s14321_step8192_v1/"
        "development_response_analysis_v1.json"
    ),
    "d1_training_report": FORMAL_RUN / "training_report.json",
    "d1_sidecar": FORMAL_RUN / (
        "pusht_motion_damping_d1_energy_stratified_native_method_v1.json"
    ),
    "d1_result": FORMAL_RUN / "development_response_analysis_v1.json",
    "d1_diagnostic": FORMAL_RUN / "conditional_signal_diagnostic_v1.json",
}

EXPECTED_SOURCE_SHA256 = "bafcd5406d70efcc2133014b287cc86f4543b5f615860fbcdf18a57b0ea7bab3"
EXPECTED_INITIAL_CHECKPOINT_SHA256 = "9f13b2c28bb909338047e08d62bf6eb16ea3616cb53e57cfa9b5072b43db1c59"
EXPECTED_MANIFEST_SHA256 = "48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b"
EXPECTED_LANCE_SHA256 = "64d43c931f106c2d53e3c3084e62381d2f2640c9d943e269475f3fb76aaa2de4"
EXPECTED_RELEASE_SHA256 = "1795717d8bfa1d1cfcc01a69931b6241b0e7759dcf43553a6c4cca225ec9326b"
EXPECTED_SCHEDULE_SHA256 = {
    "config.json": "45c5ed29aa950877c2ccc091ee041762c64b7ba060a24af1ea9ca220fa169171",
    "multiplicity.jsonl": "4be57c44b5e9485902edabdfbfb1c629b4bf433ed375ab00c783ae0ed187abb8",
    "receipt.json": "6c043e3b0169e721b0c54289e9b449b3c8690e9cc3c8c88270829d8c6bf04ad6",
    "schedule.jsonl": "e058384b66f129ace7e30dec354373fc14c885581bd57a5e86fd446be6f45b96",
    "summary.json": "64bbbc9a39649c9e8d8283006226a738615b0b440167d6d66396114a670faa47",
}
PAIR_COUNT = 256
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 14321
SIGN_FLIP_REPLICATES = 100_000
SIGN_FLIP_SEED = 14321
EXPECTED_DIAGNOSTIC_SOURCE_SHA256 = "62dbfd7af2e82a6166c2415576f9e0e1405f88e0351b59c010aa2af56eb89c54"
EXPECTED_EVALUATOR_SOURCE_SHA256 = "9a4e130d9301d7f9de906685b4845f9631fd5c5b23c43b62a6468f5f08a08f1a"
GAIN_GATE = 0.1
HISTORY_GATE = 0.55


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_sha256 = file_sha256


def _load_frozen(name: str, path: Path | None = None) -> dict[str, Any]:
    path = Path(path or INPUT_PATHS[name]).expanduser().resolve()
    observed = file_sha256(path)
    _require(observed == EXPECTED_INPUT_SHA256[name], f"{name} hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{name} is not a JSON object")
    return payload


def _finite(value: Any, label: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{label} is non-finite")
    return result


def _identity_from_diagnostic(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    _require(payload.get("schema_version") == 1, f"{label} schema changed")
    _require(payload.get("status") == "completed_zero_training_development_diagnostic", f"{label} incomplete")
    source = payload.get("source")
    _require(isinstance(source, Mapping), f"{label} source missing")
    _require(
        source.get("sha256") == EXPECTED_DIAGNOSTIC_SOURCE_SHA256,
        f"{label} source changed",
    )
    data = payload.get("data")
    _require(isinstance(data, Mapping), f"{label} data missing")
    description = data.get("description")
    identity = data.get("development_identity")
    _require(isinstance(description, Mapping) and isinstance(identity, Mapping), f"{label} Development identity missing")
    _require(data.get("public_test_opened") is False, f"{label} opened Public Test")
    _require(description.get("split") == "Development", f"{label} is not Development")
    _require(description.get("split_name") == "loader_validation", f"{label} split changed")
    _require(description.get("pair_count") == PAIR_COUNT, f"{label} pair count changed")
    _require(identity.get("split") == "loader_validation", f"{label} identity split changed")
    _require(identity.get("pair_count") == PAIR_COUNT and identity.get("passed") is True, f"{label} identity failed")
    _require(identity.get("data_manifest_sha256") == EXPECTED_MANIFEST_SHA256, f"{label} manifest changed")
    _require(identity.get("observed_data_manifest_sha256") == EXPECTED_MANIFEST_SHA256, f"{label} observed manifest changed")
    _require(identity.get("lance_table") == "loader_validation.lance", f"{label} table changed")
    _require(identity.get("lance_table_sha256") == EXPECTED_LANCE_SHA256, f"{label} table hash changed")
    _require(identity.get("observed_lance_table_sha256") == EXPECTED_LANCE_SHA256, f"{label} observed table hash changed")
    return {
        "manifest_sha256": identity["data_manifest_sha256"],
        "lance_table": identity["lance_table"],
        "lance_sha256": identity["lance_table_sha256"],
        "split": description["split"],
        "split_name": description["split_name"],
        "pair_count": PAIR_COUNT,
    }


def _record_arrays(payload: Mapping[str, Any], label: str) -> tuple[list[str], dict[str, np.ndarray]]:
    _identity_from_diagnostic(payload, label)
    records = payload.get("records")
    _require(isinstance(records, list) and len(records) == PAIR_COUNT, f"{label} records changed")
    pair_ids: list[str] = []
    values = {name: [] for name in ("g_swap", "response_gain", "nre")}
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"{label} record {index} malformed")
        raw_pair_id = record.get("pair_id")
        _require(isinstance(raw_pair_id, str) and raw_pair_id, f"{label} pair_id missing")
        pair_id = raw_pair_id
        _require(pair_id not in pair_ids, f"{label} pair_id duplicate")
        pair_ids.append(pair_id)
        g = _finite(record.get("g_swap"), f"{label} g_swap")
        gain = _finite(record.get("response_gain"), f"{label} response_gain")
        prediction = _finite(record.get("prediction_delta_energy"), f"{label} prediction energy")
        target = _finite(record.get("target_delta_energy"), f"{label} target energy")
        _require(target > 0.0, f"{label} target energy is not positive")
        nre = prediction / target + 1.0 - 2.0 * gain
        _require(math.isfinite(nre), f"{label} NRE is non-finite")
        values["g_swap"].append(g)
        values["response_gain"].append(gain)
        values["nre"].append(nre)
    return pair_ids, {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def _check_diagnostic_model(payload: Mapping[str, Any], label: str, *, expected_checkpoint: str | None = None) -> None:
    model = payload.get("model")
    _require(isinstance(model, Mapping), f"{label} model missing")
    adapter = model.get("adapter")
    _require(isinstance(adapter, Mapping), f"{label} adapter missing")
    _require(model.get("checkpoint_sha256") == adapter.get("checkpoint_sha256"), f"{label} checkpoint identity mismatch")
    if expected_checkpoint is not None:
        _require(model.get("checkpoint_sha256") == expected_checkpoint, f"{label} checkpoint changed")
    _require(adapter.get("protocol", {}).get("native_target_encoder") is True, f"{label} target encoder changed")


def _summary(payload: Mapping[str, Any], records: Mapping[str, np.ndarray], label: str) -> dict[str, Any]:
    result = payload.get("metrics")
    _require(isinstance(result, Mapping), f"{label} result metrics missing")
    g = records["g_swap"]
    gain = records["response_gain"]
    nre = records["nre"]
    latent = result.get("latent_response", {})
    return {
        "pair_count": int(len(g)),
        "g_swap": {"mean": float(np.mean(g)), "median": float(np.median(g)), "positive_fraction": float(np.mean(g > 0.0))},
        "response_gain": {"mean": float(np.mean(gain)), "median": float(np.median(gain)), "positive_fraction": float(np.mean(gain > 0.0))},
        "normalized_response_error": {"mean": float(np.mean(nre)), "median": float(np.median(nre))},
        "assignment": {
            "correct_history_preference_rate": _finite(result.get("correct_history_preference_rate"), f"{label} history rate"),
            "correct_rule_switch_rate": _finite(result.get("correct_rule_switch_rate"), f"{label} switch rate"),
            "two_real_future_target_selection_rate": _finite(result.get("two_real_future_target_selection_rate"), f"{label} future rate"),
            "worst_mode_target_selection_rate": _finite(result.get("worst_mode_target_selection_rate"), f"{label} worst rate"),
        },
        "latent_response": {
            "response_gain": _finite(latent.get("response_gain"), f"{label} aggregate gain"),
            "normalized_response_error": _finite(latent.get("normalized_response_error"), f"{label} aggregate NRE"),
            "aggregate_cosine_alignment": _finite(latent.get("aggregate_cosine_alignment"), f"{label} alignment"),
            "mean_pair_cosine_alignment": _finite(latent.get("mean_pair_cosine_alignment"), f"{label} pair alignment"),
        },
    }


def _randomization_summary(
    payload: Mapping[str, Any], label: str
) -> dict[str, float]:
    baselines = payload.get("random_pairing_baselines")
    _require(isinstance(baselines, Mapping), f"{label} randomization missing")
    sign_flip = baselines.get("within_pair_sign_flip")
    cross_query = baselines.get("cross_query_pairing")
    _require(
        isinstance(sign_flip, Mapping) and isinstance(cross_query, Mapping),
        f"{label} randomization malformed",
    )
    return {
        "within_pair_sign_flip_two_sided_p": _finite(
            sign_flip.get("two_sided_monte_carlo_p"), f"{label} sign-flip p"
        ),
        "cross_query_one_sided_p": _finite(
            cross_query.get("one_sided_monte_carlo_p"),
            f"{label} cross-query p",
        ),
    }


def _cluster_values(pair_ids: Sequence[str], values: np.ndarray) -> np.ndarray:
    _require(len(pair_ids) == len(values), "pair/value length mismatch")
    groups: dict[str, list[float]] = {}
    order: list[str] = []
    for pair_id, value in zip(pair_ids, values, strict=True):
        if pair_id not in groups:
            groups[pair_id] = []
            order.append(pair_id)
        groups[pair_id].append(float(value))
    return np.asarray([float(np.mean(groups[pair_id])) for pair_id in order], dtype=np.float64)


def _statistic(values: np.ndarray, statistic: str) -> float:
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "median":
        return float(np.median(values))
    if statistic == "positive_fraction":
        return float(np.mean(values > 0.0))
    raise ValueError(f"unknown statistic {statistic}")


def paired_bootstrap(values: Sequence[float] | np.ndarray, *, seed: int = BOOTSTRAP_SEED, replicates: int = BOOTSTRAP_REPLICATES, cluster: Sequence[str] | None = None) -> dict[str, Any]:
    """Return deterministic percentile CIs, resampling whole pair clusters."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(array.size > 0 and bool(np.isfinite(array).all()), "bootstrap values invalid")
    if cluster is None:
        clustered = array
    else:
        clustered = _cluster_values([str(value) for value in cluster], array)
    _require(clustered.size > 0, "bootstrap has no clusters")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, clustered.size, size=(int(replicates), clustered.size))
    samples = clustered[indices]
    output: dict[str, Any] = {}
    for statistic in ("mean", "median", "positive_fraction"):
        point = _statistic(clustered, statistic)
        estimates = np.mean(samples, axis=1) if statistic == "mean" else (
            np.median(samples, axis=1) if statistic == "median" else np.mean(samples > 0.0, axis=1)
        )
        output[statistic] = {"point": point, "lower_95": float(np.quantile(estimates, 0.025)), "upper_95": float(np.quantile(estimates, 0.975))}
    return {"cluster": "pair", "seed": int(seed), "replicates": int(replicates), "quantile_method": "numpy_linear", "statistics": output}


def paired_sign_flip(values: Sequence[float] | np.ndarray, *, seed: int = SIGN_FLIP_SEED, replicates: int = SIGN_FLIP_REPLICATES, cluster: Sequence[str] | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(array.size > 0 and bool(np.isfinite(array).all()), "sign-flip values invalid")
    clustered = array if cluster is None else _cluster_values([str(value) for value in cluster], array)
    observed = float(np.mean(clustered))
    rng = np.random.default_rng(int(seed))
    signs = rng.integers(0, 2, size=(int(replicates), clustered.size), dtype=np.int8) * 2 - 1
    null = np.mean(signs * clustered, axis=1)
    exceed = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    return {"cluster": "pair", "seed": int(seed), "replicates": int(replicates), "observed_mean": observed, "null_mean": 0.0, "two_sided_p": float((exceed + 1) / (int(replicates) + 1))}


def _delta_report(pair_ids: Sequence[str], values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(len(pair_ids) == len(array), "delta pair/value mismatch")
    return {
        "point": {"mean": _statistic(array, "mean"), "median": _statistic(array, "median"), "positive_fraction": _statistic(array, "positive_fraction")},
        "paired_bootstrap_95": paired_bootstrap(array, cluster=pair_ids),
        "paired_sign_flip": paired_sign_flip(array, cluster=pair_ids),
        "per_pair": [{"pair_id": str(pair_id), "delta": float(value)} for pair_id, value in zip(pair_ids, array, strict=True)],
    }


def _validate_formal(d1_train: Mapping[str, Any], d1_side: Mapping[str, Any], d1_result: Mapping[str, Any], d1_diag: Mapping[str, Any]) -> dict[str, Any]:
    _require(d1_train.get("schema_version") == 1 and d1_train.get("status") == "completed", "D1 training report incomplete")
    provenance = d1_train.get("provenance", {})
    result = d1_train.get("result", {})
    _require(provenance.get("seed") == 14321 and result.get("seed") == 14321, "D1 training seed changed")
    _require(provenance.get("optimizer_steps") == 8192 and result.get("optimizer_steps") == 8192, "D1 optimizer schedule changed")
    _require(d1_train.get("fixed_checkpoint_step") == 8192, "D1 checkpoint step changed")
    _require(d1_train.get("independent_validation_used_for_selection") is False and d1_train.get("loader_validation_used_for_selection") is False, "D1 selection boundary changed")
    data = provenance.get("data", {})
    _require(data.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256 and data.get("release_manifest_sha256") == EXPECTED_MANIFEST_SHA256, "D1 training data identity changed")
    _require(data.get("loader_validation_pairs") == PAIR_COUNT and data.get("train_pairs") == 8192 and data.get("independent_validation_opened") is False, "D1 Development boundary changed")
    release = provenance.get("release", {})
    _require(release.get("sha256") == EXPECTED_RELEASE_SHA256, "D1 release identity changed")
    upstream = provenance.get("upstream", {}).get("initial_checkpoint", {})
    _require(upstream.get("sha256") == EXPECTED_INITIAL_CHECKPOINT_SHA256, "D1 source checkpoint changed")
    method = provenance.get("method", {})
    _require(method.get("source_sha256") == EXPECTED_SOURCE_SHA256 and method.get("candidate") == "pusht_motion_damping_d1_energy_stratified_native_v1", "D1 source identity changed")
    _require(d1_side.get("candidate") == method["candidate"] and d1_side.get("source_sha256") == EXPECTED_SOURCE_SHA256, "D1 sidecar source identity changed")
    _require(d1_side.get("source_checkpoint_sha256") == EXPECTED_INITIAL_CHECKPOINT_SHA256, "D1 sidecar checkpoint identity changed")
    _require(d1_side.get("schedule_batches_consumed") == 8192 and d1_side.get("release_train_pairs") == 8192 and d1_side.get("fresh_optimizer_steps") == 8192, "D1 schedule was not fully consumed")
    _require(d1_side.get("single_change") == "hidden twin exposure schedule" and d1_side.get("model_or_loss_changed") is False, "D1 is not a data-only change")
    _require(d1_side.get("public_test_opened") is False and d1_side.get("score_or_arm_metadata_at_model_or_loss_boundary") is False, "D1 Public Test or metadata boundary changed")
    _require(d1_side.get("schedule_sha256") == EXPECTED_SCHEDULE_SHA256, "D1 schedule hash changed")
    _require(
        d1_side.get("training_report_sha256")
        == EXPECTED_INPUT_SHA256["d1_training_report"],
        "D1 sidecar training report reference changed",
    )
    contract = result.get("d1_ms50_schedule_contract", {})
    _require(contract.get("candidate_id") == "D1-MS50" and contract.get("optimizer_steps") == 8192 and contract.get("training_seed") == 14321, "D1 schedule contract changed")
    checks = contract.get("checks", {})
    _require(isinstance(checks, Mapping) and checks and all(value is True for value in checks.values()), "D1 schedule checks failed")
    _require(d1_result.get("optimizer_step") == 8192 and d1_result.get("role") == "native", "D1 endpoint identity changed")
    _require(d1_result.get("evidence_boundary", {}).get("public_test_opened") is False, "D1 endpoint opened Public Test")
    _require(d1_result.get("source_sha256") == EXPECTED_EVALUATOR_SOURCE_SHA256, "D1 evaluator source changed")
    _require(
        d1_result.get("training_report_sha256")
        == EXPECTED_INPUT_SHA256["d1_training_report"],
        "D1 endpoint training report reference changed",
    )
    _require(d1_result.get("checkpoint_sha256") == d1_train.get("result", {}).get("final_checkpoint", {}).get("sha256"), "D1 checkpoint identity mismatch")
    _check_diagnostic_model(d1_diag, "D1 diagnostic", expected_checkpoint=d1_result.get("checkpoint_sha256"))
    parity = d1_diag.get("reference_result_parity")
    _require(isinstance(parity, Mapping), "D1 diagnostic endpoint parity missing")
    _require(
        parity.get("sha256") == EXPECTED_INPUT_SHA256["d1_result"]
        and parity.get("all_available_within_2e_4") is True,
        "D1 diagnostic endpoint parity failed",
    )
    return {"optimizer_steps": 8192, "schedule_batches_consumed": 8192, "single_change": d1_side["single_change"], "public_test_opened": False, "schedule_sha256": dict(d1_side["schedule_sha256"])}


def analyze(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Analyze already-loaded frozen payloads; useful for synthetic tests."""
    d0_ids, d0_values = _record_arrays(payloads["d0_diagnostic"], "D0 diagnostic")
    coja_ids, coja_values = _record_arrays(payloads["coja_diagnostic"], "COJA diagnostic")
    d1_ids, d1_values = _record_arrays(payloads["d1_diagnostic"], "D1 diagnostic")
    _require(d0_ids == coja_ids, "COJA pair_id/order changed")
    _require(d0_ids == d1_ids, "D1 pair_id/order changed")
    summaries = {
        "D0": _summary(payloads["d0_result"], d0_values, "D0"),
        "COJA": _summary(payloads["coja_result"], coja_values, "COJA"),
        "D1": _summary(payloads["d1_result"], d1_values, "D1"),
    }
    randomization = {
        "D0": _randomization_summary(payloads["d0_diagnostic"], "D0"),
        "COJA": _randomization_summary(payloads["coja_diagnostic"], "COJA"),
        "D1": _randomization_summary(payloads["d1_diagnostic"], "D1"),
    }
    deltas = {}
    for metric in ("g_swap", "response_gain", "nre"):
        deltas[metric] = _delta_report(d0_ids, d1_values[metric] - d0_values[metric])
    assignments = {}
    for metric in summaries["D0"]["assignment"]:
        assignments[metric] = {"D1_minus_D0": summaries["D1"]["assignment"][metric] - summaries["D0"]["assignment"][metric], "D1": summaries["D1"]["assignment"][metric], "D0": summaries["D0"]["assignment"][metric], "COJA": summaries["COJA"]["assignment"][metric]}
    aggregate = {
        "response_gain": {name: summaries[name]["latent_response"]["response_gain"] for name in ("D0", "D1", "COJA")},
        "normalized_response_error": {name: summaries[name]["latent_response"]["normalized_response_error"] for name in ("D0", "D1", "COJA")},
        "assignment": assignments,
        "D1_minus_D0": {"response_gain": summaries["D1"]["latent_response"]["response_gain"] - summaries["D0"]["latent_response"]["response_gain"], "normalized_response_error": summaries["D1"]["latent_response"]["normalized_response_error"] - summaries["D0"]["latent_response"]["normalized_response_error"]},
    }
    d1_natural_positive = bool(
        summaries["D1"]["g_swap"]["mean"] > 0.0
        and randomization["D1"]["within_pair_sign_flip_two_sided_p"] < 0.05
        and randomization["D1"]["cross_query_one_sided_p"] < 0.05
        and summaries["D1"]["latent_response"]["response_gain"] >= GAIN_GATE
        and summaries["D1"]["assignment"]["correct_history_preference_rate"]
        >= HISTORY_GATE
        and summaries["D1"]["latent_response"]["normalized_response_error"] < 1.0
    )
    directional = bool(
        deltas["g_swap"]["paired_bootstrap_95"]["statistics"]["mean"]["lower_95"] > 0.0
        and deltas["response_gain"]["paired_bootstrap_95"]["statistics"]["mean"]["lower_95"] > 0.0
        and deltas["nre"]["paired_bootstrap_95"]["statistics"]["mean"]["upper_95"] <= 0.0
        and deltas["g_swap"]["paired_sign_flip"]["two_sided_p"] < 0.05
        and deltas["response_gain"]["paired_sign_flip"]["two_sided_p"] < 0.05
    )
    cross_query = {}
    for name, payload in (("D0", payloads["d0_diagnostic"]), ("COJA", payloads["coja_diagnostic"]), ("D1", payloads["d1_diagnostic"])):
        baseline = payload.get("random_pairing_baselines", {}).get("cross_query_pairing")
        cross_query[name] = dict(baseline) if isinstance(baseline, Mapping) else {"status": "not_available"}
    return {
        "summaries": summaries,
        "randomization": randomization,
        "deltas": deltas,
        "aggregate_point_estimate_changes": aggregate,
        "cross_query_null": cross_query,
        "decision": {
            "single_seed_directional_data_effect": directional,
            "history_use_positive": d1_natural_positive,
            "history_use_gates": {
                "within_pair_sign_flip_p_lt_0p05": (
                    randomization["D1"]["within_pair_sign_flip_two_sided_p"]
                    < 0.05
                ),
                "cross_query_p_lt_0p05": (
                    randomization["D1"]["cross_query_one_sided_p"] < 0.05
                ),
                "aggregate_gain_ge_0p1": (
                    summaries["D1"]["latent_response"]["response_gain"]
                    >= GAIN_GATE
                ),
                "history_rate_ge_0p55": (
                    summaries["D1"]["assignment"][
                        "correct_history_preference_rate"
                    ]
                    >= HISTORY_GATE
                ),
                "aggregate_nre_lt_1": (
                    summaries["D1"]["latent_response"][
                        "normalized_response_error"
                    ]
                    < 1.0
                ),
            },
            "removed_history": "not_measured",
            "training_seed_count": 1,
            "public_test_authorized": False,
        },
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    loaded = {name: _load_frozen(name) for name in INPUT_PATHS}
    _check_diagnostic_model(loaded["d0_diagnostic"], "D0 diagnostic", expected_checkpoint=loaded["d0_result"].get("checkpoint_sha256"))
    _check_diagnostic_model(loaded["coja_diagnostic"], "COJA diagnostic", expected_checkpoint=loaded["coja_result"].get("checkpoint_sha256"))
    _validate_formal(loaded["d1_training_report"], loaded["d1_sidecar"], loaded["d1_result"], loaded["d1_diagnostic"])
    report = {"schema_version": 1, "analysis_id": "pusht_motion_damping_d1_native_endpoint_comparison_v1", "status": "completed_development_only_read_only", "evidence_boundary": {"development_only": True, "public_test_opened": False, "optimizer_steps_added": 0, "checkpoint_loaded": False, "cem_run": False, "data_files_opened": False}, "inputs": {name: {"path": str(INPUT_PATHS[name].resolve()), "sha256": EXPECTED_INPUT_SHA256[name]} for name in INPUT_PATHS}, **analyze(loaded)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.output)
    print(json.dumps({"status": report["status"], "output": str(Path(args.output).expanduser().resolve())}, indent=2))


if __name__ == "__main__":
    main()
