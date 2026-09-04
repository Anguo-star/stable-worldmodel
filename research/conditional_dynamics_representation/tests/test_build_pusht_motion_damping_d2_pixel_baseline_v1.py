from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_pusht_motion_damping_d2_pixel_baseline_v1.py"
SPEC = importlib.util.spec_from_file_location("motion_d2_pixel_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def synthetic_rows(cells: int = 4, twins_per_cell: int = 6) -> list[dict]:
    rows = []
    twin = 0
    for cell in range(cells):
        for offset in range(twins_per_cell):
            rows.append(
                {
                    "coverage_cell": cell,
                    "twin_id": twin,
                    "pair_ids": [
                        f"pmd-train-{2 * twin:06d}-forward",
                        f"pmd-train-{2 * twin + 1:06d}-reverse",
                    ],
                    "realized_pi_ms50": float(offset + 1),
                }
            )
            twin += 1
    return rows


def test_panel_selection_is_exact_stratified_and_deterministic() -> None:
    rows = synthetic_rows()
    first = module.select_panel(
        rows, seed=17, twins_per_cell=4, expected_cells=4, expected_twins=None
    )
    second = module.select_panel(
        list(reversed(rows)), seed=17, twins_per_cell=4, expected_cells=4,
        expected_twins=None,
    )
    assert first == second
    assert len(first) == 16
    assert np.bincount([row["coverage_cell"] for row in first]).tolist() == [4] * 4
    assert module.panel_hash(first) == module.panel_hash(second)


def test_equal_cell_weights_and_d1_within_cell_mass() -> None:
    indices, weights = module.equal_cell_weights(
        [0, 0, 1, 1], [1.0, 3.0, 2.0, 2.0], [[0, 1], [2, 3], [4, 5], [6, 7]]
    )
    assert indices.tolist() == list(range(8))
    np.testing.assert_allclose(weights[:2], [1 / 16, 1 / 16])
    np.testing.assert_allclose(weights[2:4], [3 / 16, 3 / 16])
    np.testing.assert_allclose(weights[4:], [1 / 8] * 4)
    assert np.isclose(weights.sum(), 1.0)


def test_neighbor_graph_excludes_the_whole_twin() -> None:
    descriptors = np.arange(16, dtype=float).reshape(8, 2)
    twin_ids = np.repeat(np.arange(4), 2)
    neighbors = module.build_neighbor_graph(descriptors, twin_ids, max_k=3)
    for query, selected in enumerate(neighbors):
        assert np.all(twin_ids[selected] != twin_ids[query])


def _bootstrap_inputs():
    panel = []
    mapping = {}
    query = 0
    for cell in range(2):
        for offset in range(4):
            twin = cell * 4 + offset
            panel.append({"coverage_cell": cell, "twin_id": twin,
                          "realized_pi_ms50": float(1 + offset)})
            mapping[twin] = [query, query + 1]
            query += 2
    x = np.arange(query, dtype=float)
    C = 0.2 + 0.01 * x + 0.003 * np.square(x % 3)
    backgrounds = {2: 1.0 + 0.02 * x, 3: 1.3 + 0.01 * x,
                   4: 0.8 + 0.03 * x}
    return panel, mapping, C, backgrounds


def test_simultaneous_bootstrap_is_deterministic() -> None:
    args = _bootstrap_inputs()
    first = module.stratified_cluster_bootstrap_differences(
        *args, ks=(2, 3, 4), n_resamples=128, seed=99
    )
    second = module.stratified_cluster_bootstrap_differences(
        *args, ks=(2, 3, 4), n_resamples=128, seed=99
    )
    assert first == second
    assert first["max_statistic_critical_value"] > 0
    for result in first["per_k"].values():
        assert result["simultaneous_ci_lo"] <= result["rho_difference_d1_minus_d0"]
        assert result["simultaneous_ci_hi"] >= result["rho_difference_d1_minus_d0"]


def test_hash_and_forbidden_split_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pytest's own temporary directory contains a component beginning with
    # ``test_``; isolate hash behavior from the independently checked guard.
    frozen_guard = module.training_only_guard
    monkeypatch.setattr(module, "training_only_guard", lambda _path: None)
    target = tmp_path / "training" / "safe.txt"
    target.parent.mkdir()
    target.write_text("frozen", encoding="utf-8")
    digest = hashlib.sha256(b"frozen").hexdigest()
    assert module.verify_input_hashes(
        tmp_path, {"safe": ("training/safe.txt", digest)}
    )["safe"] == digest
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        module.verify_input_hashes(
            tmp_path, {"safe": ("training/safe.txt", "0" * 64)}
        )
    with pytest.raises(ValueError, match="forbidden token"):
        frozen_guard(Path("/data/development/x"))


def test_evidence_boundary_and_no_overwrite(tmp_path: Path) -> None:
    receipt = module.evidence_receipt()
    assert receipt["model_loaded"] is False
    assert receipt["optimizer_steps"] == 0
    assert receipt["development_split_opened"] is False
    output = tmp_path / "result.json"
    module.write_exclusive_json(output, {"ok": True})
    with pytest.raises(FileExistsError):
        module.write_exclusive_json(output, {"ok": False})
