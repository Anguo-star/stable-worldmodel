from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RESEARCH_ROOT / (
    "scripts/build_pusht_motion_damping_root_cause_comparator_schedules_v1.py"
)
SPEC = importlib.util.spec_from_file_location("motion_comparator_schedules", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_comparator_pins_and_builder_pin_match() -> None:
    frozen = module.verify_frozen_inputs()
    assert frozen["observed_sha256"] == module.EXPECTED_COMPARATOR_SHA256
    assert frozen["comparator_dir"].name == "comparators_v1_final"
    assert frozen["comparator_builder_sha256"] == "6ec83fa8f2210680dc2a2488f8dd772661738093f328bc5333de18055e28fcb8"
    assert frozen["config"]["builder_script_sha256"] == frozen["comparator_builder_sha256"]
    assert frozen["receipt"]["builder_script_sha256"] == frozen["comparator_builder_sha256"]


def test_training_only_path_guard_rejects_all_held_out_tokens(tmp_path: Path) -> None:
    for part in ("development", "public", "test", "validation", "public_test", "loader_validation"):
        with pytest.raises(RuntimeError, match="forbidden"):
            module.assert_training_only_path(tmp_path / part / "projected_weights.jsonl")


def test_largest_remainder_is_cell_local_and_twin_id_stable() -> None:
    probabilities = np.full(8, 0.125)
    cells = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    twin_ids = np.arange(8)
    counts, desired = module.largest_remainder_counts(
        probabilities, cells, twin_ids, slots_per_cell=6
    )
    assert counts.tolist() == [2, 2, 1, 1, 2, 2, 1, 1]
    assert np.allclose(desired, 1.5)
    shuffled = np.asarray([3, 2, 1, 0])
    tied, _ = module.largest_remainder_counts(
        np.full(4, 0.25), np.zeros(4, dtype=int), shuffled, slots_per_cell=6
    )
    assert tied.tolist() == [1, 1, 2, 2]


def test_natural_and_high_incidence_are_deterministic_and_balanced() -> None:
    cells = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    twin_ids = np.arange(8)
    natural = module.natural_cycle_incidence(cells, twin_ids, cycles=4, seed=11)
    assert np.array_equal(natural, module.natural_cycle_incidence(cells, twin_ids, cycles=4, seed=11))
    assert np.all(natural.sum(axis=1) == 2)
    for cell in (0, 1):
        local = natural[cells == cell]
        assert np.all(local.sum(axis=0) == 2)
        assert np.all(local[:, 0].astype(int) + local[:, 1].astype(int) == 1)
        assert np.all(local[:, 2].astype(int) + local[:, 3].astype(int) == 1)

    counts = np.asarray([5, 4, 3, 2, 1])
    one_cell = np.zeros(5, dtype=int)
    high = module.high_cycle_incidence(counts, one_cell, np.arange(5), cycles=5, seed=17)
    assert np.array_equal(high.sum(axis=1), counts)
    assert np.all(high.sum(axis=0) == 3)
    with pytest.raises(RuntimeError, match="exceeds cycle support"):
        module.high_cycle_incidence(np.asarray([6, 4]), np.zeros(2, dtype=int), np.arange(2), cycles=5)


@pytest.fixture(scope="module")
def constructed() -> tuple[dict[str, object], object, dict[str, list[dict[str, object]]], dict[str, dict[str, object]], dict[str, object]]:
    return module.construct()


def test_real_pool_has_four_positive_projected_arms() -> None:
    frozen = module.verify_frozen_inputs()
    pool = module.load_comparator_pool(frozen["paths"]["projected_weights.jsonl"])
    assert np.array_equal(pool.twin_ids, np.arange(module.EXPECTED_TWINS))
    assert set(pool.high_weights) == set(module.ARMS)
    assert set(pool.full_weights) == set(module.ARMS)
    for arm in module.ARMS:
        assert np.all(pool.high_weights[arm] > 0.0)
        assert np.all(pool.full_weights[arm] > 0.0)
        assert np.isclose(pool.full_weights[arm].sum(), 1.0)


def test_constructed_schedules_share_layout_and_keep_each_pair_group_together(constructed: tuple[object, ...]) -> None:
    _, pool, schedules, audits, summary = constructed
    assert set(schedules) == set(module.ARMS)
    assert all(len(schedules[arm]) == 8192 for arm in module.ARMS)
    assert summary["gates"]["passed"] is True
    assert summary["shared_layout_audit"]["checks"]["shared_batch_coverage_cell_signature_exact"] is True
    assert summary["shared_layout_audit"]["checks"]["shared_comparator_rank_weight_signature_exact"] is True
    assert summary["shared_layout_audit"]["checks"]["shared_natural_batch_signature_exact"] is True
    for index in (0, 255, 256, 8191):
        rows = [schedules[arm][index] for arm in module.ARMS]
        assert all(len(row["twin_ids"]) == len(set(row["twin_ids"])) == 16 for row in rows)
        assert all(row["hidden_row_indices"] == module.expand_hidden_rows(row["twin_ids"]) for row in rows)
        assert all(
            pair == list(pool.pair_ids[int(twin)])
            for row in rows
            for twin, pair in zip(row["twin_ids"], row["pair_ids"], strict=True)
        )
        assert rows[1]["high_slot_ids"] == rows[2]["high_slot_ids"] == rows[3]["high_slot_ids"]
    for arm in module.ARMS:
        assert audits[arm]["checks"]["high_multiplicity_matches_largest_remainder"] is True
        assert audits[arm]["checks"]["full_multiplicity_matches_largest_remainder"] is True


def test_check_only_does_not_create_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "check_only_output"
    assert module.main(["--check-only", "--output-dir", str(output)]) == 0
    assert not output.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed_go"
    assert report["gates"]["passed"] is True


def test_write_is_exclusive_and_receipt_rehashes_every_output(
    constructed: tuple[object, ...], tmp_path: Path
) -> None:
    frozen, pool, schedules, audits, summary = constructed
    output = tmp_path / "comparator_schedules"
    paths = module.write_artifacts(
        output,
        frozen=frozen,
        pool=pool,
        schedules=schedules,
        audits=audits,
        summary=summary,
    )
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert set(receipt["output_sha256"]) == {
        "config.json",
        "summary.json",
        "schedule_D0.jsonl",
        "schedule_REL50.jsonl",
        "schedule_ABS50.jsonl",
        "schedule_HASH50.jsonl",
        "multiplicity_D0.jsonl",
        "multiplicity_REL50.jsonl",
        "multiplicity_ABS50.jsonl",
        "multiplicity_HASH50.jsonl",
    }
    for name, expected in receipt["output_sha256"].items():
        assert _sha(output / name) == expected
    assert receipt["comparator_builder_script_sha256"] == "6ec83fa8f2210680dc2a2488f8dd772661738093f328bc5333de18055e28fcb8"
    assert receipt["schedule_builder_script_sha256"] == _sha(SCRIPT)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        module.write_artifacts(
            output,
            frozen=frozen,
            pool=pool,
            schedules=schedules,
            audits=audits,
            summary=summary,
        )


def test_script_stays_index_only_and_training_bounded() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "optimizer.step" not in source
    assert "lance.dataset" not in source
    assert '"development_lance_opened": False' in source
    assert '"public_test_lance_opened": False' in source
    assert '"validation_lance_opened": False' in source
    assert "DEFAULT_COMPARATOR_DIR = RESEARCH_ROOT /" in source
    assert "comparators_v1_final" in source
