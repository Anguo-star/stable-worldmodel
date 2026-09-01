from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RESEARCH_ROOT / "scripts/build_pusht_motion_damping_d1_schedule_v1.py"
SPEC = importlib.util.spec_from_file_location("d1_schedule_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
d1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d1
SPEC.loader.exec_module(d1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_v2_artifact_hashes_match_constants() -> None:
    for name, expected in d1.EXPECTED_V2_SHA256.items():
        assert _sha(d1.DEFAULT_V2_DIR / name) == expected


def test_largest_remainder_uses_cell_totals_and_twin_id_ties() -> None:
    probabilities = np.full(8, 0.125)
    cells = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    twin_ids = np.arange(8)
    counts, desired = d1.largest_remainder_counts(
        probabilities, cells, twin_ids, slots_per_cell=6
    )
    assert counts.tolist() == [2, 2, 1, 1, 2, 2, 1, 1]
    assert np.allclose(desired, 1.5)
    assert [int(counts[cells == cell].sum()) for cell in (0, 1)] == [6, 6]

    shuffled_ids = np.asarray([3, 2, 1, 0])
    tied, _ = d1.largest_remainder_counts(
        np.full(4, 0.25), np.zeros(4, dtype=int), shuffled_ids, slots_per_cell=6
    )
    assert tied.tolist() == [1, 1, 2, 2]


def test_natural_cycles_are_complementary_and_exact() -> None:
    cells = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    twin_ids = np.arange(8)
    incidence = d1.natural_cycle_incidence(cells, twin_ids, cycles=4, seed=11)
    assert incidence.shape == (8, 4)
    assert np.all(incidence.sum(axis=1) == 2)
    for cell in (0, 1):
        local = incidence[cells == cell]
        assert np.all(local.sum(axis=0) == 2)
        assert np.all(local[:, 0].astype(int) + local[:, 1].astype(int) == 1)
        assert np.all(local[:, 2].astype(int) + local[:, 3].astype(int) == 1)
    np.testing.assert_array_equal(
        incidence,
        d1.natural_cycle_incidence(cells, twin_ids, cycles=4, seed=11),
    )


def test_high_cycle_allocation_preserves_row_and_column_sums() -> None:
    counts = np.asarray([5, 4, 3, 2, 1])
    cells = np.zeros(5, dtype=int)
    twin_ids = np.arange(5)
    incidence = d1.high_cycle_incidence(counts, cells, twin_ids, cycles=5, seed=17)
    np.testing.assert_array_equal(incidence.sum(axis=1), counts)
    assert np.all(incidence.sum(axis=0) == 3)
    assert np.all(incidence <= 1)
    np.testing.assert_array_equal(
        incidence,
        d1.high_cycle_incidence(counts, cells, twin_ids, cycles=5, seed=17),
    )

    with pytest.raises(RuntimeError, match="exceeds cycle support"):
        d1.high_cycle_incidence(
            np.asarray([6, 4]), np.zeros(2, dtype=int), np.arange(2), cycles=5
        )


def test_disjoint_block_matching_is_deterministic_and_rejects_impossible_case() -> None:
    high = [(0, 1), (2, 3), (4, 5), (6, 7)]
    natural = [(0, 1), (2, 3), (4, 5), (6, 7)]
    first = d1.match_disjoint_blocks(high, natural, seed=23, cycle=0)
    second = d1.match_disjoint_blocks(high, natural, seed=23, cycle=0)
    assert first == second
    assert sorted(first) == list(range(4))
    for high_index, natural_index in enumerate(first):
        assert set(high[high_index]).isdisjoint(natural[natural_index])

    with pytest.raises(RuntimeError, match="no conflict-free"):
        d1.match_disjoint_blocks([(0,)], [(0,)], seed=1, cycle=0)


def test_hidden_row_expansion_preserves_complete_twin_order() -> None:
    assert d1.expand_hidden_rows([2, 0, 3]) == [
        8,
        9,
        10,
        11,
        0,
        1,
        2,
        3,
        12,
        13,
        14,
        15,
    ]


def test_total_variation_requires_normalized_inputs() -> None:
    assert d1.total_variation(
        np.asarray([0.75, 0.25]), np.asarray([0.5, 0.5])
    ) == pytest.approx(0.25)
    with pytest.raises(RuntimeError, match="not normalized"):
        d1.total_variation(np.asarray([0.6, 0.3]), np.asarray([0.5, 0.5]))


def test_real_projected_pool_has_the_frozen_soft_structure() -> None:
    pool = d1.load_projected_pool(d1.DEFAULT_V2_DIR / "projected_weights.jsonl")
    assert len(pool.twin_ids) == 4096
    assert np.array_equal(pool.twin_ids, np.arange(4096))
    assert set(pool.cells.tolist()) == set(range(64))
    assert pool.pair_ids[0] == (
        "pmd-train-000000-forward",
        "pmd-train-000001-reverse",
    )
    assert pool.pair_ids[-1] == (
        "pmd-train-008190-forward",
        "pmd-train-008191-reverse",
    )
    assert np.all(pool.pi_high > 0.0)
    assert np.allclose(pool.pi_full, 0.5 / 4096 + 0.5 * pool.pi_high)
    for cell in range(64):
        assert pool.pi_high[pool.cells == cell].sum() == pytest.approx(1 / 64)


def test_corrupted_v2_artifact_fails_before_schedule_construction(tmp_path: Path) -> None:
    copied = tmp_path / "v2"
    shutil.copytree(d1.DEFAULT_V2_DIR, copied)
    config = copied / "config.json"
    config.write_bytes(config.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="artifact SHA256 changed"):
        d1.verify_frozen_inputs(copied)


@pytest.fixture(scope="module")
def built_twice(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("d1_schedule")
    first = root / "first"
    second = root / "second"
    assert d1.main(["--output-dir", str(first)]) == 0
    assert d1.main(["--output-dir", str(second)]) == 0
    return first, second


def test_end_to_end_core_outputs_are_byte_identical(
    built_twice: tuple[Path, Path],
) -> None:
    first, second = built_twice
    for name in ("summary.json", "multiplicity.jsonl", "schedule.jsonl"):
        assert _sha(first / name) == _sha(second / name)
    summary = json.loads((first / "summary.json").read_text())
    assert summary["status"] == "passed_go"
    assert summary["gates"]["passed"] is True
    assert all(summary["gates"]["checks"].values())
    assert summary["dimensions"]["optimizer_steps"] == 8192
    assert summary["batch_audit"]["cross_arm_conflict_count"] == 0
    realized = summary["realized_distribution"]
    assert realized["high_tv_vs_projection"] == pytest.approx(
        0.00801712427383814, abs=1e-15
    )
    assert realized["full_tv_vs_projection"] == pytest.approx(
        0.0040085621369190675, abs=1e-15
    )
    assert realized["weighted_conditional_energy"] == pytest.approx(
        2.168315876876289, abs=1e-14
    )
    expected_rho = {
        "32": 0.14251716870785303,
        "64": 0.1083042925528566,
        "128": 0.08036213501656038,
    }
    for k, expected in expected_rho.items():
        assert realized["by_k"][k]["realized_rho_phys"] == pytest.approx(
            expected, abs=1e-15
        )


def test_end_to_end_schedule_and_multiplicity_invariants(
    built_twice: tuple[Path, Path],
) -> None:
    first, _ = built_twice
    schedule = [json.loads(line) for line in (first / "schedule.jsonl").read_text().splitlines()]
    assert len(schedule) == 8192
    for index, row in enumerate(schedule):
        assert row["schedule_index"] == index
        assert row["optimizer_step"] == index + 1
        assert row["cycle"] == index // 256
        assert row["batch_in_cycle"] == index % 256
        assert len(row["high_twin_ids"]) == len(row["natural_twin_ids"]) == 8
        assert len(row["twin_ids"]) == len(set(row["twin_ids"])) == 16
        assert set(row["high_twin_ids"]).isdisjoint(row["natural_twin_ids"])
        assert row["hidden_row_indices"] == d1.expand_hidden_rows(row["twin_ids"])

    multiplicity = [
        json.loads(line) for line in (first / "multiplicity.jsonl").read_text().splitlines()
    ]
    assert len(multiplicity) == 4096
    assert [row["twin_id"] for row in multiplicity] == list(range(4096))
    assert {row["realized_natural_count"] for row in multiplicity} == {16}
    high_counts = np.asarray([row["realized_high_count"] for row in multiplicity])
    assert int(high_counts.sum()) == 65536
    assert (int(high_counts.min()), int(high_counts.max())) == (0, 32)
    assert set(high_counts.tolist()) == set(range(33))


def test_receipt_rehashes_every_core_output(
    built_twice: tuple[Path, Path],
) -> None:
    first, _ = built_twice
    receipt = json.loads((first / "receipt.json").read_text())
    assert receipt["status"] == "passed_go"
    assert receipt["schedule_generated"] is True
    assert receipt["optimizer_steps_run"] == 0
    assert receipt["development_lance_opened"] is False
    assert receipt["public_test_lance_opened"] is False
    for name, expected in receipt["output_sha256"].items():
        assert _sha(first / name) == expected


def test_builder_refuses_to_overwrite_completed_output(
    built_twice: tuple[Path, Path],
) -> None:
    first, _ = built_twice
    with pytest.raises(FileExistsError):
        d1.main(["--output-dir", str(first)])


def test_source_is_cpu_only_and_names_forbidden_tables_only_in_guard() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "optimizer.step" not in source
    assert "lance.dataset" not in source
    assert "pixels" not in source.split("def assert_training_only_path", 1)[0]
    for table in ("validation.lance", "loader_validation.lance", "test.lance"):
        assert source.count(f'"{table}"') == 1
    assert '"development_lance_opened": False' in source
    assert '"public_test_lance_opened": False' in source
