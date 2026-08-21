from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
CONTEXTWORLD = ROOT.parent / "ContextWorld"
for value in (CONTEXTWORLD, CONTEXTWORLD / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_pusht_hidden_actuation_pilot as pilot  # noqa: E402
import run_pusht_motion_damping_h3_train as motion  # noqa: E402


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pusht_motion_damping_native_twin_sampler_v1.py"
)


def _load():
    name = "_test_pusht_motion_damping_native_twin_sampler_v1"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _complete_twin_count(indices: torch.Tensor) -> int:
    pairs = {int(value) for value in (indices.reshape(-1, 2)[:, 0] // 2)}
    return sum(
        value % 2 == 0 and value + 1 in pairs for value in pairs
    )


def test_twin_sampler_is_the_only_change_to_hidden_batch_membership():
    module = _load()
    pair_count = 8192
    hidden_batch_size = 64
    batches = pair_count // (hidden_batch_size // 2)

    regular = iter(
        pilot.PairedBatchStream(
            pair_count,
            batch_size=hidden_batch_size,
            seed=14321,
        )
    )
    twin = iter(
        motion.CompleteTwinPairedBatchStream(
            pair_count,
            batch_size=hidden_batch_size,
            seed=14321,
        )
    )
    regular_counts = []
    twin_counts = []
    twin_rows = []
    for _ in range(batches):
        regular_counts.append(_complete_twin_count(next(regular)))
        selected = next(twin)
        twin_counts.append(_complete_twin_count(selected))
        twin_rows.append(selected)

    assert regular_counts[0] == 0
    assert sum(regular_counts) == 9
    assert regular_counts.count(0) == 247
    assert set(twin_counts) == {module.EXPECTED_TWIN_GROUPS_PER_BATCH}
    observed = torch.cat(twin_rows)
    expected = torch.arange(2 * pair_count)
    torch.testing.assert_close(torch.sort(observed).values, expected)
