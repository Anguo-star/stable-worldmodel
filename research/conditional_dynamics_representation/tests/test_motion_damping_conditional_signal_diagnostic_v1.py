from types import SimpleNamespace

import numpy as np

from research.conditional_dynamics_representation.scripts import (
    run_motion_damping_conditional_signal_diagnostic_v1 as diagnostic,
)


def test_standardized_state_targets_drop_constant_dimensions() -> None:
    faster = np.zeros((3, 4, 3), dtype=np.float32)
    slower = faster.copy()
    faster[:, 3, 0] = np.asarray([0.0, 1.0, 2.0])
    slower[:, 3, 0] = np.asarray([1.0, 2.0, 3.0])
    faster[:, 3, 1] = np.asarray([2.0, 3.0, 4.0])
    slower[:, 3, 1] = np.asarray([4.0, 5.0, 6.0])
    arrays = SimpleNamespace(
        faster_decay_states=faster,
        no_extra_decay_states=slower,
    )

    targets, protocol = diagnostic._standardized_state_targets(arrays)

    assert targets.shape == (3, 2, 2)
    assert protocol["raw_dimensions"] == 3
    assert protocol["active_standardized_dimensions"] == 2
    assert protocol["inactive_dimension_indices"] == [2]
