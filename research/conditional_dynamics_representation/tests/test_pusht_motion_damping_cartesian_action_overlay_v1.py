import numpy as np

from contextworld.evaluation.pusht_motion_damping_h3 import (
    make_base_template,
)
from research.conditional_dynamics_representation.scripts import (
    build_pusht_motion_damping_cartesian_action_overlay_v1 as builder,
)


def test_alternate_query_action_is_valid_nonzero_unit_block() -> None:
    action = builder.alternate_query_actions(make_base_template())
    assert action.shape == (5, 2)
    assert np.array_equal(action, np.repeat(action[:1], 5, axis=0))
    assert np.isclose(np.linalg.norm(action[0]), 1.0)
    assert np.max(np.abs(action)) <= 1.0
    assert np.count_nonzero(action) > 0
