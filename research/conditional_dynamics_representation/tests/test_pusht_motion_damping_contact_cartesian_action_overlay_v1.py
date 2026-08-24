from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.conditional_dynamics_representation.scripts import (
    build_pusht_motion_damping_contact_cartesian_action_overlay_v1 as subject,
)


def _template(agent=(10.0, 20.0), block=(13.0, 24.0)):
    snapshot = np.zeros(12, dtype=np.float64)
    snapshot[0:2] = agent
    snapshot[6:8] = block
    return SimpleNamespace(expected_natural_query_snapshot=tuple(snapshot))


def test_contact_action_points_toward_block_at_frozen_scale():
    actions = subject.contact_query_actions(_template())
    expected = np.asarray([0.27, 0.36])
    assert actions.shape == (5, 2)
    assert np.allclose(actions, np.repeat(expected[None], 5, axis=0))
    assert np.allclose(np.linalg.norm(actions, axis=1), 0.45)


def test_contact_action_rejects_coincident_query_bodies():
    with pytest.raises(ValueError, match="must differ"):
        subject.contact_query_actions(
            _template(agent=(10.0, 20.0), block=(10.0, 20.0))
        )


def test_playfield_bounds_are_strict():
    inside = {"agent": [5.0, 5.0, 20.0, 20.0], "block": [6, 6, 506, 506]}
    outside = {"agent": [4.9, 5.0, 20.0, 20.0], "block": [6, 6, 30, 30]}
    assert subject._bounds_inside_playfield(inside)
    assert not subject._bounds_inside_playfield(outside)
