from __future__ import annotations

import numpy as np

from research.conditional_dynamics_representation.scripts import (
    qualify_pusht_motion_damping_label_blind_overlap_collection_v1 as subject,
)


def test_black_box_shooting_recovers_query_without_dynamics_label() -> None:
    target = np.zeros(12, dtype=np.float64)
    target[6:8] = (13.0, -4.0)
    target[8:10] = (2.0, -1.0)
    velocity_scale = 0.37
    displacement_coefficient = 0.83

    def query_from_reset(reset: np.ndarray) -> np.ndarray:
        result = reset.copy()
        result[6:8] = (
            reset[6:8] + displacement_coefficient * reset[8:10]
        )
        result[8:10] = velocity_scale * reset[8:10]
        return result

    solved, receipt = subject.solve_reset_from_query_feedback(
        target,
        query_from_reset,
    )

    np.testing.assert_allclose(query_from_reset(solved), target, atol=1e-12)
    assert np.isclose(receipt["observed_velocity_scale"], velocity_scale)
    assert np.isclose(
        receipt["observed_displacement_coefficient"],
        displacement_coefficient,
    )


def test_row_identity_is_order_invariant() -> None:
    pixels = np.arange(4 * 3 * 2 * 2, dtype=np.uint8).reshape(4, 3, 2, 2)
    actions = np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2)
    forward = subject._row_sha256(pixels, actions)
    backward = subject._row_sha256(pixels.copy(), actions.copy())
    assert forward == backward
