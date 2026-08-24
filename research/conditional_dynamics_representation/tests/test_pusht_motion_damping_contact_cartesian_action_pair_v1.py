from __future__ import annotations

from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_contact_cartesian_action_pair_v1 as subject,
)


def test_contact_candidate_freezes_only_support_change():
    assert subject.OVERLAY_TEMPLATE_COUNT == 2048
    assert subject.OVERLAY_CONDITION_PAIR_COUNT == 4096
    assert subject.ACTION_BRANCHES == [
        "observed_zero",
        "toward_block_0p45",
    ]
    assert len(subject.OVERLAY_SHA256) == 64
    assert len(subject.OVERLAY_RECEIPT_SHA256) == 64


def test_main_installs_contact_identity(monkeypatch):
    monkeypatch.setattr(
        subject,
        "_overlay_receipt",
        lambda _argv: (
            subject.Path("overlay.pt.json"),
            {"unused": True},
        ),
    )
    monkeypatch.setattr(subject.base, "main", lambda _argv: 17)
    assert subject.main(["--cartesian-overlay", "overlay.pt"]) == 17
    assert subject.base.CANDIDATE == subject.CANDIDATE
    assert subject.base.OVERLAY_SHA256 == subject.OVERLAY_SHA256
    assert subject.base.OVERLAY_TEMPLATE_COUNT == 2048
