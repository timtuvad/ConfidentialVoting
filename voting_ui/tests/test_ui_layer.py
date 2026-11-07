"""Актуальные UI-регрессии для OHE и трёх malicious-сценариев.

Заменяет старые тесты load_demo/одномерного sample_trace.json.
"""
from __future__ import annotations
from ui_layer.source import load_backend
from ui_layer.contracts import parse_trace, trace_to_json
from ui_layer.projections import project
from ui_layer.graph import build_dot

OPTIONS = ('A', 'B', 'C', 'D')
CHOICES = (0, 1, 2, 1, 3)


def test_live_ohe_13_steps_roundtrip():
    trace = load_backend(5, 3, 101, 17, CHOICES, OPTIONS)
    assert len(trace.snapshots) == 13
    assert set(trace.snapshots[-1].reconstructed.values()) == {(1, 2, 1, 1)}
    assert parse_trace(trace_to_json(trace)) == trace


def test_god_and_participant_views_are_isolated():
    trace = load_backend(5, 3, 101, 17, CHOICES, OPTIONS)
    for snapshot in trace.snapshots:
        frame = project(trace, snapshot, 'participant', (1,))
        for card in frame.cards:
            if card.idx != 1:
                assert card.vote is None
                assert card.polys is None
                assert not card.received
        assert all(d.receiver == -1 or d.sender == 1 or d.receiver == 1
                   for d in frame.visible_deals)
        dot = build_dot(frame)
        assert isinstance(dot, str) and dot


def test_membership_history_available_without_leaking_votes():
    trace = load_backend(5, 4, 101, 17, CHOICES, OPTIONS,
                         scenario='wrong_private_share', attacker=2,
                         enable_consensus=True, detector_mode='demo_oracle')
    last = trace.snapshots[-1]
    view = project(trace, last, 'participant', (1,))
    assert last.session_phase == 'finalized'
    assert any(kind == 'excluded' and pid == 2
               for kind, pid, _ in last.membership_events)
    assert view.excluded_participants == (2,)
    assert view.cards[1].vote is None
    assert view.attack_info is None
