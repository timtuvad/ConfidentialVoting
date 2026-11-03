"""Регрессии внешнего UI: данные фильтруются ДО визуализации."""
from pathlib import Path
import sys
import itertools
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_layer.source import load_demo
from ui_layer.contracts import trace_from_json
from ui_layer.projections import project
from ui_layer.graph import build_dot
from ui_layer.observer import interpolate_coefficients, reconstruct_public, alternative_curves, horner


def test_demo_13_steps_and_sum():
    trace = load_demo()
    assert (trace.n, trace.t, trace.p) == (5, 3, 101)
    assert len(trace.snapshots) == 13
    assert [s.step for s in trace.snapshots] == list(range(1, 14))
    assert [s.phase for s in trace.snapshots] == (
        ['setup'] + ['deal'] * 5 + ['sum'] + ['publish'] * 5 + ['reconstruct'])
    assert trace.snapshots[-1].aggregate_F[0] == 3
    assert len(trace.snapshots[-1].reconstructed) == 10
    assert set(trace.snapshots[-1].reconstructed.values()) == {3}


def test_board_receives_only_published_deals():
    trace = load_demo()
    for snap in trace.snapshots:
        frame = project(trace, snap, 'god')
        assert len(frame.board_deals) == max(0, min(5, snap.step - 7))
        assert all(d.receiver == -1 for d in frame.board_deals)


def test_participant_cannot_see_others_secrets():
    trace = load_demo()
    for snap in trace.snapshots:
        for k in range(1, 6):
            frame = project(trace, snap, 'participant', (k,))
            for card in frame.cards:
                if card.idx != k:
                    assert card.vote is None
                    assert card.poly is None
                    assert card.received == ()
                    if card.idx not in snap.published_senders:
                        assert card.share is None
            for d in frame.visible_deals:
                assert d.receiver == -1 or d.sender == k or d.receiver == k
            # Невидимые рёбра не могут содержать буквальные значения.
            dot = build_dot(frame)
            for d in snap.all_deals:
                if d.receiver > 0 and d.sender != k and d.receiver != k:
                    assert f'f_{d.sender}({d.receiver})={d.value}' not in dot
            assert all(v == 'hidden' for i, v in frame.checks.items() if i != 'I5')


def test_coalition_sees_only_incident_private_edges():
    trace = load_demo()
    for snap in trace.snapshots:
        for c in itertools.combinations(range(1, 6), 2):
            frame = project(trace, snap, 'coalition', c)
            for card in frame.cards:
                if card.idx not in c:
                    assert card.poly is None
                    assert card.vote is None
                    assert card.received == ()
            assert all(d.receiver == -1 or d.sender in c or d.receiver in c
                       for d in frame.visible_deals)


def test_public_reconstruction_each_subset():
    trace = load_demo()
    last = trace.snapshots[-1]
    shares = {d.sender: d.value for d in last.all_deals if d.receiver == -1}
    assert len(shares) == 5
    for c in itertools.combinations(shares, 3):
        assert reconstruct_public({j: shares[j] for j in c}, 3, 101) == 3


def test_ambiguity_when_fewer_than_threshold():
    trace = load_demo()
    shares = {d.sender: d.value for d in trace.snapshots[-1].all_deals if d.receiver == -1}
    pts = {1: shares[1], 2: shares[2]}
    a, b = alternative_curves(pts, 3, 101)
    assert (a[0], b[0]) == (0, 1)
    for x, y in pts.items():
        assert horner(a, x, 101) == y
        assert horner(b, x, 101) == y


def test_json_validates_indexes():
    trace = load_demo()
    path = Path(__file__).resolve().parents[1] / 'ui_layer' / 'sample_trace.json'
    data = json.loads(path.read_text(encoding='utf8'))
    data['snapshots'][0]['votes'] = [1] * 6
    import pytest
    with pytest.raises(ValueError, match='индекс 0'):
        trace_from_json(json.dumps(data))
