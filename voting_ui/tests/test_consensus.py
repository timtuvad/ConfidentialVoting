"""Дополнительные тесты консенсуса с новой общей Core Board.

Полный набор проверок миграции находится в test_core_board.py.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'voting_ui'))

from src.core.board import MaliciousReport
from threshold_voting.consensus import Evidence, majority_exclusions, manual_detector, demo_oracle
from threshold_voting.simulation import Simulation
from ui_layer.source import load_backend
from ui_layer.projections import project

OPTIONS = ('A', 'B', 'C', 'D')
CHOICES = (0, 1, 2, 1, 3)


def make(**kwargs):
    return Simulation(5, 3, 101, 17, CHOICES, OPTIONS, **kwargs)


def test_no_detector_cannot_magically_identify_attackers():
    for kind in ('invalid_vote', 'wrong_private_share', 'wrong_public_share'):
        sim = make(scenario=kind, attacker=2, enable_consensus=True)
        assert sim.snapshots[-1].consensus_status == 'unresolved'
        assert sim.snapshots[-1].final_result is None
        assert sim.snapshots[-1].excluded_participants == ()


def test_manual_and_oracle_work_without_real_detector():
    for detector in (manual_detector({1: (2,), 3: (2,), 4: (2,)}), demo_oracle(2)):
        sim = make(scenario='wrong_private_share', attacker=2,
                   enable_consensus=True, detector=detector)
        assert sim.snapshots[-1].final_result == (1, 1, 1, 1)
        assert sim.snapshots[-1].excluded_participants == (2,)


def test_detector_gets_only_local_private_data_and_public_board():
    calls = []
    def detector(observer, suspect, evidence):
        assert isinstance(evidence, Evidence)
        assert evidence.observer == observer
        assert not hasattr(evidence, 'attack_info')
        assert not hasattr(evidence, 'all_votes')
        assert set(evidence.received) == set(evidence.active_participants) - {observer}
        assert all(isinstance(value, tuple) for value in evidence.received.values())
        calls.append((observer, suspect, evidence.round_number))
        return False
    sim = make(enable_consensus=True, detector=detector)
    assert len(calls) == 20
    assert sim.snapshots[-1].consensus_status == 'finalized'


def test_reports_are_visible_even_in_participant_view_but_secrets_not():
    trace = load_backend(5, 3, 101, 17, CHOICES, OPTIONS,
                         scenario='invalid_vote', attacker=2,
                         enable_consensus=True, detector_mode='demo_oracle')
    snap = trace.snapshots[-1]
    view = project(trace, snap, 'participant', (1,))
    assert len(view.reports) >= 20
    assert all(isinstance(r.is_malicious, bool) for r in view.reports)
    assert view.cards[1].vote is None
    assert view.cards[1].polys is None
    assert view.attack_info is None
    assert view.board_events


def test_malicious_report_votes_once_per_round_and_stale_reports_do_not_count():
    active = (1, 2, 3, 4, 5)
    reports = (
        MaliciousReport(1, 1, 2, True), MaliciousReport(1, 3, 2, True),
        MaliciousReport(1, 4, 2, True), MaliciousReport(2, 5, 2, True),
    )
    excluded, tallies = majority_exclusions(active, reports, 1)
    assert excluded == (2,) and tallies[2] == 3
    excluded, tallies = majority_exclusions(active, reports, 2)
    assert excluded == () and tallies[2] == 1
