"""Интеграция новой ядровой Board и перезапуск после исключений."""
from __future__ import annotations
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'voting_ui'))

from src.core.board import InMemoryBoard, Board, PrivateShare, PublicShare, MaliciousReport
from threshold_voting.consensus import majority_exclusions, manual_detector
from threshold_voting.simulation import Simulation
from ui_layer.source import load_backend
from ui_layer.projections import project
from ui_layer.contracts import parse_trace, trace_to_json

CHOICES = (0, 1, 2, 1, 3)
OPTIONS = ('A', 'B', 'C', 'D')


def run(t=3, scenario='honest', detector=None, **kw):
    return Simulation(5, t, 101, 17, CHOICES, OPTIONS,
                      scenario=scenario, attacker=2, enable_consensus=True,
                      detector=detector, **kw)


def test_board_three_types_and_receiver_filter():
    board = InMemoryBoard()
    a = PrivateShare(0, 1, 3, (10, 20))
    b = PrivateShare(0, 2, 4, (7, 8))
    c = PublicShare(0, 1, (11, 18))
    d = MaliciousReport(1, 3, 2, True, 'message-1')
    entries = [board.post_private_share(a), board.post_private_share(b),
               board.post_public_share(c), board.post_malicious_report(d)]
    assert [x.message_id for x in entries] == [1, 2, 3, 4]
    assert board.get_private_shares(3, 0) == (a,)
    assert board.get_private_shares(4, 0) == (b,)
    assert board.get_private_shares(5, 0) == ()
    assert board.get_public_shares(0) == (c,)
    assert board.get_malicious_reports(1) == (d,)
    assert board.read_all() == tuple(entries)
    assert board.get_malicious_reports(0) == ()


def test_board_is_append_only_and_rejects_duplicate_keys():
    board = InMemoryBoard()
    board.post_private_share(PrivateShare(0, 1, 2, (1, 3)))
    board.post_public_share(PublicShare(0, 1, (4, 9)))
    board.post_malicious_report(MaliciousReport(1, 1, 2, True))
    with pytest.raises(ValueError):
        board.post_private_share(PrivateShare(0, 1, 2, (3, 3)))
    with pytest.raises(ValueError):
        board.post_public_share(PublicShare(0, 1, (3, 3)))
    with pytest.raises(ValueError):
        board.post_malicious_report(MaliciousReport(1, 1, 2, False))
    with pytest.raises(ValueError):
        board.post_malicious_report(MaliciousReport(1, 2, 2, True))
    assert len(board.read_all()) == 3
    board.post_public_share(PublicShare(2, 1, (6, 3)))
    assert len(board.read_all()) == 4


def test_board_private_ciphertext_placeholder():
    board = InMemoryBoard()
    secret = PrivateShare(3, 2, 4, b'future encrypted payload')
    board.post_private_share(secret)
    assert board.get_private_shares(4, 3) == (secret,)
    assert board.get_private_shares(1, 3) == ()


def test_initial_procedure_posts_all_to_core_board_and_13_steps():
    sim = Simulation(5, 3, 101, 17, CHOICES, OPTIONS)
    assert len(sim.snapshots) == 13
    board = sim.board
    assert isinstance(board, Board)
    assert sum(len(board.get_private_shares(j, 0)) for j in range(1, 6)) == 20
    assert len(board.get_public_shares(0)) == 5
    assert board.get_malicious_reports(0) == ()
    assert set(sim.snapshots[-1].reconstructed.values()) == {(1, 2, 1, 1)}
    assert len(sim.snapshots[-1].board_events) == 25


@pytest.mark.parametrize('scenario,kwargs', [
    ('honest', {}),
    ('invalid_vote', {'invalid_secret': 3}),
    ('wrong_private_share', {'target_receiver': 4, 'attack_option': 2}),
    ('wrong_public_share', {'attack_option': 2}),
])
def test_recount_after_excluding_attacker(scenario, kwargs):
    sim = run(3, scenario, manual_detector({1: (2,), 3: (2,), 4: (2,)}), **kwargs)
    last = sim.snapshots[-1]
    assert last.consensus_status == 'finalized'
    assert last.current_threshold == 3
    assert last.active_participants == (1, 3, 4, 5)
    assert last.final_result == (1, 1, 1, 1)
    assert len(sim.board.get_public_shares(1)) == 4
    assert len(sim.board.get_malicious_reports(1)) == 20
    assert len(sim.board.get_private_shares(4, 1)) == 0  # повторной раздачи нет
    assert all(s.phase != 'restart' for s in sim.snapshots)
    assert len(sim.board.read_all()) > 25


def test_restart_when_active_less_than_threshold():
    # После удаления P2 и P3 из 5 останется 3 < t=4: новая схема Шамира с t=3.
    sim = run(4, 'honest', manual_detector({1: (2, 3), 4: (2, 3), 5: (2, 3)}))
    final = sim.snapshots[-1]
    assert final.consensus_status == 'finalized'
    assert final.active_participants == (1, 4, 5)
    assert final.current_threshold == 3
    assert final.final_result == (1, 1, 0, 1)
    assert len(sim.board.get_public_shares(0)) == 5
    assert len(sim.board.get_public_shares(1)) == 3
    assert sum(len(sim.board.get_private_shares(j, 1)) for j in (1, 4, 5)) == 6
    assert len(sim.snapshots[0].polys[1][0]) == 4
    assert len(final.polys[1][0]) == 3  # физически созданы новые полиномы
    assert any(s.phase == 'restart' for s in sim.snapshots)
    assert any(s.phase == 'redeal' for s in sim.snapshots)
    assert len(final.reconstructed) == 1  # C(3, 3)


def test_restart_to_two_active_t_equals_two():
    # Из 5 убрали троих (все остальные обвиняют каждого). 2 <= n_active < t=4.
    detector = lambda observer, suspect, evidence: suspect in (2, 3, 4)
    sim = run(4, 'honest', detector)
    last = sim.snapshots[-1]
    assert last.active_participants == (1, 5)
    assert last.current_threshold == 2
    assert last.final_result == (1, 0, 0, 1)
    assert last.consensus_status == 'finalized'
    assert len(sim.board.get_public_shares(1)) == 2
    assert len(sim.board.get_private_shares(1, 1)) == 1
    assert len(sim.board.get_private_shares(5, 1)) == 1


@pytest.mark.parametrize('suspects,expected_active', [
    ((2, 3, 4, 5), (1,)),
    ((1, 2, 3, 4, 5), ()),
])
def test_terminate_immediately_if_fewer_than_two_remain(suspects, expected_active):
    sim = run(3, 'honest', lambda observer, suspect, evidence: suspect in suspects)
    last = sim.snapshots[-1]
    assert last.consensus_status == 'terminated_too_few'
    assert last.final_result is None
    assert last.active_participants == expected_active
    assert not sim.board.get_public_shares(1)
    assert not any(s.phase == 'reaggregate' for s in sim.snapshots)


def test_rounds_and_reports_on_single_board():
    def detector(observer, suspect, evidence):
        return ((evidence.round_number == 1 and suspect == 2) or
                (evidence.round_number == 2 and suspect == 3))
    sim = run(3, 'honest', detector)
    last = sim.snapshots[-1]
    assert last.excluded_participants == (2, 3)
    assert last.final_result == (1, 1, 0, 1)
    assert len(sim.board.get_malicious_reports(1)) == 20
    assert len(sim.board.get_malicious_reports(2)) == 12
    assert len(sim.board.get_malicious_reports(3)) == 6
    assert len(sim.board.get_public_shares(1)) == 4
    assert len(sim.board.get_public_shares(2)) == 3


def test_majority_is_strict_and_uses_active_denominator():
    active = (1, 2, 3, 4)
    reports = (MaliciousReport(1, 1, 3, True),
               MaliciousReport(1, 2, 3, True))
    excluded, count = majority_exclusions(active, reports, 1)
    assert excluded == () and count[3] == 2
    excluded, count = majority_exclusions(active, reports + (MaliciousReport(1, 4, 3, True),), 1)
    assert excluded == (3,) and count[3] == 3


def test_max_rounds_no_false_finalization():
    sim = run(3, 'honest', lambda observer, suspect, evidence: suspect == 2,
              max_consensus_rounds=1)
    assert sim.snapshots[-1].consensus_status == 'max_rounds'
    assert sim.snapshots[-1].final_result is None


def test_board_logs_are_public_metadata_only_in_participant_view():
    trace = load_backend(5, 4, 101, 17, CHOICES, OPTIONS, scenario='honest',
                         enable_consensus=True, detector_mode='manual',
                         manual_accusations={1: (2, 3), 4: (2, 3), 5: (2, 3)})
    assert parse_trace(trace_to_json(trace)) == trace
    snap = trace.snapshots[-1]
    viewer = project(trace, snap, 'participant', (1,))
    assert viewer.t == 3
    assert viewer.final_result == (1, 1, 0, 1)
    assert viewer.cards[1].vote is None
    assert viewer.cards[1].polys is None
    assert viewer.attack_info is None
    assert len(viewer.board_events) > 0
    assert all(len(e) == 5 for e in viewer.board_events)
    assert all(d.receiver == -1 or d.sender == 1 or d.receiver == 1 for d in viewer.visible_deals)
    assert viewer.reports


def test_two_sequential_restarts_reduce_t_and_refresh_polynomials_twice():
    def detector(observer, suspect, evidence):
        if evidence.round_number == 1:
            return suspect in (2, 3)
        if evidence.round_number == 2:
            return suspect == 4
        return False
    sim = run(4, 'honest', detector)
    final = sim.snapshots[-1]
    assert final.consensus_status == 'finalized'
    assert final.current_threshold == 2
    assert final.active_participants == (1, 5)
    assert final.final_result == (1, 0, 0, 1)
    assert len([s for s in sim.snapshots if s.phase == 'restart']) == 2
    assert len(sim.board.get_public_shares(1)) == 3
    assert len(sim.board.get_public_shares(2)) == 2
    assert sum(len(sim.board.get_private_shares(j, 2)) for j in (1, 5)) == 2
    assert len(final.polys[1][0]) == 2


def test_simulation_accepts_injected_board_backend():
    board = InMemoryBoard()
    sim = Simulation(5, 3, 101, 17, CHOICES, OPTIONS, board=board)
    assert sim.board is board
    assert len(board.get_public_shares(0)) == 5


def test_malicious_public_share_remains_malicious_after_unrelated_exclusion():
    # Плохой P2 остался, исключили честного P3: новое F(2) всё ещё подменяется.
    sim = run(3, 'wrong_public_share', manual_detector({1: (3,), 2: (3,), 4: (3,)}))
    last = sim.snapshots[-1]
    assert last.active_participants == (1, 2, 4, 5)
    assert last.consensus_status == 'unresolved'
    assert last.final_result is None
    assert len(set(last.reconstructed.values())) > 1


def test_malicious_private_share_remains_malicious_after_restart():
    # При смене порога P2 снова отправляет некорректную приватную долю.
    sim = run(4, 'wrong_private_share', manual_detector({1: (3, 4), 2: (3, 4), 5: (3, 4)}),
              target_receiver=5)
    last = sim.snapshots[-1]
    assert last.active_participants == (1, 2, 5)
    assert last.current_threshold == 3
    assert last.consensus_status == 'unresolved'
    # При n_active == t_new существует только одна тройка: нет избыточных
    # точек для проверки согласованности, но фактическая сумма всё ещё искажена.
    assert len(last.reconstructed) == 1
    assert next(iter(last.reconstructed.values())) != (1, 1, 0, 1)
    assert len(sim.board.get_private_shares(5, 1)) == 2
