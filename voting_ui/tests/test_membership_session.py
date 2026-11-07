"""Новые тесты Registry / Admission / Exclusion / Session без зависимостей от Streamlit."""
from datetime import datetime, timezone, timedelta
import pytest

from src.core.board import InMemoryBoard, MaliciousReport
from src.core.membership import ParticipantRegistry, ParticipantAdmission, ParticipantExclusion, ExclusionDecision
from src.core.voting import VotingSchedule, VotingSession, SessionPhase, NextAction


def report_all(board, voters, suspects, round_id):
    for reporter in voters:
        for suspect in voters:
            if reporter != suspect:
                board.post_malicious_report(MaliciousReport(
                    round_id, reporter, suspect, suspect in suspects))


def test_registry_stable_ids_and_atomic_exclusion():
    r = ParticipantRegistry()
    adm = ParticipantAdmission(r)
    adm.register_many((1, 3, 5))
    assert r.all_ids == (1, 3, 5)
    assert r.active_ids == (1, 3, 5)
    with pytest.raises(ValueError):
        adm.register_many((7, 3))
    assert r.all_ids == (1, 3, 5)
    adm.close()
    r.exclude_many((3,), round_id=1)
    assert r.active_ids == (1, 5)
    assert r.excluded_ids == (3,)
    assert [e.kind for e in r.events] == ['registered'] * 3 + ['excluded']
    with pytest.raises(RuntimeError):
        adm.register(7)
    with pytest.raises(ValueError):
        r.exclude_many((1, 2), round_id=2)
    assert r.active_ids == (1, 5)


def test_schedule_and_deadline_are_optional_and_injectable():
    now = datetime(2026, 11, 6, 16, tzinfo=timezone.utc)
    sch = VotingSchedule(registration_deadline=now + timedelta(minutes=1))
    r = ParticipantRegistry()
    adm = ParticipantAdmission(r, sch, clock=lambda: now)
    adm.register(1)
    with pytest.raises(ValueError):
        VotingSchedule(registration_deadline=datetime(2026, 11, 6))
    expired = ParticipantAdmission(r, sch, clock=lambda: now + timedelta(minutes=2))
    assert not expired.can_register(2)
    with pytest.raises(ValueError):
        expired.register(2)


def test_session_decision_recount_from_board_reports():
    board = InMemoryBoard()
    session = VotingSession(3, board, range(1, 6))
    session.start()
    session.begin_reporting(1)
    report_all(board, session.active_ids, {2}, round_id=1)
    result = session.process_reports(1)
    assert result.action == NextAction.RECOUNT
    assert result.excluded_ids == (2,)
    assert result.active_ids == (1, 3, 4, 5)
    assert result.decision.required_votes == 3
    assert session.current_threshold == 3
    assert session.phase == SessionPhase.RECOUNT_REQUIRED
    session.mark_recalculated()
    session.begin_reporting(2)
    report_all(board, session.active_ids, set(), 2)
    second = session.process_reports(2)
    assert second.action == NextAction.VERIFY
    session.finish(True)
    assert session.phase == SessionPhase.FINALIZED


def test_session_restart_reduces_threshold_without_reindexing():
    board = InMemoryBoard()
    session = VotingSession(4, board, range(1, 6))
    session.start()
    session.begin_reporting(1)
    report_all(board, session.active_ids, {2, 3}, 1)
    res = session.process_reports(1)
    assert res.action == NextAction.RESTART
    assert res.active_ids == (1, 4, 5)
    assert res.current_threshold == 3
    assert session.phase == SessionPhase.RESTART_REQUIRED
    session.mark_recalculated()
    session.begin_reporting(2)
    report_all(board, session.active_ids, {4}, 2)
    res2 = session.process_reports(2)
    assert res2.action == NextAction.RESTART
    assert res2.active_ids == (1, 5)
    assert res2.current_threshold == 2
    assert session.registry.all_ids == (1, 2, 3, 4, 5)


def test_session_terminates_on_one_or_zero_left():
    for suspects, expected in [({2, 3, 4, 5}, (1,)), ({1, 2, 3, 4, 5}, ())]:
        board = InMemoryBoard()
        session = VotingSession(3, board, range(1, 6))
        session.start()
        session.begin_reporting(1)
        report_all(board, session.active_ids, suspects, 1)
        res = session.process_reports(1)
        assert res.action == NextAction.TERMINATE
        assert session.active_ids == expected
        assert session.phase == SessionPhase.TERMINATED
        with pytest.raises(RuntimeError):
            session.mark_recalculated()


def test_counting_strict_majority_and_no_duplicate_reports():
    board = InMemoryBoard()
    session = VotingSession(3, board, range(1, 5))
    session.start()
    session.begin_reporting(1)
    board.post_malicious_report(MaliciousReport(1, 1, 3, True))
    board.post_malicious_report(MaliciousReport(1, 2, 3, True))
    dec = session.exclusion.evaluate(1)
    assert dec.excluded_ids == ()  # 2/4 недостаточно
    board.post_malicious_report(MaliciousReport(1, 4, 3, True))
    dec = session.exclusion.evaluate(1)
    assert dec.excluded_ids == (3,)
    assert dec.required_votes == 3
    with pytest.raises(ValueError):
        board.post_malicious_report(MaliciousReport(1, 4, 3, False))


def test_exclusion_decision_rejected_if_membership_changed():
    board = InMemoryBoard()
    session = VotingSession(3, board, range(1, 5))
    session.start()
    session.begin_reporting(1)
    board.post_malicious_report(MaliciousReport(1, 1, 3, True))
    dec = session.exclusion.evaluate(1)
    session.registry.exclude_many((2,), 1)
    with pytest.raises(RuntimeError):
        session.exclusion.apply(dec)


def test_session_without_enough_people_at_start():
    s = VotingSession(5, InMemoryBoard(), (1,))
    s.start()
    assert s.phase == SessionPhase.TERMINATED
    s2 = VotingSession(5, InMemoryBoard(), (1, 2, 3))
    s2.start()
    assert s2.current_threshold == 3


def test_cannot_inject_fake_exclusion_decision():
    board = InMemoryBoard()
    session = VotingSession(3, board, range(1, 5))
    session.start()
    session.begin_reporting(1)
    fake = ExclusionDecision(1, (1, 2, 3, 4), (2,), ((1, 0), (2, 4), (3, 0), (4, 0)), 3)
    with pytest.raises(ValueError):
        session.exclusion.apply(fake)
    assert session.active_ids == (1, 2, 3, 4)
