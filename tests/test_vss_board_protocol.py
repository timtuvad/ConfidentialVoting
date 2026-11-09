"""Запускать на ветке feature/PedersenVSS после применения патча.

    python -m pytest -q tests/test_vss_board_protocol.py
"""
import pytest

from src.core.board import (InMemoryBoard, PrivateShare, PublicCommitments,
                            MaliciousReport, ReportKind)
from src.core.environment import CryptographyEnvironment
from src.core.vector import VectorShamirCryptography
from src.core.membership import ParticipantRegistry, VSSAdjudicator, VerdictKind
from src.core.shamir import ShamirCryptography
from threshold_voting.vss_protocol import VSSVotingProtocol


def test_valid_vss_and_board():
    env = CryptographyEnvironment.random(23, 3)
    a = VectorShamirCryptography(env, 1, (1, 0, 0))
    b = VectorShamirCryptography(env, 2, (0, 1, 0))
    board = InMemoryBoard()
    board.open_round(0, (1, 2, 3))
    board.post_commitments(PublicCommitments(0, 1, a.commitments))
    sent = a.verifiable_share_to(2)
    entry = board.post_private_share(PrivateShare(0, 1, 2, sent))
    assert b.verify(board.get_private_shares(2, 0)[0].value, a.commitments)
    assert not board.get_private_shares(1, 0)
    with pytest.raises(ValueError, match='Повторная'):
        board.post_private_share(PrivateShare(0, 1, 2, sent))
    assert board.read_all()[1].message_id == entry.message_id


def test_verified_private_complaint_excludes_dealer():
    env = CryptographyEnvironment.random(23, 3)
    a = VectorShamirCryptography(env, 1, (1, 0))
    b = VectorShamirCryptography(env, 2, (0, 1))
    board = InMemoryBoard()
    board.open_round(0, (1, 2, 3))
    board.post_commitments(PublicCommitments(0, 1, a.commitments))
    original = a.verifiable_share_to(2)
    old = original[0]
    from src.core.verifiable_share import VerifiableShare
    from src.core.algebra.field import F_pElement
    altered = (VerifiableShare(old.x, F_pElement(23, old.value.value + 7),
                               old.blinding_share), original[1])
    entry = board.post_private_share(PrivateShare(0, 1, 2, altered))
    report = MaliciousReport(0, 2, 1, ReportKind.PRIVATE_SHARE, entry.message_id, altered)
    board.post_malicious_report(report)
    reg = ParticipantRegistry((1, 2, 3))
    judge = VSSAdjudicator(board, b.coordinates[0].verify,
                           ShamirCryptography.commitments_of_total_sum)
    verdicts = judge.apply_reports(reg, 0)
    assert len(verdicts) == 1
    assert verdicts[0].kind == VerdictKind.DEALER_GUILTY
    assert reg.active_ids == (2, 3)


def test_false_complaint_excludes_reporter():
    env = CryptographyEnvironment.random(23, 2)
    a = VectorShamirCryptography(env, 1, (1, 0))
    b = VectorShamirCryptography(env, 2, (0, 1))
    board = InMemoryBoard()
    board.open_round(0, (1, 2, 3))
    board.post_commitments(PublicCommitments(0, 1, a.commitments))
    original = a.verifiable_share_to(2)
    entry = board.post_private_share(PrivateShare(0, 1, 2, original))
    report = MaliciousReport(0, 2, 1, ReportKind.PRIVATE_SHARE, entry.message_id, original)
    board.post_malicious_report(report)
    reg = ParticipantRegistry((1, 2))
    judge = VSSAdjudicator(board, b.coordinates[0].verify,
                           ShamirCryptography.commitments_of_total_sum)
    verdicts = judge.apply_reports(reg, 0)
    assert verdicts[0].kind == VerdictKind.REPORTER_GUILTY
    assert reg.active_ids == (1,)


@pytest.mark.parametrize('scenario,excluded', [
    ('honest', ()),
    ('wrong_private_share', (2,)),
    ('wrong_public_share', (2,)),
    ('false_complaint_changed', (2,)),
    ('false_complaint_valid', (2,)),
])
def test_protocol_cases(scenario, excluded):
    protocol = VSSVotingProtocol(
        choices=(0, 1, 2, 1, 0), variants=3, t=3, p=23,
        scenario=scenario, attacker=2, receiver=4,
    )
    result = protocol.run()
    assert result.status == 'complete'
    assert result.excluded_ids == excluded
    assert result.counts == ((2, 2, 1) if not excluded else (2, 1, 1))
    assert result.active_ids == ((1, 2, 3, 4, 5) if not excluded else (1, 3, 4, 5))
    assert len(result.rounds) == (1 if not excluded else 2)


def test_invalid_vote_not_detected_without_ohe_zkp():
    result = VSSVotingProtocol((0, 1, 2, 1, 0), 3, 3, 23,
                               scenario='invalid_vote', attacker=2).run()
    assert result.status == 'complete'
    assert result.excluded_ids == ()
    assert result.counts == (2, 4, 1)  # некорректный голос 3 вместо 1


def test_restart_with_lower_threshold():
    protocol = VSSVotingProtocol((0, 1, 2, 0), 3, 4, 23,
                                 scenario='wrong_private_share', attacker=2)
    result = protocol.run()
    assert result.status == 'complete'
    assert result.threshold == 3
    assert result.active_ids == (1, 3, 4)
    assert result.counts == (2, 0, 1)
    assert result.rounds[0].threshold == 4
    assert result.rounds[1].threshold == 3


def test_terminate_if_fewer_than_two():
    result = VSSVotingProtocol((0, 1), 2, 2, 23,
                               scenario='wrong_private_share', attacker=2).run()
    assert result.status == 'terminated_too_few'
    assert result.counts is None
    assert len(result.active_ids) == 1


def test_missing_source_cannot_exclude_anyone():
    env = CryptographyEnvironment.random(23, 2)
    a = VectorShamirCryptography(env, 1, (1, 0))
    b = VectorShamirCryptography(env, 2, (0, 1))
    board = InMemoryBoard()
    board.open_round(0, (1, 2, 3))
    board.post_commitments(PublicCommitments(0, 1, a.commitments))
    report = MaliciousReport(0, 2, 1, ReportKind.PRIVATE_SHARE,
                             9876, a.verifiable_share_to(2))
    board.post_malicious_report(report)
    reg = ParticipantRegistry((1, 2))
    judge = VSSAdjudicator(board, b.coordinates[0].verify,
                           ShamirCryptography.commitments_of_total_sum)
    verdicts = judge.apply_reports(reg, 0)
    assert verdicts[0].kind == VerdictKind.UNRESOLVED
    assert reg.active_ids == (1, 2)


def test_board_does_not_return_its_internal_mutable_objects():
    env = CryptographyEnvironment.random(23, 2)
    a = VectorShamirCryptography(env, 1, (1, 0))
    board = InMemoryBoard()
    board.open_round(0, (1, 2, 3))
    board.post_commitments(PublicCommitments(0, 1, a.commitments))
    board.post_private_share(PrivateShare(0, 1, 2, a.verifiable_share_to(2)))
    retrieved = board.get_private_shares(2, 0)
    original = retrieved[0].value[0].value.value
    # Нельзя изменением результата read() заменить запись в Board.
    from src.core.algebra.field import F_pElement
    retrieved[0].value[0]._value = F_pElement(23, original + 3)
    assert board.get_private_shares(2, 0)[0].value[0].value.value == original


def test_false_public_accusation_excludes_reporter_if_adjudicated():
    protocol = VSSVotingProtocol((0, 1, 2), 3, 2, 23)
    result = protocol.run()
    assert result.status == 'complete'
    from src.core.board import ParticipantTotalShare
    entry = next(e for e in protocol.board.read_all()
                 if isinstance(e.message, ParticipantTotalShare) and e.message.sender == 2)
    report = MaliciousReport(0, 1, 2, ReportKind.TOTAL_SHARE, entry.message_id)
    protocol.board.post_malicious_report(report)
    v = protocol._adjudicator().resolve(report, (1, 2, 3), observer_id=1)
    assert v.kind == VerdictKind.REPORTER_GUILTY
    assert v.guilty_id == 1


def test_p_101_is_rejected_by_crypto_environment():
    with pytest.raises(ValueError):
        VSSVotingProtocol((0, 1, 0), 2, 2, 101)


def test_private_and_total_share_have_different_read_permissions():
    """До реального шифрования ограничения существуют только на уровне API."""
    from src.core.board import ParticipantTotalShare
    env = CryptographyEnvironment.random(23, 2)
    actors = {i: VectorShamirCryptography(env, i, (1, 0)) for i in (1, 2, 3)}
    board = InMemoryBoard()
    board.open_round(0, (1, 2, 3))
    board.post_commitments(PublicCommitments(0, 1, actors[1].commitments))
    private = actors[1].verifiable_share_to(2)
    board.post_private_share(PrivateShare(0, 1, 2, private))
    total = actors[1].verifiable_share_to(1)
    board.post_total_share(ParticipantTotalShare(0, 1, total))

    assert len(board.get_private_shares(receiver=2, round_id=0)) == 1
    assert not board.get_private_shares(receiver=3, round_id=0)
    assert len(board.get_total_shares(viewer_id=2, round_id=0)) == 1
    assert len(board.get_total_shares(viewer_id=3, round_id=0)) == 1
    assert len(board.get_commitments(0)) == 1
    with pytest.raises(PermissionError):
        board.get_private_shares(receiver=99, round_id=0)
    with pytest.raises(PermissionError):
        board.get_total_shares(viewer_id=99, round_id=0)


def test_old_and_new_rounds_keep_membership_access_separate():
    """Исключённый участник не читает доли последующих раундов."""
    p = VSSVotingProtocol((0, 1, 2, 1, 0), 3, 3, 23,
                          scenario='wrong_public_share', attacker=2)
    res = p.run()
    assert res.status == 'complete'
    assert res.excluded_ids == (2,)
    assert len(p.board.get_total_shares(viewer_id=2, round_id=0)) == 5
    with pytest.raises(PermissionError):
        p.board.get_total_shares(viewer_id=2, round_id=1)
    assert len(p.board.get_total_shares(viewer_id=1, round_id=1)) == 4


def test_report_source_requires_authenticated_round_member_in_future():
    """Скрытый источник раскрывается проверяющим только при опубликованном report."""
    env = CryptographyEnvironment.random(23, 2)
    actor = VectorShamirCryptography(env, 1, (1, 0))
    board = InMemoryBoard()
    board.open_round(0, (1, 2))
    original = actor.verifiable_share_to(2)
    entry = board.post_private_share(PrivateShare(0, 1, 2, original))
    report = MaliciousReport(0, 2, 1, ReportKind.PRIVATE_SHARE,
                             entry.message_id, original)
    with pytest.raises(ValueError, match='не опубликована'):
        board.get_source_for_report(report, viewer_id=1)
    board.post_malicious_report(report)
    with pytest.raises(PermissionError):
        board.get_source_for_report(report, viewer_id=100)
    assert board.get_source_for_report(report, viewer_id=1).value == original


def test_total_share_not_present_in_public_board_getters():
    """Обычные публичные методы не дают F(j), даже если запись есть в памяти."""
    from src.core.board import ParticipantTotalShare
    env = CryptographyEnvironment.random(23, 2)
    actor = VectorShamirCryptography(env, 1, (1, 0))
    board = InMemoryBoard()
    board.open_round(0, (1, 2))
    board.post_commitments(PublicCommitments(0, 1, actor.commitments))
    board.post_total_share(ParticipantTotalShare(0, 1, actor.verifiable_share_to(1)))
    assert all(not isinstance(x, ParticipantTotalShare)
               for x in board.get_commitments(0) + board.get_malicious_reports(0))
    assert not hasattr(board, 'get_public_shares')
    # read_all() осознанно оставлен только для отладчика God View.
    assert any(isinstance(x.message, ParticipantTotalShare) for x in board.read_all())
