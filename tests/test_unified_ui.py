"""Проверки видимости сообщений в едином UI. Запуск из корня репозитория."""
from types import SimpleNamespace

from src.core.board import (
    BoardEntry, PrivateShare, ParticipantTotalShare,
    PublicCommitments, MaliciousReport, ReportKind,
)
from voting_ui.app import visible_events, describe, graph_dot


class FakeBoard:
    def __init__(self):
        self.messages = (
            PublicCommitments(0, 1, ('commitments',)),
            PrivateShare(0, 1, 2, ('secret_12',)),
            PrivateShare(0, 1, 3, ('secret_13',)),
            ParticipantTotalShare(0, 2, ('aggregated_2',)),
            MaliciousReport(0, 2, 1, ReportKind.TOTAL_SHARE, 4),
        )
        self.privileged_reads = 0

    def get_commitments(self, r):
        return tuple(m for m in self.messages if isinstance(m, PublicCommitments) and m.round_id == r)

    def get_malicious_reports(self, r):
        return tuple(m for m in self.messages if isinstance(m, MaliciousReport) and m.round_id == r)

    def get_private_shares(self, viewer, r):
        if viewer not in (1, 2, 3):
            raise PermissionError(viewer)
        return tuple(m for m in self.messages if isinstance(m, PrivateShare) and m.receiver == viewer and m.round_id == r)

    def get_total_shares(self, viewer, r):
        if viewer not in (1, 2, 3):
            raise PermissionError(viewer)
        return tuple(m for m in self.messages if isinstance(m, ParticipantTotalShare) and m.round_id == r)

    def read_all(self):
        self.privileged_reads += 1
        return tuple(BoardEntry(i + 1, msg) for i, msg in enumerate(self.messages))


def test_public_never_requests_private_or_total():
    board = FakeBoard()
    events = visible_events(board, 0, 'public')
    assert {e.kind for e in events} == {'PublicCommitments', 'MaliciousReport'}
    assert board.privileged_reads == 0


def test_participant_receives_only_own_private_shares():
    board = FakeBoard()
    events = visible_events(board, 0, 'participant', 2)
    private = [e for e in events if e.kind == 'PrivateShare']
    assert [(e.sender, e.receiver) for e in private] == [(1, 2)]
    assert any(e.kind == 'ParticipantTotalShare' for e in events)
    assert board.privileged_reads == 0


def test_god_view_explicitly_uses_all_messages():
    board = FakeBoard()
    events = visible_events(board, 0, 'god')
    assert len(events) == len(board.messages)
    assert board.privileged_reads == 1


def test_graph_avoids_secret_payloads():
    board = FakeBoard()
    events = visible_events(board, 0, 'participant', 2)
    dot = graph_dot(events, (1, 2, 3), (), events[-1])
    assert 'secret_12' not in dot
    assert 'secret_13' not in dot
    assert 'aggregated_2' not in dot
    assert 'report P1' in dot


def test_foreign_viewer_denied():
    import pytest
    with pytest.raises(PermissionError):
        visible_events(FakeBoard(), 0, 'participant', 99)


def test_coalition_sees_union_of_own_incoming_shares():
    board = FakeBoard()
    events = visible_events(board, 0, 'coalition', (2, 3))
    private = {(e.sender, e.receiver) for e in events if e.kind == 'PrivateShare'}
    assert private == {(1, 2), (1, 3)}
    assert sum(e.kind == 'ParticipantTotalShare' for e in events) == 1
    assert board.privileged_reads == 0
