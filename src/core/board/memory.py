"""Пассивная InMemoryBoard с тремя уровнями видимости.

- PublicCommitments / MaliciousReport: публичные.
- PrivateShare (f_i(j)): только получателю по его индексу.
- ParticipantTotalShare (s_j=F(j)): только участникам данного раунда.

ВАЖНО: это модель доступа по аргументу reader_id, а не аутентификация или
шифрование. Метод read_all() — исключительно для тестов и God View.
"""
from __future__ import annotations
from copy import deepcopy
from .interface import Board
from .messages import (BoardEntry, BoardMessage, PrivateShare, PublicCommitments,
                       ParticipantTotalShare, MaliciousReport, ReportKind)


class InMemoryBoard(Board):
    def __init__(self) -> None:
        self._entries: list[BoardEntry] = []
        self._logical_keys: set[tuple[object, ...]] = set()
        self._voters: dict[int, frozenset[int]] = {}

    @staticmethod
    def _check_ids(round_id: int, *ids: int) -> None:
        if type(round_id) is not int or round_id < 0:
            raise ValueError('round_id должен быть неотрицательным целым')
        if any(type(x) is not int or x < 1 for x in ids):
            raise ValueError('ID участников должны быть положительными целыми')

    def open_round(self, round_id: int, voter_ids: tuple[int, ...]) -> None:
        self._check_ids(round_id, *voter_ids)
        if len(voter_ids) < 2 or len(voter_ids) != len(set(voter_ids)):
            raise ValueError('Нужны минимум два различных голосующих')
        if round_id in self._voters:
            raise ValueError('Раунд уже открыт')
        self._voters[round_id] = frozenset(voter_ids)

    def _require_voter(self, viewer_id: int, round_id: int) -> None:
        self._check_ids(round_id, viewer_id)
        if viewer_id not in self._voters.get(round_id, frozenset()):
            raise PermissionError('Доли этого раунда доступны только голосующим')

    def _require_sender(self, round_id: int, sender: int) -> None:
        self._require_voter(sender, round_id)

    @staticmethod
    def _check_share_bundle(bundle, expected_x: int) -> None:
        from src.core.verifiable_share import VerifiableShare
        if not isinstance(bundle, tuple) or not bundle:
            raise ValueError('Нужен непустой tuple[VerifiableShare, ...]')
        p = bundle[0].p
        if any(not isinstance(s, VerifiableShare) or s.x.value != expected_x or
               s.p != p for s in bundle):
            raise ValueError('Все VerifiableShare должны иметь правильную x и поле')

    @staticmethod
    def _check_commitment_bundle(bundle) -> None:
        from src.core.commitments import PolynomialCommitments
        if not isinstance(bundle, tuple) or not bundle:
            raise ValueError('Нужен непустой tuple[PolynomialCommitments, ...]')
        p = bundle[0].p
        if any(not isinstance(c, PolynomialCommitments) or c.p != p or
               len(c) != len(bundle[0]) for c in bundle):
            raise ValueError('Commitments должны иметь общую группу и длину')

    def _put(self, key: tuple[object, ...], message: BoardMessage) -> BoardEntry:
        if key in self._logical_keys:
            raise ValueError(f'Повторная запись для {key}')
        entry = BoardEntry(len(self._entries) + 1, deepcopy(message))
        self._entries.append(entry)
        self._logical_keys.add(key)
        return deepcopy(entry)

    def post_commitments(self, message: PublicCommitments) -> BoardEntry:
        self._check_ids(message.round_id, message.sender)
        self._require_sender(message.round_id, message.sender)
        self._check_commitment_bundle(message.value)
        return self._put(('C', message.round_id, message.sender), message)

    def post_private_share(self, message: PrivateShare) -> BoardEntry:
        self._check_ids(message.round_id, message.sender, message.receiver)
        self._require_sender(message.round_id, message.sender)
        self._require_voter(message.receiver, message.round_id)
        if message.sender == message.receiver:
            raise ValueError('Свою долю участник вычисляет локально')
        self._check_share_bundle(message.value, message.receiver)
        return self._put(('f', message.round_id, message.sender, message.receiver), message)

    def post_total_share(self, message: ParticipantTotalShare) -> BoardEntry:
        self._check_ids(message.round_id, message.sender)
        self._require_sender(message.round_id, message.sender)
        self._check_share_bundle(message.value, message.sender)
        return self._put(('s', message.round_id, message.sender), message)

    def post_malicious_report(self, message: MaliciousReport) -> BoardEntry:
        self._check_ids(message.round_id, message.reporter, message.suspect)
        self._require_sender(message.round_id, message.reporter)
        self._require_voter(message.suspect, message.round_id)
        if message.reporter == message.suspect:
            raise ValueError('Нельзя жаловаться на себя')
        if not isinstance(message.kind, ReportKind):
            raise TypeError('Неверный вид MaliciousReport')
        if type(message.source_message_id) is not int or message.source_message_id < 1:
            raise ValueError('Неверный source_message_id')
        if message.kind == ReportKind.PRIVATE_SHARE:
            if message.revealed_share is None:
                raise ValueError('Приватная жалоба должна раскрывать VerifiableShare')
            self._check_share_bundle(message.revealed_share, message.reporter)
        elif message.revealed_share is not None:
            raise ValueError('Суммарная доля уже доступна голосующим; раскрытие не требуется')
        return self._put(('R', message.round_id, message.reporter, message.suspect,
                          message.kind.value), message)

    def get_commitments(self, round_id: int) -> tuple[PublicCommitments, ...]:
        self._check_ids(round_id)
        return tuple(deepcopy(e.message) for e in self._entries
                     if isinstance(e.message, PublicCommitments)
                     and e.message.round_id == round_id)

    def get_private_shares(self, receiver: int, round_id: int) -> tuple[PrivateShare, ...]:
        self._require_voter(receiver, round_id)
        return tuple(deepcopy(e.message) for e in self._entries
                     if isinstance(e.message, PrivateShare)
                     and e.message.round_id == round_id
                     and e.message.receiver == receiver)

    def get_total_shares(self, viewer_id: int, round_id: int) -> tuple[ParticipantTotalShare, ...]:
        self._require_voter(viewer_id, round_id)
        return tuple(deepcopy(e.message) for e in self._entries
                     if isinstance(e.message, ParticipantTotalShare)
                     and e.message.round_id == round_id)

    def get_malicious_reports(self, round_id: int) -> tuple[MaliciousReport, ...]:
        self._check_ids(round_id)
        return tuple(deepcopy(e.message) for e in self._entries
                     if isinstance(e.message, MaliciousReport)
                     and e.message.round_id == round_id)

    def get_source_for_report(self, report: MaliciousReport, viewer_id: int) -> (
        PrivateShare | ParticipantTotalShare | None
    ):
        """Только независимому проверяющему-участнику, после публикации жалобы.

        В MVP сравнение выполняется с открытым объектом. После реального
        шифрования проверка соответствия источнику потребует доказательства.
        """
        self._require_voter(viewer_id, report.round_id)
        if report not in self.get_malicious_reports(report.round_id):
            raise ValueError('Жалоба ещё не опубликована на доске')
        if not 1 <= report.source_message_id <= len(self._entries):
            return None
        source = self._entries[report.source_message_id - 1].message
        if source.round_id != report.round_id or source.sender != report.suspect:
            return None
        if (report.kind == ReportKind.PRIVATE_SHARE
                and isinstance(source, PrivateShare)
                and source.receiver == report.reporter):
            return deepcopy(source)
        if report.kind == ReportKind.TOTAL_SHARE and isinstance(source, ParticipantTotalShare):
            return deepcopy(source)
        return None

    def read_all(self) -> tuple[BoardEntry, ...]:
        """Только системная диагностика. Не использовать для получения долей UI."""
        return tuple(deepcopy(e) for e in self._entries)
