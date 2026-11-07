"""Единая append-only доска в памяти; не считает голоса и не ищет malicious.

Достоверность `sender` и идентичность клиента здесь НЕ проверяются:
это ответственность внешнего слоя аутентификации (в будущем).
"""
from __future__ import annotations
from .interface import Board
from .messages import BoardEntry, BoardMessage, PrivateShare, PublicShare, MaliciousReport


class InMemoryBoard(Board):
    def __init__(self) -> None:
        self._entries: list[BoardEntry] = []
        self._private_keys: set[tuple[int, int, int]] = set()
        self._public_keys: set[tuple[int, int]] = set()
        self._report_keys: set[tuple[int, int, int]] = set()

    def _append(self, msg: BoardMessage) -> BoardEntry:
        entry = BoardEntry(len(self._entries) + 1, msg)
        self._entries.append(entry)
        return entry

    @staticmethod
    def _check_header(round_id: int, *participants: int) -> None:
        if type(round_id) is not int or round_id < 0:
            raise ValueError('round_id должен быть неотрицательным целым числом')
        if any(type(i) is not int or i < 1 for i in participants):
            raise ValueError('Индексы участников должны быть положительными')

    @staticmethod
    def _check_vector(value: tuple[int, ...] | bytes) -> None:
        if isinstance(value, bytes):
            if not value:
                raise ValueError('Пустой ciphertext')
        elif (not isinstance(value, tuple) or len(value) < 2 or
              any(type(x) is not int or x < 0 for x in value)):
            raise ValueError('Ожидался непустой вектор долей или ciphertext')

    def post_private_share(self, share: PrivateShare) -> BoardEntry:
        self._check_header(share.round_id, share.sender, share.receiver)
        self._check_vector(share.value)
        if share.sender == share.receiver:
            raise ValueError('Приватная доля самому себе не публикуется')
        key = (share.round_id, share.sender, share.receiver)
        if key in self._private_keys:
            raise ValueError('Повторная приватная доля (round, sender, receiver)')
        self._private_keys.add(key)
        return self._append(share)

    def post_public_share(self, share: PublicShare) -> BoardEntry:
        self._check_header(share.round_id, share.sender)
        self._check_vector(share.value)
        if isinstance(share.value, bytes):
            raise ValueError('Публичная доля должна быть открытым OHE-вектором')
        key = (share.round_id, share.sender)
        if key in self._public_keys:
            raise ValueError('Повторная публичная доля (round, sender)')
        self._public_keys.add(key)
        return self._append(share)

    def post_malicious_report(self, report: MaliciousReport) -> BoardEntry:
        self._check_header(report.round_id, report.reporter, report.suspect)
        if type(report.is_malicious) is not bool:
            raise TypeError('is_malicious должен быть bool')
        if report.reporter == report.suspect:
            raise ValueError('Самообвинение не учитывается')
        key = (report.round_id, report.reporter, report.suspect)
        if key in self._report_keys:
            raise ValueError('Повторный MaliciousReport от того же участника')
        self._report_keys.add(key)
        return self._append(report)

    def get_private_shares(self, receiver: int, round_id: int) -> tuple[PrivateShare, ...]:
        self._check_header(round_id, receiver)
        return tuple(entry.message for entry in self._entries
                     if isinstance(entry.message, PrivateShare)
                     and entry.message.receiver == receiver
                     and entry.message.round_id == round_id)

    def get_public_shares(self, round_id: int) -> tuple[PublicShare, ...]:
        self._check_header(round_id)
        return tuple(entry.message for entry in self._entries
                     if isinstance(entry.message, PublicShare)
                     and entry.message.round_id == round_id)

    def get_malicious_reports(self, round_id: int) -> tuple[MaliciousReport, ...]:
        self._check_header(round_id)
        return tuple(entry.message for entry in self._entries
                     if isinstance(entry.message, MaliciousReport)
                     and entry.message.round_id == round_id)

    def read_all(self) -> tuple[BoardEntry, ...]:
        """Привилегированный god/debug read, никогда не отдавать в Participant View."""
        return tuple(self._entries)
