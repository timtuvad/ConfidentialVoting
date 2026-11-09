"""Независимая проверка MaliciousReport по публичным Pedersen VSS обязательствам.

Решение определяется ДОКАЗАТЕЛЬСТВОМ, а не большинством недоказанных мнений.
Каждый честный участник, имеющий одну и ту же Board, получает один verdict.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from src.core.board import (Board, PrivateShare, ParticipantTotalShare,
                            MaliciousReport, ReportKind)
from .registry import ParticipantRegistry


class VerdictKind(str, Enum):
    DEALER_GUILTY = 'dealer_guilty'
    REPORTER_GUILTY = 'reporter_guilty'
    UNRESOLVED = 'unresolved'


@dataclass(frozen=True, slots=True)
class Verdict:
    report: MaliciousReport
    kind: VerdictKind
    guilty_id: int | None
    reason: str


class VSSAdjudicator:
    """Адаптер к уже реализованному ShamirCryptography.verify.

    verify(share, commitments) -- функция ядра. sum_commitments(list)
    -- ShamirCryptography.commitments_of_total_sum. Доска эти функции
    НЕ вызывает и никаких вычислений не выполняет.
    """
    def __init__(self, board: Board, verify: Callable,
                 sum_commitments: Callable) -> None:
        self.board = board
        self.verify = verify
        self.sum_commitments = sum_commitments

    def _commitments(self, round_id: int) -> dict[int, tuple]:
        return {c.sender: c.value for c in self.board.get_commitments(round_id)}

    def _verify_bundle(self, bundle: tuple, commitments: tuple) -> bool:
        if len(bundle) != len(commitments):
            return False
        try:
            return all(self.verify(share, commitment)
                       for share, commitment in zip(bundle, commitments))
        except (TypeError, ValueError, AttributeError):
            return False

    def _aggregate_commitments(self, round_id: int,
                               active_dealers: tuple[int, ...]) -> tuple | None:
        published = self._commitments(round_id)
        if not active_dealers or any(i not in published for i in active_dealers):
            return None
        m = len(published[active_dealers[0]])
        if not m or any(len(published[i]) != m for i in active_dealers):
            return None
        return tuple(self.sum_commitments([published[i][k]
                                          for i in active_dealers])
                     for k in range(m))

    def check_total_share(self, message: ParticipantTotalShare,
                           active_dealers: tuple[int, ...]) -> bool | None:
        """None — недостаточно обязательств для доказательной проверки."""
        comm = self._aggregate_commitments(message.round_id, active_dealers)
        return None if comm is None else self._verify_bundle(message.value, comm)

    def resolve(self, report: MaliciousReport,
                active_dealers: tuple[int, ...], observer_id: int) -> Verdict:
        """Независимое решение от имени участника текущего раунда.

        Board проверяет observer_id до выдачи источника жалобы.
        Реальная аутентификация будет обеспечиваться сетевым слоем.
        """
        source = self.board.get_source_for_report(report, observer_id)
        if source is None:
            return Verdict(report, VerdictKind.UNRESOLVED, None,
                           'Не найдена исходная запись, связанная с жалобой')

        if report.kind == ReportKind.PRIVATE_SHARE:
            if not isinstance(source, PrivateShare):
                return Verdict(report, VerdictKind.UNRESOLVED, None, 'Неверный тип источника')
            # Сравниваем value, x И blinding_share у VerifiableShare.
            if source.value != report.revealed_share:
                return Verdict(report, VerdictKind.REPORTER_GUILTY,
                               report.reporter, 'Раскрытая доля отличается от оригинала')
            comm = self._commitments(report.round_id).get(report.suspect)
            if comm is None:
                return Verdict(report, VerdictKind.UNRESOLVED, None,
                               'Не опубликованы обязательства дилера')
            if self._verify_bundle(report.revealed_share, comm):
                return Verdict(report, VerdictKind.REPORTER_GUILTY,
                               report.reporter, 'Доля правильная: обвинение ложное')
            return Verdict(report, VerdictKind.DEALER_GUILTY,
                           report.suspect, 'Исходная доля не соответствует обязательствам')

        if report.kind == ReportKind.TOTAL_SHARE:
            if not isinstance(source, ParticipantTotalShare):
                return Verdict(report, VerdictKind.UNRESOLVED, None, 'Неверный тип источника')
            verified = self.check_total_share(source, active_dealers)
            if verified is None:
                return Verdict(report, VerdictKind.UNRESOLVED, None,
                               'Не хватает обязательств для проверки суммы')
            if verified:
                return Verdict(report, VerdictKind.REPORTER_GUILTY,
                               report.reporter, 'Опубликованная сумма корректна')
            return Verdict(report, VerdictKind.DEALER_GUILTY,
                           report.suspect, 'Агрегированная F(j) не соответствует сумме обязательств')
        return Verdict(report, VerdictKind.UNRESOLVED, None, 'Неизвестный тип жалобы')

    def apply_reports(self, registry: ParticipantRegistry, round_id: int,
                      active_dealers: tuple[int, ...] | None = None,
                      observer_id: int | None = None) -> tuple[Verdict, ...]:
        # Фиксируем один electorate на весь этап: исключение — атомарно.
        electorate = registry.active_ids
        if active_dealers is None:
            active_dealers = electorate
        if observer_id is None:
            if not electorate:
                return ()
            observer_id = electorate[0]
        verdicts = tuple(
            self.resolve(report, active_dealers, observer_id)
            for report in self.board.get_malicious_reports(round_id)
            if report.reporter in electorate and report.suspect in electorate
        )
        guilty = tuple(sorted({v.guilty_id for v in verdicts if v.guilty_id is not None}))
        registry.exclude_many(guilty, round_id,
                              evidence=(f'{v.report.source_message_id}:{v.reason}'
                                        for v in verdicts if v.guilty_id in guilty))
        return verdicts
