"""Публичные MaliciousReport -> строгое большинство -> изменение реестра."""
from __future__ import annotations
from dataclasses import dataclass
from ..board import Board, MaliciousReport
from .registry import ParticipantRegistry


@dataclass(frozen=True, slots=True)
class ExclusionDecision:
    round_id: int
    electorate: tuple[int, ...]
    excluded_ids: tuple[int, ...]
    tallies: tuple[tuple[int, int], ...]
    required_votes: int

    @property
    def votes(self) -> dict[int, int]:
        return dict(self.tallies)


def count_reports(electorate: tuple[int, ...], reports: tuple[MaliciousReport, ...],
                  round_id: int) -> ExclusionDecision:
    """Подсчитать репорты от уникальных активных участников (не учитывая отсутствующие)."""
    if type(round_id) is not int or round_id < 1:
        raise ValueError('Номер раунда для MaliciousReport должен быть >= 1')
    ids = tuple(electorate)
    if len(ids) != len(set(ids)) or any(type(i) is not int or i <= 0 for i in ids):
        raise ValueError('Некорректный состав электората')
    eligible = set(ids)
    seen: set[tuple[int, int]] = set()
    tallies = {i: 0 for i in ids}
    for report in reports:
        if report.round_id != round_id:
            continue  # история других раундов не учитывается
        if (report.reporter not in eligible or report.suspect not in eligible
                or report.reporter == report.suspect):
            raise ValueError('Отчёт от неактивного участника или самообвинение')
        if type(report.is_malicious) is not bool:
            raise TypeError('Решение MaliciousReport должно быть bool')
        key = (report.reporter, report.suspect)
        if key in seen:
            raise ValueError('Повторный MaliciousReport в одном раунде')
        seen.add(key)
        if report.is_malicious:
            tallies[report.suspect] += 1
    minimum = len(ids) // 2 + 1
    excluded = tuple(i for i in ids if tallies[i] >= minimum)
    return ExclusionDecision(round_id, ids, excluded, tuple(sorted(tallies.items())), minimum)


class ParticipantExclusion:
    """Считает отчёты с Board и изменяет Registry; Board не считает голоса."""

    def __init__(self, registry: ParticipantRegistry, board: Board):
        self.registry = registry
        self.board = board

    def evaluate(self, round_id: int) -> ExclusionDecision:
        return count_reports(self.registry.active_ids,
                             self.board.get_malicious_reports(round_id), round_id)

    def apply(self, decision: ExclusionDecision) -> tuple[int, ...]:
        """Защита от решения для устаревшего состава участников."""
        if decision.electorate != self.registry.active_ids:
            raise RuntimeError('Состав участников изменился после подсчёта отчётов')
        if decision != self.evaluate(decision.round_id):
            raise ValueError('Нельзя применить решение, не подтверждённое записями Board')
        self.registry.exclude_many(decision.excluded_ids, decision.round_id)
        return decision.excluded_ids

    def exclude_from_reports(self, round_id: int) -> ExclusionDecision:
        result = self.evaluate(round_id)
        self.apply(result)
        return result
