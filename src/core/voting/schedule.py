"""Будущие дедлайны этапов. Пока не заданы, ограничения выключены."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class VotingSchedule:
    registration_deadline: datetime | None = None
    voting_deadline: datetime | None = None
    reporting_deadline: datetime | None = None

    def __post_init__(self) -> None:
        dates = (self.registration_deadline, self.voting_deadline, self.reporting_deadline)
        if any(d is not None and (d.tzinfo is None or d.utcoffset() is None) for d in dates):
            raise ValueError('Дедлайны должны содержать информацию о часовом поясе')
        configured = tuple(d for d in dates if d is not None)
        if any(a > b for a, b in zip(configured, configured[1:])):
            raise ValueError('Дедлайны должны идти в неубывающем порядке')

    @staticmethod
    def _before(deadline: datetime | None, now: datetime) -> bool:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError('Для дедлайна нужны часы с часовым поясом')
        return deadline is None or now < deadline

    def registration_open(self, now: datetime) -> bool:
        return self._before(self.registration_deadline, now)

    def voting_open(self, now: datetime) -> bool:
        return self._before(self.voting_deadline, now)

    def reporting_open(self, now: datetime) -> bool:
        return self._before(self.reporting_deadline, now)
