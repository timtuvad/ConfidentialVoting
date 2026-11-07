"""Политика регистрации: отдельна от хранения состава участников."""
from __future__ import annotations
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from .registry import ParticipantRegistry
from ..voting.schedule import VotingSchedule


class ParticipantAdmission:
    """Принимает участников только пока реестр не закрыт и не истёк дедлайн."""

    def __init__(self, registry: ParticipantRegistry,
                 schedule: VotingSchedule | None = None,
                 clock: Callable[[], datetime] | None = None):
        self.registry = registry
        self.schedule = schedule or VotingSchedule()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def can_register(self, participant_id: int) -> bool:
        return (not self.registry.sealed
                and type(participant_id) is int and participant_id > 0
                and participant_id not in self.registry.all_ids
                and self.schedule.registration_open(self.clock()))

    def register(self, participant_id: int) -> None:
        if self.registry.sealed:
            raise RuntimeError('Регистрация закрыта')
        if not self.can_register(participant_id):
            raise ValueError('Регистрация недоступна: повторный ID, дедлайн или закрытая сессия')
        self.registry.register(participant_id)

    def register_many(self, participant_ids: Iterable[int]) -> None:
        """Проверяет всю партию до изменения реестра."""
        if self.registry.sealed:
            raise RuntimeError('Регистрация закрыта')
        ids = tuple(participant_ids)
        if len(ids) != len(set(ids)) or any(not self.can_register(i) for i in ids):
            raise ValueError('Нельзя зарегистрировать весь предложенный набор ID')
        for i in ids:
            self.registry.register(i)

    def close(self) -> None:
        self.registry.seal()
