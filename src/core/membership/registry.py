"""Постоянные ID и состав участников. Реестр не выполняет голосование."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MembershipEvent:
    """Событие изменения состава (для журнала и последующего blockchain-адаптера)."""
    kind: str  # registered | excluded
    participant_id: int
    round_id: int | None = None


class ParticipantRegistry:
    """Индексы ID постоянны и одновременно являются x координатами Шамира.

    Ни register, ни exclude не перенумеровывают оставшихся участников.
    Новые участники допускаются только до seal() (старта голосования).
    """

    def __init__(self) -> None:
        self._registered: set[int] = set()
        self._excluded: set[int] = set()
        self._sealed = False
        self._events: list[MembershipEvent] = []

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def all_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._registered))

    @property
    def active_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._registered - self._excluded))

    @property
    def excluded_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._excluded))

    @property
    def events(self) -> tuple[MembershipEvent, ...]:
        return tuple(self._events)

    def register(self, participant_id: int) -> None:
        """Зарегистрировать уникальный положительный ID до старта."""
        if self._sealed:
            raise RuntimeError('Регистрация закрыта после начала голосования')
        if type(participant_id) is not int or participant_id < 1:
            raise ValueError('ID участника должен быть положительным целым')
        if participant_id in self._registered:
            raise ValueError(f'Участник {participant_id} уже зарегистрирован')
        self._registered.add(participant_id)
        self._events.append(MembershipEvent('registered', participant_id))

    def seal(self) -> None:
        """Необратимо закрыть регистрацию."""
        self._sealed = True

    def exclude_many(self, participant_ids: tuple[int, ...], round_id: int) -> None:
        """Исключить атомарно весь набор; ID остаются зарезервированными."""
        if not self._sealed:
            raise RuntimeError('Исключения разрешены только после старта')
        if type(round_id) is not int or round_id < 1:
            raise ValueError('Номер раунда исключения должен быть >= 1')
        ids = tuple(participant_ids)
        if len(ids) != len(set(ids)):
            raise ValueError('Повторные ID в запросе на исключение')
        if any(i not in self._registered or i in self._excluded for i in ids):
            raise ValueError('Исключить можно только активных участников')
        for i in ids:
            self._excluded.add(i)
            self._events.append(MembershipEvent('excluded', i, round_id))
