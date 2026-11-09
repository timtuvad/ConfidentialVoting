"""Постоянные номера участников при исключении. Нумерация не меняется."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExclusionEvent:
    round_id: int
    excluded: tuple[int, ...]
    evidence: tuple[str, ...]


class ParticipantRegistry:
    def __init__(self, participant_ids):
        ids = tuple(sorted(participant_ids))
        if len(ids) < 2 or len(ids) != len(set(ids)) or any(
                type(i) is not int or i <= 0 for i in ids):
            raise ValueError('Нужно не менее двух уникальных положительных ID')
        self._all_ids = ids
        self._active = set(ids)
        self._events: list[ExclusionEvent] = []

    @property
    def all_ids(self) -> tuple[int, ...]:
        return self._all_ids

    @property
    def active_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._active))

    @property
    def excluded_ids(self) -> tuple[int, ...]:
        return tuple(i for i in self._all_ids if i not in self._active)

    @property
    def events(self) -> tuple[ExclusionEvent, ...]:
        return tuple(self._events)

    def exclude_many(self, participant_ids, round_id: int, evidence=()) -> None:
        ids = tuple(sorted(set(participant_ids)))
        if not ids:
            return
        if type(round_id) is not int or round_id < 0:
            raise ValueError('Неверный round_id')
        if any(i not in self._active for i in ids):
            raise ValueError('Можно исключать только активных участников')
        self._active.difference_update(ids)
        self._events.append(ExclusionEvent(round_id, ids, tuple(evidence)))
