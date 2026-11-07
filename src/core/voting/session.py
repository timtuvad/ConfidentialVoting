"""Жизненный цикл голосования и политика перезапуска без криптографических операций."""
from __future__ import annotations
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from ..board import Board
from ..membership import ParticipantRegistry, ParticipantAdmission, ParticipantExclusion, ExclusionDecision
from .schedule import VotingSchedule


class SessionPhase(str, Enum):
    REGISTRATION = 'registration'
    VOTING = 'voting'
    REPORTING = 'reporting'
    RECOUNT_REQUIRED = 'recount_required'
    RESTART_REQUIRED = 'restart_required'
    VERIFYING = 'verifying'
    FINALIZED = 'finalized'
    UNRESOLVED = 'unresolved'
    TERMINATED = 'terminated_too_few'


class NextAction(str, Enum):
    VERIFY = 'verify'
    RECOUNT = 'recount'
    RESTART = 'restart'
    TERMINATE = 'terminate'


@dataclass(frozen=True, slots=True)
class SessionTransition:
    round_id: int
    action: NextAction
    active_ids: tuple[int, ...]
    excluded_ids: tuple[int, ...]
    current_threshold: int
    decision: ExclusionDecision


class VotingSession:
    """Хранит статус, порог и Registry; делегирует решения ParticipantExclusion.

    Не вычисляет полиномы или итоги: RESTART означает, что внешний движок
    обязан создать НОВЫЕ многочлены при изменившемся пороге.
    """

    def __init__(self, threshold: int, board: Board,
                 participant_ids: Iterable[int] = (),
                 schedule: VotingSchedule | None = None,
                 clock: Callable[[], datetime] | None = None):
        if type(threshold) is not int or threshold < 2:
            raise ValueError('Порог должен быть целым >= 2')
        self.initial_threshold = threshold
        self.current_threshold = threshold
        self.board = board
        self.registry = ParticipantRegistry()
        self.schedule = schedule or VotingSchedule()
        self.admission = ParticipantAdmission(self.registry, self.schedule, clock)
        self.exclusion = ParticipantExclusion(self.registry, board)
        self.phase = SessionPhase.REGISTRATION
        self._last_reporting_round = 0
        self.admission.register_many(participant_ids)

    @property
    def active_ids(self) -> tuple[int, ...]:
        return self.registry.active_ids

    @property
    def excluded_ids(self) -> tuple[int, ...]:
        return self.registry.excluded_ids

    def start(self) -> None:
        if self.phase != SessionPhase.REGISTRATION:
            raise RuntimeError('Сессия уже начата')
        if not self.schedule.voting_open(self.admission.clock()):
            raise RuntimeError('Дедлайн начала голосования истёк')
        self.admission.close()
        if len(self.active_ids) < 2:
            self.phase = SessionPhase.TERMINATED
            return
        if len(self.active_ids) < self.current_threshold:
            # До старта нет старых долей; двигателю сразу нужен порог n_active.
            self.current_threshold = len(self.active_ids)
        self.phase = SessionPhase.VOTING

    def mark_reconstructed(self) -> None:
        """Завершена публикация F(j), возможна проверка и этап отчётов."""
        if self.phase != SessionPhase.VOTING:
            raise RuntimeError('Переход к проверке разрешён только из voting')
        self.phase = SessionPhase.VERIFYING

    def begin_reporting(self, round_id: int) -> None:
        if self.phase not in (SessionPhase.VOTING, SessionPhase.VERIFYING):
            raise RuntimeError(f'Нельзя начать поиск malicious из {self.phase.value}')
        if not self.schedule.reporting_open(self.admission.clock()):
            raise RuntimeError('Приём MaliciousReport закрыт по расписанию')
        if type(round_id) is not int or round_id != self._last_reporting_round + 1:
            raise ValueError('Раунды отчётов должны идти последовательно, начиная с 1')
        self._last_reporting_round = round_id
        self.phase = SessionPhase.REPORTING

    def process_reports(self, round_id: int) -> SessionTransition:
        if self.phase != SessionPhase.REPORTING or round_id != self._last_reporting_round:
            raise RuntimeError('Сначала нужно открыть соответствующий reporting-раунд')
        decision = self.exclusion.exclude_from_reports(round_id)
        n = len(self.active_ids)
        if n < 2:
            action = NextAction.TERMINATE
            self.phase = SessionPhase.TERMINATED
        elif not decision.excluded_ids:
            action = NextAction.VERIFY
            self.phase = SessionPhase.VERIFYING
        elif n < self.current_threshold:
            action = NextAction.RESTART
            self.current_threshold = n
            self.phase = SessionPhase.RESTART_REQUIRED
        else:
            action = NextAction.RECOUNT
            self.phase = SessionPhase.RECOUNT_REQUIRED
        return SessionTransition(round_id, action, self.active_ids, self.excluded_ids,
                                 self.current_threshold, decision)

    def mark_recalculated(self) -> None:
        if self.phase not in (SessionPhase.RECOUNT_REQUIRED, SessionPhase.RESTART_REQUIRED):
            raise RuntimeError('Нет ожидающего пересчёта или перезапуска')
        self.phase = SessionPhase.VERIFYING

    def finish(self, verified_result: bool) -> None:
        """Финализировать только при фактически проверенном результате."""
        if self.phase != SessionPhase.VERIFYING:
            raise RuntimeError('Сначала нужно завершить раунд обнаружения')
        self.phase = SessionPhase.FINALIZED if verified_result else SessionPhase.UNRESOLVED

    def mark_round_limit(self) -> None:
        if self.phase not in (SessionPhase.VERIFYING, SessionPhase.VOTING):
            raise RuntimeError('Лимит раундов нельзя применять в текущей фазе')
        self.phase = SessionPhase.UNRESOLVED
