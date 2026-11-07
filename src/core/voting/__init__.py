"""Границы сессии и будущих временных ограничений."""
from .schedule import VotingSchedule

__all__ = ['VotingSchedule', 'VotingSession', 'SessionPhase', 'SessionTransition', 'NextAction']


def __getattr__(name: str):
    # Ленивая загрузка разрывает цикл: admission -> voting.schedule -> voting.__init__.
    if name in ('VotingSession', 'SessionPhase', 'SessionTransition', 'NextAction'):
        from .session import VotingSession, SessionPhase, SessionTransition, NextAction
        return {'VotingSession': VotingSession, 'SessionPhase': SessionPhase,
                'SessionTransition': SessionTransition, 'NextAction': NextAction}[name]
    raise AttributeError(name)
