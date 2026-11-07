"""OHE-протокол: независимые, сериализуемые структуры сообщений и снимков."""
from __future__ import annotations
from dataclasses import dataclass, field
from src.core.board.messages import MaliciousReport

Vector = tuple[int, ...]
Coefficients = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Deal:
    sender: int
    receiver: int  # -1 — PUBLIC_SHARE, иначе PRIVATE_SHARE
    value: Vector


# Старые клиенты могут продолжать импортировать Opinion позиционно, но
# канонический тип и название во всём новом коде — MaliciousReport.
Opinion = MaliciousReport


@dataclass(frozen=True)
class Snapshot:
    step: int
    phase: str
    active: int | None
    description: str
    new_deals: tuple[Deal, ...]
    all_deals: tuple[Deal, ...]
    votes: tuple[Vector | None, ...]
    polys: tuple[Coefficients | None, ...]
    received: tuple[dict[int, Vector] | None, ...]
    shares: tuple[Vector | None, ...]
    published_senders: tuple[int, ...]
    reconstructed: dict[tuple[int, ...], Vector] | None
    aggregate_F: Coefficients | None
    checks: dict[str, str] = field(default_factory=dict)
    attack_info: dict[str, object] | None = None
    active_participants: tuple[int, ...] = ()
    excluded_participants: tuple[int, ...] = ()
    round_number: int = 0
    reports: tuple[MaliciousReport, ...] = ()
    new_reports: tuple[MaliciousReport, ...] = ()
    tallies: tuple[tuple[int, int], ...] = ()
    new_exclusions: tuple[int, ...] = ()
    final_result: Vector | None = None
    consensus_status: str = 'not_started'
    current_threshold: int | None = None
    board_events: tuple[tuple[int, str, int, int | None, int], ...] = ()
    round_electorate: tuple[int, ...] = ()
    session_phase: str = 'registration'
    membership_events: tuple[tuple[str, int, int | None], ...] = ()

    @property
    def opinions(self):
        """Совместимость с прежним UI; приоритетное имя: reports."""
        return self.reports

    @property
    def new_opinions(self):
        return self.new_reports
