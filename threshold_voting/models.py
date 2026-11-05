"""Векторные сообщения и снимки. Индекс участников 1..n, позиция 0 = None."""
from __future__ import annotations
from dataclasses import dataclass, field

Vector = tuple[int, ...]
Coefficients = tuple[tuple[int, ...], ...]  # [option][degree]


@dataclass(frozen=True)
class Deal:
    sender: int
    receiver: int   # -1 — публикация на BulletinBoard
    value: Vector


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
