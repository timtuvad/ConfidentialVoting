"""Контракт внешнего UI: снимки и сообщения, без импорта криптографического ядра."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json


@dataclass(frozen=True)
class Deal:
    sender: int
    receiver: int
    value: int


@dataclass(frozen=True)
class Snapshot:
    step: int
    phase: str
    active: int | None
    description: str
    new_deals: tuple[Deal, ...]
    all_deals: tuple[Deal, ...]
    votes: tuple[int | None, ...]
    polys: tuple[tuple[int, ...] | None, ...]
    received: tuple[dict[int, int] | None, ...]
    shares: tuple[int | None, ...]
    published_senders: tuple[int, ...]
    reconstructed: dict[tuple[int, ...], int] | None
    aggregate_F: tuple[int, ...] | None
    # Диагностики поступают от движка/внешнего тестера, UI их не вычисляет.
    checks: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Trace:
    n: int
    t: int
    p: int
    seed: int
    scenario: str
    snapshots: tuple[Snapshot, ...]


def _deals(items: Any) -> tuple[Deal, ...]:
    return tuple(Deal(int(d['sender']), int(d['receiver']), int(d['value'])) for d in items)


def _parse_snapshot(s: dict[str, Any], n: int) -> Snapshot:
    def indexed(name: str, convert):
        raw = s[name]
        if len(raw) != n + 1 or raw[0] is not None:
            raise ValueError(f'{name}: ожидается массив длины n+1, индекс 0 = null')
        return tuple(None if v is None else convert(v) for v in raw)

    reconstructed = s.get('reconstructed')
    if reconstructed is not None:
        # JSON-представление: [{"subset": [1, 2, 3], "value": 3}, ...]
        reconstructed = {
            tuple(int(j) for j in item['subset']): int(item['value'])
            for item in reconstructed
        }
    return Snapshot(
        step=int(s['step']), phase=str(s['phase']),
        active=None if s.get('active') is None else int(s['active']),
        description=str(s.get('description', '')),
        new_deals=_deals(s.get('new_deals', [])),
        all_deals=_deals(s.get('all_deals', [])),
        votes=indexed('votes', int),
        polys=indexed('polys', lambda v: tuple(int(a) for a in v)),
        received=indexed('received', lambda v: {int(k): int(x) for k, x in v.items()}),
        shares=indexed('shares', int),
        published_senders=tuple(int(j) for j in s.get('published_senders', [])),
        reconstructed=reconstructed,
        aggregate_F=(None if s.get('aggregate_F') is None
                     else tuple(int(a) for a in s['aggregate_F'])),
        checks={str(k): str(v) for k, v in s.get('checks', {}).items()},
    )


def trace_from_json(content: bytes | str) -> Trace:
    """Прочитать сериализованные Snapshot с проверкой структуры, но не доказать их корректность."""
    data = json.loads(content)
    meta = data['meta']
    n, t, p = int(meta['n']), int(meta['t']), int(meta['p'])
    if not (2 <= n <= 20 and 2 <= t <= n and p > n):
        raise ValueError('Ожидаются 2 ≤ t ≤ n ≤ 20 и p > n')
    snapshots = tuple(_parse_snapshot(s, n) for s in data['snapshots'])
    if not snapshots:
        raise ValueError('В JSON нет снимков')
    steps = [s.step for s in snapshots]
    if steps != list(range(1, len(snapshots) + 1)):
        raise ValueError('Снимки должны идти по шагам 1, 2, 3, ...')
    return Trace(n, t, p, int(meta.get('seed', 0)),
                 str(meta.get('scenario', 'honest')), snapshots)
