"""Контракт UI для OHE: только данные Snapshot, без объектов Participant."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
from typing import Any

Vector = tuple[int, ...]
Coefficients = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Deal:
    sender: int
    receiver: int
    value: Vector


@dataclass(frozen=True)
class MaliciousReport:
    round_id: int
    reporter: int
    suspect: int
    is_malicious: bool
    evidence_ref: str | None = None


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
        return self.reports

    @property
    def new_opinions(self):
        return self.new_reports


@dataclass(frozen=True)
class Trace:
    n: int
    t: int
    p: int
    seed: int
    options: tuple[str, ...]
    scenario: str
    snapshots: tuple[Snapshot, ...]


def parse_trace(data: bytes | str) -> Trace:
    """JSON с индексом 0=null и векторами строго одинаковой размерности."""
    raw = json.loads(data)
    meta = raw['meta']
    n, t, p = int(meta['n']), int(meta['t']), int(meta['p'])
    options = tuple(str(name) for name in meta['options'])
    m = len(options)
    if not (2 <= t <= n <= 10 and p > n and 2 <= m <= 10):
        raise ValueError('Некорректные n, t, p или варианты')

    def vector(x):
        y = tuple(int(v) for v in x)
        if len(y) != m or any(not 0 <= v < p for v in y):
            raise ValueError('Неверная размерность или координата вектора')
        return y

    def polys(x):
        p0 = tuple(tuple(int(v) for v in coeffs) for coeffs in x)
        if len(p0) != m or any(not 2 <= len(poly) <= t for poly in p0):
            raise ValueError('Неверный формат коэффициентов')
        return p0

    def deals(items):
        return tuple(Deal(int(d['sender']), int(d['receiver']), vector(d['value'])) for d in items)

    def indexed(s, name, convert):
        arr = s[name]
        if len(arr) != n+1 or arr[0] is not None:
            raise ValueError(f'{name}: ожидается n+1 элементов; первый = null')
        return tuple(None if v is None else convert(v) for v in arr)

    def reports(items):
        return tuple(MaliciousReport(
            int(v.get('round_id', v.get('round_number'))),
            int(v.get('reporter', v.get('observer'))),
            int(v['suspect']), bool(v.get('is_malicious', v.get('accuses'))),
            v.get('evidence_ref'))
            for v in items)

    shots = []
    for snap in raw['snapshots']:
        recon = snap.get('reconstructed')
        if recon is not None:
            recon = {tuple(int(i) for i in item['subset']): vector(item['value']) for item in recon}
        shots.append(Snapshot(
            int(snap['step']), str(snap['phase']), snap.get('active'),
            str(snap.get('description', '')), deals(snap['new_deals']), deals(snap['all_deals']),
            indexed(snap, 'votes', vector), indexed(snap, 'polys', polys),
            indexed(snap, 'received', lambda d: {int(k): vector(v) for k, v in d.items()}),
            indexed(snap, 'shares', vector),
            tuple(int(i) for i in snap.get('published_senders', [])), recon,
            None if snap.get('aggregate_F') is None else polys(snap['aggregate_F']),
            dict(snap.get('checks', {})), snap.get('attack_info'),
            tuple(int(i) for i in snap.get('active_participants', ())),
            tuple(int(i) for i in snap.get('excluded_participants', ())),
            int(snap.get('round_number', 0)),
            reports(snap.get('reports', snap.get('opinions', []))),
            reports(snap.get('new_reports', snap.get('new_opinions', []))),
            tuple((int(i), int(c)) for i, c in snap.get('tallies', [])),
            tuple(int(i) for i in snap.get('new_exclusions', [])),
            None if snap.get('final_result') is None else vector(snap['final_result']),
            str(snap.get('consensus_status', 'not_started')),
            None if snap.get('current_threshold') is None else int(snap['current_threshold']),
            tuple((int(e[0]), str(e[1]), int(e[2]), None if e[3] is None else int(e[3]), int(e[4]))
                  for e in snap.get('board_events', ())),
            tuple(int(i) for i in snap.get('round_electorate', ())),
            str(snap.get('session_phase', 'registration')),
            tuple((str(e[0]), int(e[1]), None if e[2] is None else int(e[2]))
                  for e in snap.get('membership_events', ())),
        ))
    if not shots or [s.step for s in shots] != list(range(1, len(shots)+1)):
        raise ValueError('Снимки должны идти последовательно 1, 2, ...')
    return Trace(n, t, p, int(meta['seed']), options, str(meta['scenario']), tuple(shots))


def trace_to_json(trace: Trace) -> str:
    """Сохранить трассу (полную God View!) для отладки, не публиковать как приватную."""
    shots = []
    for s in trace.snapshots:
        obj = asdict(s)
        obj['reconstructed'] = (None if s.reconstructed is None else
                                [{'subset': list(k), 'value': list(v)}
                                 for k, v in s.reconstructed.items()])
        shots.append(obj)
    return json.dumps({'meta': {'n':trace.n, 't':trace.t, 'p':trace.p,
                                'seed':trace.seed, 'options':trace.options,
                                'scenario':trace.scenario},
                       'snapshots':shots}, indent=2, ensure_ascii=False)
