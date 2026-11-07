"""Безопасные проекции God/Participant/Coalition из единственного Snapshot."""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import Deal, Snapshot, Trace, MaliciousReport


@dataclass(frozen=True)
class Card:
    idx: int
    vote: tuple[int, ...] | None
    polys: tuple[tuple[int, ...], ...] | None
    received: tuple[tuple[int, tuple[int, ...]], ...]
    share: tuple[int, ...] | None
    published: bool
    can_see_private: bool


@dataclass(frozen=True)
class Frame:
    n: int
    t: int
    p: int
    options: tuple[str, ...]
    step: int
    total_steps: int
    phase: str
    active: int | None
    mode: str
    coalition: tuple[int, ...]
    cards: tuple[Card, ...]
    visible_deals: tuple[Deal, ...]
    board_deals: tuple[Deal, ...]
    event: str
    checks: dict[str, str]
    attack_info: dict[str, object] | None
    active_participants: tuple[int, ...] = ()
    excluded_participants: tuple[int, ...] = ()
    round_number: int = 0
    reports: tuple[MaliciousReport, ...] = ()
    new_reports: tuple[MaliciousReport, ...] = ()
    tallies: tuple[tuple[int, int], ...] = ()
    new_exclusions: tuple[int, ...] = ()
    final_result: tuple[int, ...] | None = None
    consensus_status: str = 'not_started'
    board_events: tuple[tuple[int, str, int, int | None, int], ...] = ()
    round_electorate: tuple[int, ...] = ()
    session_phase: str = 'registration'
    membership_events: tuple[tuple[str, int, int | None], ...] = ()

    @property
    def opinions(self):
        return self.reports


def safe_description(phase: str, active: int | None, n: int, t: int) -> str:
    return {
        'setup': f'Участники 1..{n} создали OHE-бюллетени и векторные многочлены степени ≤ {t-1}',
        'deal': f'Участник {active} рассылает вектор f_{active}(j) по закрытым каналам',
        'sum': 'Участники локально вычислили вектор s_j = Σ_i f_i(j) mod p',
        'publish': f'Участник {active} публикует вектор s_{active} на публичной доске',
        'reconstruct': f'Наблюдатель восстанавливает по {t} долям счётчики всех вариантов',
        'report': f'Участник {active} публикует MaliciousReport',
        'restart': 'Перезапуск протокола с новыми многочленами и меньшим t',
        'redeal': 'Повторная рассылка приватных долей через Board',
        'tally': 'Публичный подсчёт обвинений: порог >50% активных',
        'exclude': 'Исключение агентов, получивших строгое большинство обвинений',
        'reaggregate': 'Вычитание исключённых приватных вкладов по модулю p, новая публикация',
        'finalize': 'Завершение консенсуса и попытка утвердить результат',

    }.get(phase, 'Событие протокола')


def project(trace: Trace, snap: Snapshot, mode: str, selected: tuple[int, ...]=()) -> Frame:
    if mode not in ('god', 'participant', 'coalition'):
        raise ValueError('Неизвестный режим')
    if mode == 'participant' and len(selected) != 1:
        raise ValueError('Выберите одного участника')
    if mode == 'coalition' and not selected:
        raise ValueError('Нужны члены коалиции')
    if len(selected) != len(set(selected)) or any(not 1 <= i <= trace.n for i in selected):
        raise ValueError('Неверный список участников')
    members = set(range(1, trace.n+1)) if mode=='god' else set(selected)
    active_ids = snap.active_participants or tuple(range(1, trace.n+1))
    # Повторные публикации заменяют старые для реконструкции, но не стирают историю.
    latest = {d.sender: d for d in snap.all_deals
              if d.receiver == -1 and d.sender in active_ids}
    public = tuple(latest[i] for i in sorted(latest))
    pub_values = {d.sender: d.value for d in public}
    visible = tuple(d for d in snap.all_deals
                    if d.receiver==-1 or d.sender in members or d.receiver in members)
    cards = tuple(Card(
        idx=i,
        vote=snap.votes[i] if i in members else None,
        polys=snap.polys[i] if i in members else None,
        received=tuple(sorted(snap.received[i].items())) if i in members else (),
        share=snap.shares[i] if i in members else pub_values.get(i),
        published=i in pub_values,
        can_see_private=i in members,
    ) for i in range(1, trace.n+1))
    # Приватные инварианты, в том числе OHE, могут раскрыть голос: не показываем.
    checks = dict(snap.checks) if mode=='god' else {
        name: value if name == 'I5' else 'hidden'
        for name, value in snap.checks.items()
    }
    return Frame(trace.n, snap.current_threshold or trace.t, trace.p, trace.options, snap.step,
                 len(trace.snapshots), snap.phase, snap.active, mode,
                 tuple(sorted(selected)), cards, visible, public,
                 safe_description(snap.phase, snap.active, trace.n, trace.t),
                 checks, dict(snap.attack_info) if mode=='god' and snap.attack_info else None,
                 active_ids, snap.excluded_participants, snap.round_number,
                 snap.reports, snap.new_reports, snap.tallies,
                 snap.new_exclusions, snap.final_result, snap.consensus_status,
                 snap.board_events, snap.round_electorate,
                 snap.session_phase, snap.membership_events)
