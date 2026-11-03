"""Чистые проекции Snapshot: UI никогда не обращается к Participant."""
from __future__ import annotations
from dataclasses import dataclass
from .contracts import Deal, Snapshot, Trace


@dataclass(frozen=True)
class Card:
    idx: int
    vote: int | None
    poly: tuple[int, ...] | None
    received: tuple[tuple[int, int], ...]
    share: int | None
    published: bool
    can_see_private: bool


@dataclass(frozen=True)
class Frame:
    n: int
    t: int
    p: int
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


def safe_description(phase: str, active: int | None, n: int, t: int) -> str:
    """Описание шага без тайных значений и произвольных строк из god-снимка."""
    if phase == 'setup':
        return f'Участники 1..{n} выбрали голоса и многочлены степени ≤ {t - 1}'
    if phase == 'deal':
        return f'Участник {active} рассылает f_{active}(j) по защищённым каналам'
    if phase == 'sum':
        return 'Каждый участник локально вычислил s_j = Σ_i f_i(j) mod p'
    if phase == 'publish':
        return f'Участник {active} публикует s_{active} на доске'
    if phase == 'reconstruct':
        return f'Наблюдатель проверяет восстановление S = F(0) по любым {t} долям'
    return 'Событие протокола'


def project(trace: Trace, snap: Snapshot, mode: str,
            selected: tuple[int, ...] = ()) -> Frame:
    """Получить только разрешённые данные для всех панелей интерфейса.

    Это отладочная модель наблюдений, а не криптографическая граница доступа:
    оригинальный Snapshot по определению содержит god view.
    """
    if mode not in {'god', 'participant', 'coalition'}:
        raise ValueError('Неизвестный режим видимости')
    if mode == 'participant' and len(selected) != 1:
        raise ValueError('Нужно выбрать одного участника')
    if mode == 'coalition' and not selected:
        raise ValueError('Коалиция не должна быть пустой')
    if len(set(selected)) != len(selected) or any(i < 1 or i > trace.n for i in selected):
        raise ValueError('Некорректные участники')

    members = set(range(1, trace.n + 1)) if mode == 'god' else set(selected)
    public = tuple(d for d in snap.all_deals if d.receiver == -1)
    public_by_sender = {d.sender: d.value for d in public}
    visible = tuple(d for d in snap.all_deals
                    if d.receiver == -1 or d.sender in members or d.receiver in members)
    cards = []
    for idx in range(1, trace.n + 1):
        mine = idx in members
        recv = snap.received[idx] if mine else None
        # Чужая доля видна только после публикации, даже если в god-snapshot она уже есть.
        share = snap.shares[idx] if mine else public_by_sender.get(idx)
        cards.append(Card(
            idx=idx,
            vote=snap.votes[idx] if mine else None,
            poly=snap.polys[idx] if mine else None,
            received=tuple(sorted(recv.items())) if recv else (),
            share=share,
            published=idx in public_by_sender,
            can_see_private=mine,
        ))
    # В ограниченных режимах закрываем диагностические статусы, которые
    # вычислены по чужим приватным значениям: сами статусы могут быть побочным каналом.
    checks = snap.checks.copy() if mode == 'god' else {
        name: value if name == 'I5' else 'hidden'
        for name, value in snap.checks.items()
    }
    return Frame(
        n=trace.n, t=trace.t, p=trace.p, step=snap.step,
        total_steps=len(trace.snapshots), phase=snap.phase, active=snap.active,
        mode=mode, coalition=tuple(sorted(selected)), cards=tuple(cards),
        visible_deals=visible, board_deals=public,
        event=safe_description(snap.phase, snap.active, trace.n, trace.t),
        checks=checks,
    )
