"""Источники данных внешнего интерфейса: JSON или будущая Simulation."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .contracts import Trace, trace_from_json
import json


DEMO_PATH = Path(__file__).with_name('sample_trace.json')


def load_demo() -> Trace:
    return trace_from_json(DEMO_PATH.read_bytes())


def load_backend(n: int, t: int, p: int, seed: int,
                 votes: tuple[int, ...], scenario: str = 'honest') -> Trace:
    """Единственная точка интеграции: берёт Snapshot из внешнего движка.

    Протокол ещё не реализован в текущем репозитории. Когда появится
    threshold_voting.simulation.Simulation, адаптер сможет читать snapshots.
    Для substitution нужен поддерживающий его движок.
    """
    import sys
    import inspect
    # Если приложение запущено из подпапки voting_ui_layer, найти пакет из корня репозитория.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from threshold_voting.simulation import Simulation  # импорт только в адаптере

    args: dict[str, Any] = dict(n=n, t=t, p=p, seed=seed, votes=list(votes))
    if scenario != 'honest':
        signature = inspect.signature(Simulation)
        if 'scenario' not in signature.parameters:
            raise NotImplementedError('Simulation пока не принимает scenario')
        args['scenario'] = scenario
    sim = Simulation(**args)
    serialized = []
    for snap in sim.snapshots:
        rec = getattr(snap, 'reconstructed', None)
        serialized.append({
            'step': snap.step, 'phase': snap.phase, 'active': snap.active,
            'description': snap.description,
            'new_deals': [{'sender': d.sender, 'receiver': d.receiver, 'value': d.value} for d in snap.new_deals],
            'all_deals': [{'sender': d.sender, 'receiver': d.receiver, 'value': d.value} for d in snap.all_deals],
            'votes': list(snap.votes),
            'polys': [list(p0) if p0 is not None else None for p0 in snap.polys],
            'received': [dict(v) if v is not None else None for v in snap.received],
            'shares': list(snap.shares),
            'published_senders': list(snap.published_senders),
            'reconstructed': ([{'subset': list(k), 'value': v} for k, v in rec.items()]
                              if rec is not None else None),
            'aggregate_F': list(snap.aggregate_F) if snap.aggregate_F is not None else None,
            # Проверки должны приходить из модуля invariants и быть сформированы
            # без секретных подробностей: UI намеренно ничего не вычисляет здесь.
        })
    return trace_from_json(json.dumps({
        'meta': {'n': n, 't': t, 'p': p, 'seed': seed, 'scenario': scenario},
        'snapshots': serialized,
    }))
