"""Единственный адаптер между настоящим движком и чистыми Snapshot в UI."""
from __future__ import annotations
from pathlib import Path
import json
from .contracts import Trace, parse_trace


def load_backend(n: int, t: int, p: int, seed: int, choices: tuple[int, ...],
                 options: tuple[str, ...], scenario: str='honest', attacker: int=2,
                 attack_delta: int=7, target_receiver: int | None=None,
                 invalid_secret: int=3, attack_option: int | None=None) -> Trace:
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from threshold_voting.simulation import Simulation
    sim = Simulation(n=n, t=t, p=p, seed=seed, choices=choices, options=options,
                     scenario=scenario, attacker=attacker, attack_delta=attack_delta,
                     target_receiver=target_receiver, invalid_secret=invalid_secret,
                     attack_option=attack_option)
    # Контракт UI сериализуем независимо от классов Participant/VectorPolynomial.
    items = []
    for snap in sim.snapshots:
        items.append({
            'step': snap.step, 'phase': snap.phase, 'active': snap.active,
            'description': snap.description,
            'new_deals': [dict(sender=d.sender, receiver=d.receiver, value=d.value)
                          for d in snap.new_deals],
            'all_deals': [dict(sender=d.sender, receiver=d.receiver, value=d.value)
                          for d in snap.all_deals],
            'votes': snap.votes, 'polys': snap.polys, 'received': snap.received,
            'shares': snap.shares, 'published_senders': snap.published_senders,
            'reconstructed': (None if snap.reconstructed is None else
                              [{'subset': k, 'value': v} for k, v in snap.reconstructed.items()]),
            'aggregate_F': snap.aggregate_F, 'checks': snap.checks,
            'attack_info': snap.attack_info,
        })
    return parse_trace(json.dumps({'meta': {'n':n,'t':t,'p':p,'seed':seed,
                                           'options':options,'scenario':sim.scenario},
                                   'snapshots':items}))
