"""Единственный адаптер между настоящим движком и чистыми Snapshot в UI."""
from __future__ import annotations
from pathlib import Path
import json
from .contracts import Trace, parse_trace


def load_backend(n: int, t: int, p: int, seed: int, choices: tuple[int, ...],
                 options: tuple[str, ...], scenario: str='honest', attacker: int=2,
                 attack_delta: int=7, target_receiver: int | None=None,
                 invalid_secret: int=3, attack_option: int | None=None,
                 enable_consensus: bool=False,
                 detector_mode: str='none',
                 manual_accusations: dict[int, tuple[int,...]] | None=None,
                 max_consensus_rounds: int=10) -> Trace:
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from threshold_voting.simulation import Simulation
    from threshold_voting.consensus import demo_oracle, manual_detector
    if detector_mode not in ('none', 'demo_oracle', 'manual', 'external'):
        raise ValueError('Неизвестный режим детектора')
    detector = (demo_oracle(attacker) if detector_mode == 'demo_oracle' and scenario not in ('honest', 'coalition')
                else manual_detector(manual_accusations or {}) if detector_mode == 'manual'
                else None)
    if detector_mode == 'external':
        from threshold_voting.detector_adapter import detect
        detector = detect
    sim = Simulation(n=n, t=t, p=p, seed=seed, choices=choices, options=options,
                     scenario=scenario, attacker=attacker, attack_delta=attack_delta,
                     target_receiver=target_receiver, invalid_secret=invalid_secret,
                     attack_option=attack_option, enable_consensus=enable_consensus,
                     detector=detector, max_consensus_rounds=max_consensus_rounds)
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
            'active_participants': snap.active_participants,
            'excluded_participants': snap.excluded_participants,
            'round_number': snap.round_number,
            'reports': [dict(round_id=v.round_id, reporter=v.reporter,
                             suspect=v.suspect, is_malicious=v.is_malicious,
                             evidence_ref=v.evidence_ref) for v in snap.reports],
            'new_reports': [dict(round_id=v.round_id, reporter=v.reporter,
                                 suspect=v.suspect, is_malicious=v.is_malicious,
                                 evidence_ref=v.evidence_ref) for v in snap.new_reports],
            'tallies': snap.tallies,
            'new_exclusions': snap.new_exclusions,
            'final_result': snap.final_result,
            'consensus_status': snap.consensus_status,
            'current_threshold': snap.current_threshold,
            'board_events': snap.board_events,
            'round_electorate': snap.round_electorate,
            'session_phase': snap.session_phase,
            'membership_events': snap.membership_events,
        })
    return parse_trace(json.dumps({'meta': {'n':n,'t':t,'p':p,'seed':seed,
                                           'options':options,'scenario':sim.scenario},
                                   'snapshots':items}))
