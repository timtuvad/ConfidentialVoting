"""Симуляция OHE-голосования на 2n+3 шага с тремя независимыми атаками."""
from __future__ import annotations
from dataclasses import replace
from itertools import combinations
from random import Random
from datetime import datetime
from typing import Callable

from src.core.utils import is_prime
from src.core.vector import one_hot, VectorShare, reconstruct_vector, interpolate_vector_polynomials
from src.core.board import Board, InMemoryBoard, PrivateShare, PublicShare
from src.core.voting import VotingSession, VotingSchedule
from .participant import Participant
from .models import Deal, Snapshot
from .invariants import check_snapshot
from .consensus import ConsensusCoordinator, Detector


class Simulation:
    """Выбор 0..m-1 для каждого из n участников, публичный результат из m счётчиков.

    Малicious 1 invalid_vote:  выбранная координата получает значение 3 вместо 1.
    Malicious 2 wrong_private_share: в приватном Deal одной координаты +Δ.
    Malicious 3 wrong_public_share: в опубликованной доле одной координаты +Δ.

    При замене одной точки интерполяция любых t точек возможна, но t-наборы
    могут восстановить разные результаты. Нельзя объявлять единый итог.
    """
    SCENARIOS = ('honest', 'coalition', 'invalid_vote', 'wrong_private_share', 'wrong_public_share')
    ALIASES = {'wrong_secret': 'invalid_vote', 'substitution': 'wrong_private_share'}

    def __init__(self, n: int = 5, t: int = 3, p: int = 101, seed: int = 42,
                 choices: list[int] | tuple[int, ...] | None = None,
                 options: list[str] | tuple[str, ...] = ('A', 'B', 'C'),
                 scenario: str = 'honest', attacker: int = 2, attack_delta: int = 7,
                 target_receiver: int | None = None, invalid_secret: int = 3,
                 attack_option: int | None = None,
                 enable_consensus: bool = False,
                 detector: Detector | None = None,
                 max_consensus_rounds: int = 10,
                 board: Board | None = None,
                 schedule: VotingSchedule | None = None,
                 clock: Callable[[], datetime] | None = None):
        if not (2 <= t <= n <= 10 and is_prime(p) and p > n):
            raise ValueError('Необходимо 2 ≤ t ≤ n ≤ 10, p > n, p — простое')
        options = tuple(options)
        if not (2 <= len(options) <= 10 and all(isinstance(x, str) and x.strip() for x in options)
                and len(set(options)) == len(options)):
            raise ValueError('Нужно от 2 до 10 уникальных названий вариантов')
        m = len(options)
        scenario = self.ALIASES.get(scenario, scenario)
        if scenario not in self.SCENARIOS:
            raise ValueError(f'Неизвестный сценарий {scenario}')
        if not 1 <= attacker <= n:
            raise ValueError('Атакующий вне диапазона 1..n')
        if not 1 <= attack_delta < p:
            raise ValueError('Искажение Δ должно быть в [1,p)')
        if not 2 <= invalid_secret < p:
            raise ValueError('Недопустимый секрет должен быть в диапазоне [2,p)')
        if target_receiver is None:
            target_receiver = attacker % n + 1
        if not (1 <= target_receiver <= n and target_receiver != attacker):
            raise ValueError('Получатель должен отличаться от атакующего')
        rng = Random(seed)
        if choices is None:
            choices = [rng.randrange(m) for _ in range(n)]
        if len(choices) != n or any(type(c) is not int or not 0 <= c < m for c in choices):
            raise ValueError('choices — номера вариантов в диапазоне [0,m) для всех n участников')
        if attack_option is None:
            attack_option = choices[attacker - 1]
        if type(attack_option) is not int or not 0 <= attack_option < m:
            raise ValueError('Искажённая координата за пределами списка вариантов')

        self.n, self.t, self.p, self.seed = n, t, p, seed
        self.options = options
        self.configured_choices = tuple(choices)
        self.scenario, self.attacker = scenario, attacker
        self.attack_delta = attack_delta
        self.target_receiver, self.invalid_secret = target_receiver, invalid_secret
        self.attack_option = attack_option
        ballots = [list(one_hot(choice, m)) for choice in choices]
        if scenario == 'invalid_vote':
            ballots[attacker - 1][self.configured_choices[attacker - 1]] = invalid_secret
        self.board = board if board is not None else InMemoryBoard()
        # Регистрация и начало сессии происходят ДО создания участников.
        self.session = VotingSession(t, self.board, range(1, n + 1),
                                     schedule=schedule, clock=clock)
        self.session.start()
        self.participants = [None] + [
            Participant(i, n, t, p, tuple(ballots[i-1]), rng)
            for i in range(1, n + 1)
        ]
        self.snapshots = self._run()
        self.consensus = None
        if enable_consensus:
            self.consensus = ConsensusCoordinator(self, detector, max_consensus_rounds)
            self.snapshots = self.consensus.extend(self.snapshots)

    def _run(self) -> tuple[Snapshot, ...]:
        n, t, p = self.n, self.t, self.p
        history: list[Deal] = []
        shots: list[Snapshot] = []
        shares: list[tuple[int, ...] | None] = [None] * (n + 1)
        published: list[int] = []
        incident: dict[str, object] | None = None

        if self.scenario == 'invalid_vote':
            original = self.configured_choices[self.attacker - 1]
            incident = {
                'type': 'invalid_vote', 'attacker': self.attacker,
                'option': original, 'original_vector': one_hot(original, len(self.options)),
                'actual_vector': self.participants[self.attacker].vote,
                'expected_counts': self._counts(self.configured_choices),
                'actual_counts': tuple(sum(p0.vote[k] for p0 in self.participants[1:]) % p
                                       for k in range(len(self.options))),
            }

        def capture(phase: str, active: int | None, description: str,
                    news: tuple[Deal, ...] = (),
                    reconstructed: dict[tuple[int, ...], tuple[int, ...]] | None = None,
                    aggregate: tuple[tuple[int, ...], ...] | None = None):
            s = Snapshot(
                step=len(shots) + 1, phase=phase, active=active, description=description,
                new_deals=news, all_deals=tuple(history),
                votes=tuple([None] + [p0.vote for p0 in self.participants[1:]]),
                polys=tuple([None] + [p0.coefficients for p0 in self.participants[1:]]),
                received=tuple([None] + [dict(p0.received) for p0 in self.participants[1:]]),
                shares=tuple(shares), published_senders=tuple(published),
                reconstructed=reconstructed, aggregate_F=aggregate,
                attack_info=None if incident is None else dict(incident),
                current_threshold=t,
                active_participants=self.session.active_ids,
                excluded_participants=self.session.excluded_ids,
                session_phase=self.session.phase.value,
                membership_events=tuple((e.kind, e.participant_id, e.round_id)
                                        for e in self.session.registry.events),
                board_events=tuple(
                    (e.message_id,
                     'PRIVATE_SHARE' if isinstance(e.message, PrivateShare) else 'PUBLIC_SHARE',
                     e.message.sender,
                     e.message.receiver if isinstance(e.message, PrivateShare) else None,
                     e.message.round_id)
                    for e in self.board.read_all()),
            )
            shots.append(replace(s, checks=check_snapshot(s, n, t, p)))

        capture('setup', None, f'Участники 1..{n} сформировали OHE-векторы и многочлены')
        for i in range(1, n + 1):
            new: list[Deal] = []
            for deal in self.participants[i].deal():
                if (self.scenario == 'wrong_private_share' and i == self.attacker and
                        deal.receiver == self.target_receiver):
                    altered = list(deal.value)
                    k = self.attack_option
                    altered[k] = (altered[k] + self.attack_delta) % p
                    incident = {
                        'type': self.scenario, 'attacker': i,
                        'receiver': deal.receiver, 'option': k,
                        'expected_vector': deal.value, 'sent_vector': tuple(altered),
                        'delta': self.attack_delta,
                    }
                    deal = Deal(i, deal.receiver, tuple(altered))
                # Сначала публикуем в Board, затем получатель читает по индексу.
                self.board.post_private_share(PrivateShare(0, deal.sender, deal.receiver, deal.value))
                stored = next(msg for msg in self.board.get_private_shares(deal.receiver, 0)
                              if msg.sender == deal.sender)
                self.participants[deal.receiver].receive(Deal(stored.sender, stored.receiver, stored.value))
                history.append(deal)
                new.append(deal)
            capture('deal', i, f'Участник {i} разослал доли по {len(self.options)} координатам', tuple(new))
        for j in range(1, n + 1):
            shares[j] = self.participants[j].compute_share()
        capture('sum', None, 'Все вычислили покоординатно векторы s_j')
        for j in range(1, n + 1):
            deal = self.participants[j].publish()
            if self.scenario == 'wrong_public_share' and j == self.attacker:
                altered = list(deal.value)
                k = self.attack_option
                altered[k] = (altered[k] + self.attack_delta) % p
                incident = {
                    'type': self.scenario, 'attacker': j, 'option': k,
                    'expected_vector': deal.value, 'published_vector': tuple(altered),
                    'delta': self.attack_delta,
                }
                deal = Deal(j, -1, tuple(altered))
            self.board.post_public_share(PublicShare(0, deal.sender, deal.value))
            history.append(deal)
            published.append(j)
            capture('publish', j, f'Участник {j} опубликовал вектор s_{j}', (deal,))

        points = {d.sender: VectorShare(d.sender, d.value, p)
                  for d in self.board.get_public_shares(0)}
        results = {
            ids: reconstruct_vector([points[i] for i in ids], t)
            for ids in combinations(range(1, n + 1), t)
        }
        aggregate = interpolate_vector_polynomials([points[j] for j in range(1, t + 1)], t)
        self.session.mark_reconstructed()
        capture('reconstruct', None, 'Наблюдатель восстановил счётчики каждого варианта',
                reconstructed=results, aggregate=aggregate)
        assert len(shots) == 2 * n + 3
        return tuple(shots)

    def _counts(self, choices: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sum(choice == k for choice in choices) for k in range(len(self.options)))
