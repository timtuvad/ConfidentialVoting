"""Этап 3: MaliciousReport -> строгое большинство -> исключение -> пересчёт.

Доска *только* хранит сообщения. Детектор подставляется внешним callback.
При n_active < t полная перераздача по новому порогу t_new=n_active.
При n_active < 2 протокол безусловно прекращается без результата.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import combinations
from random import Random
from typing import Callable, Mapping, Protocol

from src.core.board import PrivateShare, PublicShare, MaliciousReport
from src.core.membership import count_reports
from src.core.voting import NextAction
from src.core.vector import VectorShare, reconstruct_vector, interpolate_vector_polynomials, aggregate_vector_shares
from .models import Deal, Snapshot
from .participant import Participant


@dataclass(frozen=True)
class Evidence:
    """Данные, доступные непосредственно наблюдателю и из публичной доски."""
    round_number: int
    observer: int
    active_participants: tuple[int, ...]
    my_vote: tuple[int, ...]
    my_polynomials: tuple[tuple[int, ...], ...]
    received: dict[int, tuple[int, ...]]
    my_share: tuple[int, ...]
    public_shares: dict[int, tuple[int, ...]]
    threshold: int = 3


class Detector(Protocol):
    def __call__(self, observer: int, suspect: int, evidence: Evidence) -> bool: ...


def majority_exclusions(active: tuple[int, ...], reports: tuple[MaliciousReport, ...],
                        round_number: int) -> tuple[tuple[int, ...], dict[int, int]]:
    """Совместимый API; подсчёт делегируется модулю membership."""
    decision = count_reports(active, reports, round_number)
    return decision.excluded_ids, decision.votes



def consistent_result(reconstructed: dict[tuple[int, ...], tuple[int, ...]] | None,
                      participant_count: int, p: int) -> tuple[int, ...] | None:
    """Проверка согласованности и диапазона результата НЕ заменяет ZKP бюллетеней."""
    if not reconstructed:
        return None
    values = set(reconstructed.values())
    if len(values) != 1:
        return None
    result = next(iter(values))
    if any(not 0 <= c <= participant_count for c in result) or sum(result) != participant_count:
        return None
    return result


def demo_oracle(attacker: int) -> Detector:
    """ТОЛЬКО демонстрация: искусственно известный атакующий, не детектор."""
    return lambda observer, suspect, evidence: suspect == attacker


def manual_detector(accusations: Mapping[int, tuple[int, ...]]) -> Detector:
    """Ручные обвинения для проверки механики консенсуса."""
    return lambda observer, suspect, evidence: suspect in accusations.get(observer, ())


class ConsensusCoordinator:
    """Мультираундовый координатор, работающий только с общим Board API.

    При обычном исключении: пересчёт из оставшихся фактически полученных f_i(j).
    При n_active<t: НОВЫЕ независимые многочлены степени t_new-1 и новая раздача.
    Это меняет параметр порога; без дополнительных оговорок уровень защиты иной.
    """
    def __init__(self, sim, detector: Detector | None = None, max_rounds: int = 10):
        if max_rounds < 1:
            raise ValueError('max_rounds должен быть >= 1')
        self.sim = sim
        self.detector = detector
        self.max_rounds = max_rounds
        self.board = sim.board  # одна доска для f_i(j), F(j), MaliciousReport
        self.session = sim.session

    def extend(self, base: tuple[Snapshot, ...]) -> tuple[Snapshot, ...]:
        if not base:
            raise ValueError('Исходная трасса не должна быть пустой')
        sim, board = self.sim, self.board
        p = sim.p
        original_n = sim.n
        snapshots = list(base)
        base_last = base[-1]
        history = list(base_last.all_deals)
        active = self.session.active_ids
        excluded: set[int] = set(self.session.excluded_ids)
        shares: list[tuple[int, ...] | None] = list(base_last.shares)
        current_t = self.session.current_threshold
        private_round = 0   # номер эпохи, когда передавались приватные доли
        public_round = 0    # номер последней актуальной публикации F(j)
        reconstructed = base_last.reconstructed
        aggregate = base_last.aggregate_F
        tallies: dict[int, int] = {}
        round_electorate = active

        def board_events() -> tuple[tuple[int, str, int, int | None, int], ...]:
            rows = []
            for entry in board.read_all():
                msg = entry.message
                if isinstance(msg, PrivateShare):
                    rows.append((entry.message_id, 'PRIVATE_SHARE', msg.sender, msg.receiver, msg.round_id))
                elif isinstance(msg, PublicShare):
                    rows.append((entry.message_id, 'PUBLIC_SHARE', msg.sender, None, msg.round_id))
                else:
                    rows.append((entry.message_id, 'MALICIOUS_REPORT', msg.reporter, msg.suspect, msg.round_id))
            return tuple(rows)

        def capture(phase: str, desc: str, *, actor: int | None = None,
                    round_number: int = 0, new_deals: tuple[Deal, ...] = (),
                    new_reports: tuple[MaliciousReport, ...] = (),
                    newly_excluded: tuple[int, ...] = (),
                    status: str = 'voting', final_result: tuple[int, ...] | None = None):
            reports = tuple(entry.message for entry in board.read_all()
                            if isinstance(entry.message, MaliciousReport))
            checks = dict(base_last.checks)
            checks['CONSENSUS'] = status
            if reconstructed is not None:
                vals = set(reconstructed.values())
                checks['I5'] = 'ok' if vals and len(vals) == 1 else 'fail'
            # При смене порога новые полиномы имеют другую степень — не переиспользуем I2
            if current_t != sim.t:
                checks['I2'] = 'pending'
            snapshots.append(replace(base_last,
                step=len(snapshots) + 1, phase=phase, active=actor,
                description=desc, new_deals=new_deals, all_deals=tuple(history),
                votes=(None, *(sim.participants[i].vote for i in range(1, original_n + 1))),
                polys=(None, *(sim.participants[i].coefficients for i in range(1, original_n + 1))),
                received=(None, *(dict(sim.participants[i].received) for i in range(1, original_n + 1))),
                shares=tuple(shares), published_senders=active,
                reconstructed=reconstructed, aggregate_F=aggregate,
                active_participants=active, excluded_participants=tuple(sorted(excluded)),
                round_number=round_number, reports=reports, new_reports=new_reports,
                tallies=tuple(sorted(tallies.items())), new_exclusions=newly_excluded,
                final_result=final_result, consensus_status=status,
                current_threshold=current_t, board_events=board_events(),
                round_electorate=round_electorate, checks=checks,
                session_phase=self.session.phase.value,
                membership_events=tuple((e.kind, e.participant_id, e.round_id)
                                        for e in self.session.registry.events)))

        def reconstruct_active(pub_round: int) -> None:
            nonlocal reconstructed, aggregate
            records = {m.sender: m.value for m in board.get_public_shares(pub_round)
                       if m.sender in active}
            if len(records) != len(active):
                raise RuntimeError('Не все активные участники опубликовали F(j)')
            points = {i: VectorShare(i, records[i], p) for i in active}
            reconstructed = {
                group: reconstruct_vector([points[i] for i in group], current_t)
                for group in combinations(active, current_t)
            }
            if len(set(reconstructed.values())) == 1:
                aggregate = interpolate_vector_polynomials(
                    [points[i] for i in active[:current_t]], current_t)
            else:
                aggregate = None

        for round_number in range(1, self.max_rounds + 1):
            self.session.begin_reporting(round_number)
            round_electorate = self.session.active_ids
            active = round_electorate
            tallies = {}
            public_records = {m.sender: m.value for m in board.get_public_shares(public_round)
                              if m.sender in active}
            for observer in active:
                person = sim.participants[observer]
                delivered = {m.sender: m.value for m in board.get_private_shares(observer, private_round)
                             if m.sender in active}
                if any(not isinstance(v, tuple) for v in delivered.values()):
                    raise RuntimeError('В этой версии нет слоя дешифрования ciphertext')
                evidence = Evidence(round_number, observer, active, person.vote,
                                    person.coefficients, delivered, shares[observer],
                                    dict(public_records), current_t)
                fresh_reports: list[MaliciousReport] = []
                for suspect in active:
                    if suspect == observer:
                        continue
                    decision = bool(self.detector(observer, suspect, evidence)) if self.detector else False
                    msg = MaliciousReport(round_number, observer, suspect, decision)
                    board.post_malicious_report(msg)
                    fresh_reports.append(msg)
                capture('report', f'P{observer} опубликовал MaliciousReport', actor=observer,
                        round_number=round_number, new_reports=tuple(fresh_reports))

            transition = self.session.process_reports(round_number)
            exclusions = transition.decision.excluded_ids
            tallies = transition.decision.votes
            needed = transition.decision.required_votes
            capture('tally', f'Раунд {round_number}: для исключения нужно {needed} из {len(active)}',
                    round_number=round_number)

            if transition.action == NextAction.VERIFY:
                result = consistent_result(reconstructed, len(active), p)
                self.session.finish(result is not None)
                state = 'finalized' if result is not None else 'unresolved'
                capture('finalize', 'Готовый итог' if result is not None
                        else 'Нет исключений, но итог не подтверждён',
                        round_number=round_number, status=state, final_result=result)
                return tuple(snapshots)

            excluded = set(transition.excluded_ids)
            active = transition.active_ids
            reconstructed, aggregate = None, None
            for i in exclusions:
                shares[i] = None
            capture('exclude', 'Исключены: ' + ', '.join(map(str, exclusions)),
                    round_number=round_number, newly_excluded=exclusions, status='excluded')

            if transition.action == NextAction.TERMINATE:
                capture('finalize', f'Осталось {len(active)} < 2: принудительное завершение без итога',
                        round_number=round_number, status='terminated_too_few')
                return tuple(snapshots)

            if transition.action == NextAction.RESTART:
                # Понижение порога делает старые доли непригодными: ПОЛНЫЙ ПЕРЕЗАПУСК.
                previous_t = current_t
                current_t = transition.current_threshold
                private_round = round_number
                rng = Random(sim.seed + 1_000_003 * round_number + len(active))
                for i in active:
                    old = sim.participants[i]
                    sim.participants[i] = Participant(
                        i, original_n, current_t, p, old.vote, rng, active_ids=active)
                    shares[i] = None
                capture('restart', f'Активных {len(active)} < t={previous_t}; '
                        f'полный перезапуск с новым t={current_t} и свежими многочленами',
                        round_number=round_number, status='restarted')
                generated: list[Deal] = []
                attack_target = (sim.target_receiver if sim.target_receiver in active
                                 else next((j for j in active if j != sim.attacker), None))
                for i in active:
                    for item in sim.participants[i].deal():
                        if (sim.scenario == 'wrong_private_share' and i == sim.attacker
                                and item.receiver == attack_target):
                            adjusted = list(item.value)
                            adjusted[sim.attack_option] = (
                                adjusted[sim.attack_option] + sim.attack_delta) % p
                            item = Deal(item.sender, item.receiver, tuple(adjusted))
                        board.post_private_share(PrivateShare(round_number,
                                               item.sender, item.receiver, item.value))
                        generated.append(item)
                        history.append(item)
                for j in active:
                    for item in board.get_private_shares(j, round_number):
                        if item.sender in active:
                            sim.participants[j].receive(Deal(item.sender, j, item.value))
                capture('redeal', 'Новые приватные доли разосланы через Board',
                        round_number=round_number, new_deals=tuple(generated),
                        status='restarted')
                for j in active:
                    shares[j] = sim.participants[j].compute_share()
            else:
                # После обычного исключения повторно используем прежние приватные доли.
                # Каждая s_j = sum_{i active} f_i(j) mod p; XOR неверен.
                for j in active:
                    person = sim.participants[j]
                    received = {m.sender: m.value
                                for m in board.get_private_shares(j, private_round)
                                if m.sender in active}
                    parts = [person.poly.at(j)] + [VectorShare(j, received[i], p)
                                                   for i in active if i != j]
                    shares[j] = aggregate_vector_shares(parts).values

            fresh_public: list[Deal] = []
            for j in active:
                value = shares[j]
                if sim.scenario == 'wrong_public_share' and j == sim.attacker:
                    adjusted = list(value)
                    adjusted[sim.attack_option] = (
                        adjusted[sim.attack_option] + sim.attack_delta) % p
                    value = tuple(adjusted)
                msg = PublicShare(round_number, j, value)
                board.post_public_share(msg)
                item = Deal(j, -1, value)
                fresh_public.append(item)
                history.append(item)
            public_round = round_number
            reconstruct_active(public_round)
            self.session.mark_recalculated()
            capture('reaggregate', 'Новые векторы F(j) опубликованы; реконструкция пересчитана',
                    round_number=round_number, new_deals=tuple(fresh_public),
                    status='reaggregated')

        self.session.mark_round_limit()
        capture('finalize', f'Достигнут предел {self.max_rounds} раундов: итог не утверждён',
                round_number=self.max_rounds, status='max_rounds')
        return tuple(snapshots)
