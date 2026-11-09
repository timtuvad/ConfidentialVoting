"""Двухэтапный OHE/Pedersen-VSS-протокол с публичными жалобами и исключением.

Использует новую криптографию feature/PedersenVSS. Board — лишь хранилище.
Сценарии malicious предназначены только для отладки. Подписи и шифрование
пока отсутствуют; локальная имитация сравнивает открытые VerifiableShare.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from src.core.algebra.field import F_pElement
from src.core.environment import CryptographyEnvironment
from src.core.verifiable_share import VerifiableShare
from src.core.board import (Board, InMemoryBoard, PrivateShare,
                            PublicCommitments, ParticipantTotalShare, MaliciousReport, ReportKind)
from src.core.membership import ParticipantRegistry, VSSAdjudicator, Verdict
from src.core.vector import VectorShamirCryptography, one_hot


@dataclass(frozen=True, slots=True)
class RoundAudit:
    round_id: int
    threshold: int
    active_before: tuple[int, ...]
    excluded_after_round: tuple[int, ...]
    verdicts: tuple[Verdict, ...]
    phase: str


@dataclass(frozen=True, slots=True)
class ElectionResult:
    status: str                     # complete | terminated_too_few | max_rounds
    counts: tuple[int, ...] | None
    active_ids: tuple[int, ...]
    excluded_ids: tuple[int, ...]
    threshold: int
    rounds: tuple[RoundAudit, ...]


class VSSVotingProtocol:
    """Протокольная оркестрация; любое VSS-вычисление делегируется ядру.

    Нумерация участников сохраняется при исключении. При 2 <= active < t
    создаются новые полиномы и окружение с новым t; если active < 2,
    голосование останавливается без результата.
    """
    SCENARIOS = ('honest', 'invalid_vote', 'wrong_private_share',
                 'wrong_total_share', 'false_complaint_changed',
                 'false_complaint_valid')

    def __init__(self, choices: Sequence[int], variants: int, t: int,
                 p: int = 23, board: Board | None = None,
                 scenario: str = 'honest', attacker: int = 2,
                 receiver: int | None = None, delta: int = 7,
                 max_rounds: int = 10):
        n = len(choices)
        if not 2 <= n <= 10 or not 2 <= t <= n:
            raise ValueError('Требуется 2 <= t <= n <= 10')
        if type(variants) is not int or not 2 <= variants <= 10:
            raise ValueError('Требуется 2..10 вариантов')
        if any(type(c) is not int or not 0 <= c < variants for c in choices):
            raise ValueError('Неверные номера вариантов')
        scenario = {'wrong_public_share': 'wrong_total_share'}.get(scenario, scenario)
        if scenario not in self.SCENARIOS:
            raise ValueError('Неизвестный malicious-сценарий')
        if not 1 <= attacker <= n:
            raise ValueError('Неверный attacker')
        if not 1 <= delta < p:
            raise ValueError('Неверный delta')
        receiver = attacker % n + 1 if receiver is None else receiver
        if not 1 <= receiver <= n or receiver == attacker:
            raise ValueError('receiver должен отличаться от attacker')
        if max_rounds < 1:
            raise ValueError('max_rounds должен быть положительным')
        # Среда проверит, что p простое и 2p+1 тоже простое (101 НЕ подходит).
        self.environment = CryptographyEnvironment.random(p, t)
        if p <= n:  # не допускаем переполнения результатов OHE по mod p
            raise ValueError('Должно быть p > n')
        self.p = p
        self.t = t
        self.variants = variants
        self.registry = ParticipantRegistry(range(1, n + 1))
        self.board = InMemoryBoard() if board is None else board
        self.scenario, self.attacker, self.receiver, self.delta = (
            scenario, attacker, receiver, delta)
        self.max_rounds = max_rounds
        self.ballots = {i: list(one_hot(choice, variants))
                        for i, choice in enumerate(choices, 1)}
        if scenario == 'invalid_vote':
            self.ballots[attacker][choices[attacker-1]] = 3
        self.actors: dict[int, VectorShamirCryptography] = {}
        self._create_actors()

    def _create_actors(self) -> None:
        self.actors = {i: VectorShamirCryptography(self.environment, i,
                                                   self.ballots[i])
                       for i in self.registry.active_ids}

    def _verifier(self, share, comm) -> bool:
        # Проверка Pedersen VSS зависит от environment, а не от секрета
        # проверяющего. Берём экземпляр любого текущего участника.
        return next(iter(self.actors.values())).coordinates[0].verify(share, comm)

    @staticmethod
    def _sum_commitments(commitments):
        from src.core.shamir import ShamirCryptography
        return ShamirCryptography.commitments_of_total_sum(commitments)

    def _adjudicator(self) -> VSSAdjudicator:
        return VSSAdjudicator(self.board, self._verifier, self._sum_commitments)

    def _alter(self, bundle: tuple[VerifiableShare, ...], k: int = 0
               ) -> tuple[VerifiableShare, ...]:
        result = list(bundle)
        old = result[k]
        result[k] = VerifiableShare(old.x,
                                 F_pElement(self.p, (old.value.value + self.delta) % self.p),
                                 old.blinding_share)
        return tuple(result)

    def _round(self, round_id: int) -> tuple[str, tuple[Verdict, ...], tuple[int, ...] | None]:
        active = self.registry.active_ids
        ids = set(active)
        original_ids: dict[tuple[int, int], int] = {}
        # Зафиксировать права чтения s_j для участников именно этого раунда.
        self.board.open_round(round_id, active)

        # ЭТАП 1: обязательства и приватные VSS-доли на общую Board.
        for i in active:
            self.board.post_commitments(PublicCommitments(round_id, i,
                                                          self.actors[i].commitments))
        for i in active:
            for j in active:
                if i == j:
                    continue
                value = self.actors[i].verifiable_share_to(j)
                if (round_id == 0 and self.scenario == 'wrong_private_share'
                        and i == self.attacker and j == self.receiver):
                    value = self._alter(value)
                entry = self.board.post_private_share(PrivateShare(round_id, i, j, value))
                original_ids[(i, j)] = entry.message_id

        commitments = {c.sender: c.value for c in self.board.get_commitments(round_id)}
        for receiver in active:
            for msg in self.board.get_private_shares(receiver, round_id):
                if msg.sender not in ids:
                    continue
                if not self.actors[receiver].verify(msg.value, commitments[msg.sender]):
                    self.board.post_malicious_report(MaliciousReport(
                        round_id, receiver, msg.sender, ReportKind.PRIVATE_SHARE,
                        original_ids[(msg.sender, receiver)], msg.value))

        # Отдельный тест ложной жалобы: изменённое раскрытие / корректная доля.
        if round_id == 0 and self.scenario.startswith('false_complaint'):
            reporter = self.attacker
            suspect = next(i for i in active if i != reporter)
            actual = next(m for m in self.board.get_private_shares(reporter, round_id)
                          if m.sender == suspect)
            alleged = (self._alter(actual.value) if self.scenario == 'false_complaint_changed'
                       else actual.value)
            self.board.post_malicious_report(MaliciousReport(
                round_id, reporter, suspect, ReportKind.PRIVATE_SHARE,
                original_ids[(suspect, reporter)], alleged))

        adjudicator = self._adjudicator()
        # Независимо повторяем разбор жалоб каждым участником. При общих
        # проверяемых данных честные участники должны получить один verdict.
        private_reports = tuple(r for r in self.board.get_malicious_reports(round_id)
                                if r.kind == ReportKind.PRIVATE_SHARE)
        for report in private_reports:
            expected = adjudicator.resolve(report, active, active[0])
            for observer in active:
                judge = VSSAdjudicator(self.board,
                    self.actors[observer].coordinates[0].verify,
                    self._sum_commitments)
                actual = judge.resolve(report, active, observer)
                if actual != expected:
                    raise RuntimeError('Участники получили разные verdict по одной жалобе')
        private_verdicts = adjudicator.apply_reports(self.registry, round_id, active)
        if self.registry.active_ids != active:
            return 'private_exclusion', private_verdicts, None

        # ЭТАП 2: только доли НЕИСКЛЮЧЁННЫХ дилеров, включая свою локальную долю.
        # Для каждого j суммируем VerifiableShare по каждой OHE координате.
        total_ids: dict[int, int] = {}
        for j in active:
            private_by_sender = {s.sender: s.value
                                 for s in self.board.get_private_shares(j, round_id)}
            bundles = [(self.actors[j].verifiable_share_to(j) if i == j
                        else private_by_sender[i]) for i in active]
            total = VectorShamirCryptography.total_sum_at_participant(bundles)
            if (round_id == 0 and self.scenario == 'wrong_total_share'
                    and j == self.attacker):
                total = self._alter(total)
            entry = self.board.post_total_share(ParticipantTotalShare(round_id, j, total))
            total_ids[j] = entry.message_id

        # Каждый наблюдатель НЕЗАВИСИМО проверяет публичную F(j).
        # Сумма commitments общая, алгоритм verify один и тот же.
        total_comm = VectorShamirCryptography.commitments_of_total_sum(
            [commitments[i] for i in active])
        for observer in active:
            # Только голосующий получает опубликованные агрегированные доли.
            for msg in self.board.get_total_shares(observer, round_id):
                if observer == msg.sender:
                    continue
                if not self.actors[observer].verify(msg.value, total_comm):
                    self.board.post_malicious_report(MaliciousReport(
                        round_id, observer, msg.sender, ReportKind.TOTAL_SHARE,
                        total_ids[msg.sender]))

        # Только MaliciousReport на ошибочную суммарную долю s_j=F(j).
        total_verdicts = tuple(
            adjudicator.resolve(report, active, active[0])
            for report in self.board.get_malicious_reports(round_id)
            if report.kind == ReportKind.TOTAL_SHARE
        )
        bad = tuple(sorted({v.guilty_id for v in total_verdicts
                            if v.guilty_id is not None and v.guilty_id in ids}))
        if bad:
            self.registry.exclude_many(bad, round_id,
                                       evidence=[v.reason for v in total_verdicts
                                                 if v.guilty_id in bad])
            return 'total_exclusion', private_verdicts + total_verdicts, None

        results = []
        for observer in active:
            # У каждого честного участника получаются одинаковые результаты
            # из тех же долей, доступных участникам его раунда.
            received = {m.sender: m.value
                        for m in self.board.get_total_shares(observer, round_id)}
            results.append(VectorShamirCryptography.sum_of_secrets(
                [received[i] for i in active], self.t))
        if len(set(results)) != 1:
            raise RuntimeError("Разные результаты между голосующими")
        result = results[0]
        return 'complete', private_verdicts + total_verdicts, result

    def run(self) -> ElectionResult:
        audits: list[RoundAudit] = []
        for round_id in range(self.max_rounds):
            active_before = self.registry.active_ids
            if len(active_before) < 2:
                return self._result('terminated_too_few', None, audits)
            # Если участников осталось меньше порога, перезапуск с новым t.
            if len(active_before) < self.t:
                self.t = len(active_before)
                self.environment = CryptographyEnvironment.random(self.p, self.t)
                self._create_actors()
            phase, verdicts, counts = self._round(round_id)
            audits.append(RoundAudit(round_id, self.t, active_before,
                                     self.registry.excluded_ids, verdicts, phase))
            if phase == 'complete':
                return self._result('complete', counts, audits)
            if len(self.registry.active_ids) < 2:
                return self._result('terminated_too_few', None, audits)
        return self._result('max_rounds', None, audits)

    def _result(self, status, counts, audits):
        return ElectionResult(status, counts, self.registry.active_ids,
                              self.registry.excluded_ids, self.t, tuple(audits))
