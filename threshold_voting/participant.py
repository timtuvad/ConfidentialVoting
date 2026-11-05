"""Участник многовариантного голосования: локально хранит лишь свой голос."""
from __future__ import annotations
from random import Random
from src.core.vector import VectorPolynomial, VectorShare, aggregate_vector_shares
from .models import Deal


class Participant:
    def __init__(self, idx: int, n: int, t: int, p: int, vote: tuple[int, ...], rng: Random):
        if not (1 <= idx <= n and 2 <= t <= n and p > n):
            raise ValueError('Неверные параметры участника')
        self.idx, self.n, self.t, self.p = idx, n, t, p
        self.vote = tuple(vote)
        self.poly = VectorPolynomial(self.vote, t, p, rng)
        self.received: dict[int, tuple[int, ...]] = {}
        self.share: tuple[int, ...] | None = None
        self.published = False

    @property
    def coefficients(self) -> tuple[tuple[int, ...], ...]:
        return self.poly.coefficients

    def deal(self) -> tuple[Deal, ...]:
        return tuple(Deal(self.idx, j, self.poly.at(j).values)
                     for j in range(1, self.n + 1) if j != self.idx)

    def receive(self, d: Deal) -> None:
        if (d.receiver != self.idx or d.sender == self.idx or
                not 1 <= d.sender <= self.n or d.sender in self.received or
                len(d.value) != len(self.vote) or any(not 0 <= v < self.p for v in d.value)):
            raise ValueError('Некорректная приватная векторная доля')
        self.received[d.sender] = d.value

    def compute_share(self) -> tuple[int, ...]:
        if len(self.received) != self.n - 1:
            raise ValueError('Получены не все доли')
        parts = [self.poly.at(self.idx)] + [
            VectorShare(self.idx, self.received[i], self.p)
            for i in range(1, self.n + 1) if i != self.idx
        ]
        self.share = aggregate_vector_shares(parts).values
        return self.share

    def publish(self) -> Deal:
        if self.share is None:
            raise ValueError('Сначала вычислите долю суммы')
        self.published = True
        return Deal(self.idx, -1, self.share)
