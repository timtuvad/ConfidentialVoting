"""Векторная версия разделения секрета Шамира (OHE) поверх скалярного ядра.

Каждая координата one-hot бюллетеня разделяется НЕЗАВИСИМО:
    v_i = (0, ..., 1, ..., 0)
    f_i,k(x) = v_i,k + sum_{a=1}^{t-1} a_i,k,a * x**a
    F_k(x) = sum_i f_i,k(x), S_k = F_k(0).

Это расширение сохраняет скалярные F_pElement, F_pPolynomial, Share,
ShamirCryptography, LagrangeInterpolation без изменения их API.
"""
from __future__ import annotations
from dataclasses import dataclass
from random import Random
from typing import Sequence

from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial
from src.core.share import Share
from src.core.shamir import ShamirCryptography
from src.core.algebra.interpolation import LagrangeInterpolation


def one_hot(choice: int, variants: int) -> tuple[int, ...]:
    """Кодирование выбранного варианта (нумерация 0..variants-1)."""
    if type(variants) is not int or variants < 2:
        raise ValueError('Нужно минимум два варианта')
    if type(choice) is not int or not 0 <= choice < variants:
        raise ValueError('Номер варианта вне диапазона')
    return tuple(int(k == choice) for k in range(variants))


def is_one_hot(vector: Sequence[int]) -> bool:
    """Проверка допустимости вектора: сумма 1 и все координаты 0 или 1."""
    return len(vector) >= 2 and all(type(x) is int and x in (0, 1) for x in vector) and sum(vector) == 1


@dataclass(frozen=True)
class VectorShare:
    """Доля (x, (y_0,...,y_{m-1})) в поле F_p."""
    x: int
    values: tuple[int, ...]
    p: int

    def __post_init__(self):
        if not (1 <= self.x < self.p):
            raise ValueError('x должен быть в диапазоне [1, p)')
        if len(self.values) < 2 or any(type(v) is not int or not 0 <= v < self.p for v in self.values):
            raise ValueError('Некорректный вектор доли')


class VectorPolynomial:
    """m независимых многочленов Шамира для одного векторного секрета.

    rng передаётся снаружи для воспроизводимости от seed; глобальный random
    существующего ядра не используется.
    """
    def __init__(self, secret: Sequence[int], t: int, p: int, rng: Random):
        if type(t) is not int or not 2 <= t < p:
            raise ValueError('Некорректный порог t')
        if len(secret) < 2 or any(type(v) is not int or not 0 <= v < p for v in secret):
            raise ValueError('Координаты секрета должны принадлежать F_p')
        self.p, self.t = p, t
        self.secret = tuple(secret)
        self.coefficients: tuple[tuple[int, ...], ...] = tuple(
            (v, *(rng.randrange(p) for _ in range(t - 1))) for v in self.secret
        )
        self.polynomials = tuple(F_pPolynomial(
            p, coefficients=[F_pElement(p, a) for a in coeffs]
        ) for coeffs in self.coefficients)

    def at(self, x: int) -> VectorShare:
        if not 1 <= x < self.p:
            raise ValueError('Точка не в диапазоне F_p\\{0}')
        point = F_pElement(self.p, x)
        return VectorShare(x, tuple(Share.make(poly, point).value.value
                                    for poly in self.polynomials), self.p)


def aggregate_vector_shares(shares: Sequence[VectorShare]) -> VectorShare:
    """s_j[k] = sum_i f_i,k(j) mod p через скалярный ShamirCryptography."""
    if not shares:
        raise ValueError('Необходима хотя бы одна доля')
    first = shares[0]
    if any(s.x != first.x or s.p != first.p or len(s.values) != len(first.values) for s in shares):
        raise ValueError('Доли должны иметь общий x, поле и размерность')
    x = F_pElement(first.p, first.x)
    coordinates = []
    for k in range(len(first.values)):
        scalar = [Share(x, F_pElement(first.p, s.values[k])) for s in shares]
        result = ShamirCryptography.total_sum_at_participant(scalar)
        coordinates.append(result.value.value)
    return VectorShare(first.x, tuple(coordinates), first.p)


def reconstruct_vector(shares: Sequence[VectorShare], t: int) -> tuple[int, ...]:
    """S[k] = F_k(0) отдельно для каждой координаты, по ровно t точкам."""
    if len(shares) != t:
        raise ValueError('Для реконструкции выберите ровно t долей')
    first = shares[0]
    if len({s.x for s in shares}) != t or any(
        s.p != first.p or len(s.values) != len(first.values) for s in shares
    ):
        raise ValueError('x должны различаться, размерность и поле совпадать')
    result = []
    for k in range(len(first.values)):
        scalar = [Share(F_pElement(first.p, s.x), F_pElement(first.p, s.values[k]))
                  for s in shares]
        value = ShamirCryptography.sum_of_secrets(scalar, t)
        # В промежуточной ветке sum_of_secrets могла возвращать Share,
        # тогда как в актуализированном API возвращается F_pElement.
        result.append(value.value.value if isinstance(value, Share) else value.value)
    return tuple(result)


def interpolate_vector_polynomials(shares: Sequence[VectorShare], t: int) -> tuple[tuple[int, ...], ...]:
    """Коэффициенты восстановленных F_k(x) по ровно t точкам."""
    if len(shares) != t or len({s.x for s in shares}) != t:
        raise ValueError('Нужно ровно t точек с разными x')
    first = shares[0]
    if any(s.p != first.p or len(s.values) != len(first.values) for s in shares):
        raise ValueError('Несовместимые доли')
    result = []
    for k in range(len(first.values)):
        scalar = [Share(F_pElement(first.p, s.x), F_pElement(first.p, s.values[k]))
                  for s in shares]
        poly = LagrangeInterpolation(scalar).lagrange_polynomial
        result.append(tuple(poly[a].value for a in range(t)))
    return tuple(result)


# -------------------------------------------------------------------------
# Новое OHE/VSS API: одна ShamirCryptography на каждую координату вектора.
# Старые VectorPolynomial / VectorShare сохранены для совместимости.
# -------------------------------------------------------------------------
from src.core.environment import CryptographyEnvironment
from src.core.verifiable_share import VerifiableShare
from src.core.commitments import PolynomialCommitments


class VectorShamirCryptography:
    """Многовариантный бюллетень с независимым Pedersen VSS для каждой координаты.

    У каждого участника *m* экземпляров ShamirCryptography с общим окружением.
    Объект не раскрывает чужие секреты. One-hot допустимость не доказывается:
    для защиты от произвольного секретного вектора потребуется ZKP.
    """

    def __init__(self, environment: CryptographyEnvironment,
                 person_id: int | F_pElement, secret: Sequence[int]):
        p = environment.p
        if type(person_id) is int:
            person_id = F_pElement(p, person_id)
        if len(secret) < 2 or any(type(v) is not int or not 0 <= v < p for v in secret):
            raise ValueError('Некорректный секретный вектор')
        if person_id.value == 0:
            raise ValueError('ID участника не должен быть нулём')
        self.environment = environment
        self.person_id = person_id
        self.secret = tuple(secret)
        self.coordinates = tuple(ShamirCryptography(environment, person_id,
                                  F_pElement(p, v)) for v in secret)

    @property
    def commitments(self) -> tuple[PolynomialCommitments, ...]:
        return tuple(s.pedersen_vss.public_commitments for s in self.coordinates)

    def verifiable_share_to(self, recipient: int | F_pElement) -> tuple[VerifiableShare, ...]:
        if type(recipient) is int:
            recipient = F_pElement(self.environment.p, recipient)
        return tuple(s.verifiable_share_to(recipient) for s in self.coordinates)

    def verify(self, shares: tuple[VerifiableShare, ...],
               commitments: tuple[PolynomialCommitments, ...]) -> bool:
        if len(shares) != len(self.coordinates) or len(commitments) != len(self.coordinates):
            return False
        try:
            return all(s.verify(share, c) for s, share, c
                       in zip(self.coordinates, shares, commitments))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def total_sum_at_participant(bundles: Sequence[tuple[VerifiableShare, ...]]
                                 ) -> tuple[VerifiableShare, ...]:
        if not bundles:
            raise ValueError('Нет долей для суммирования')
        m = len(bundles[0])
        if not m or any(len(bundle) != m for bundle in bundles):
            raise ValueError('Размерность долей должна совпадать')
        return tuple(ShamirCryptography.total_sum_at_participant(
            [bundle[k] for bundle in bundles]) for k in range(m))

    @staticmethod
    def commitments_of_total_sum(bundles: Sequence[tuple[PolynomialCommitments, ...]]
                                 ) -> tuple[PolynomialCommitments, ...]:
        if not bundles:
            raise ValueError('Нет обязательств')
        m = len(bundles[0])
        if not m or any(len(bundle) != m for bundle in bundles):
            raise ValueError('Несогласованные обязательства')
        return tuple(ShamirCryptography.commitments_of_total_sum(
            [bundle[k] for bundle in bundles]) for k in range(m))

    @staticmethod
    def sum_of_secrets(bundles: Sequence[tuple[VerifiableShare, ...]],
                       threshold: int) -> tuple[int, ...]:
        """Реконструкция покоординатно; возвращает натуральные счётчики mod p."""
        if len(bundles) < threshold or not bundles:
            raise ValueError('Недостаточно публичных долей')
        m = len(bundles[0])
        if not m or any(len(bundle) != m for bundle in bundles):
            raise ValueError('Несовместимые векторные доли')
        result = []
        for k in range(m):
            secret = ShamirCryptography.sum_of_secrets(
                [bundle[k] for bundle in bundles], threshold)
            # Поддержка промежуточной реализации API, возвращавшей Share.
            scalar = secret.value if isinstance(secret, Share) else secret
            result.append(scalar.value)
        return tuple(result)
