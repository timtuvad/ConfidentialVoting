"""Независимая реконструкция НАБЛЮДАТЕЛЕМ только по публичным долям.

Ни доска объявлений, ни участники здесь не используются.
"""
from __future__ import annotations


def interpolate_coefficients(points: dict[int, int], p: int) -> tuple[int, ...]:
    """Коэффициенты многочлена Лагранжа над F_p; точки x обязаны быть разными."""
    if not points:
        raise ValueError('Нужна хотя бы одна точка')
    if len({x % p for x in points}) != len(points):
        raise ValueError('Повторяющиеся x по модулю p')
    result = [0] * len(points)
    for x, y in points.items():
        basis = [1]
        denominator = 1
        for other_x in points:
            if other_x == x:
                continue
            # Умножение на (X - other_x) в F_p.
            next_basis = [0] * (len(basis) + 1)
            for i, value in enumerate(basis):
                next_basis[i] = (next_basis[i] - other_x * value) % p
                next_basis[i + 1] = (next_basis[i + 1] + value) % p
            basis = next_basis
            denominator = (denominator * (x - other_x)) % p
        scale = (y * pow(denominator, -1, p)) % p
        for i, value in enumerate(basis):
            result[i] = (result[i] + scale * value) % p
    return tuple(result)


def reconstruct_public(points: dict[int, int], threshold: int, p: int) -> int:
    """F(0) по выбранным опубликованным s_j; меньше t точек недостаточно."""
    if len(points) < threshold:
        raise ValueError('Для восстановления требуется хотя бы t публичных долей')
    if len(points) != threshold:
        raise ValueError('Выберите ровно t долей; лишние можно проверить отдельно')
    return interpolate_coefficients(points, p)[0]


def alternative_curves(points: dict[int, int], threshold: int, p: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Два совместимых с <t точками многочлена с разными свободными членами.

    Используется для пояснения неоднозначности НА ОСНОВЕ ВЫБРАННЫХ точек.
    При наличии других уже опубликованных долей публичный итог может быть известен.
    """
    if len(points) >= threshold:
        raise ValueError('Альтернативы нужны только когда выбрано меньше t долей')
    if 0 in points:
        raise ValueError('Публичные доли должны иметь индексы 1..n')
    a = interpolate_coefficients({0: 0, **points}, p)
    b = interpolate_coefficients({0: 1, **points}, p)
    return a, b


def horner(coefficients: tuple[int, ...], x: int, p: int) -> int:
    """Вычислить многочлен в точке x по схеме Горнера."""
    result = 0
    for a in reversed(coefficients):
        result = (result * x + a) % p
    return result
