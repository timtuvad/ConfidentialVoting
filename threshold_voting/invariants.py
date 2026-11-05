"""Проверки OHE-протокола над полным god Snapshot; не раскрывать их в чужих views."""
from __future__ import annotations
from src.core.vector import is_one_hot
from .models import Snapshot


def _eval(coefficients: tuple[int, ...], x: int, p: int) -> int:
    result = 0
    for a in reversed(coefficients):
        result = (result * x + a) % p
    return result


def _eval_vector(polys: tuple[tuple[int, ...], ...], x: int, p: int) -> tuple[int, ...]:
    return tuple(_eval(poly, x, p) for poly in polys)


def check_snapshot(s: Snapshot, n: int, t: int, p: int) -> dict[str, str]:
    """I1..I7 + OHE. I6 требует отдельного теста неразличимости наблюдений.

    OHE: каждый бюллетень обязан содержать одну единицу и остальные нули.
    I4: честно ли участник суммировал фактически ПОЛУЧЕННЫЕ векторы.
    I7: согласованы ли переданные/опубликованные значения с оригинальными
        многочленами; диагноз возможен в god view, но не сам по себе в протоколе.
    """
    status = {'I1': 'ok', 'I2': 'ok', 'I3': 'pending', 'I4': 'pending',
              'I5': 'pending', 'I6': 'unknown', 'I7': 'pending', 'OHE': 'ok'}
    m = len(s.votes[1])
    for i in range(1, n + 1):
        polys, vector = s.polys[i], s.votes[i]
        if not is_one_hot(vector):
            status['OHE'] = 'fail'
        if (len(polys) != m or any(len(poly) != t for poly in polys) or
                tuple(poly[0] for poly in polys) != vector):
            status['I2'] = 'fail'
    if any(any(v == s.votes[d.sender][k] for k, v in enumerate(d.value))
           for d in s.all_deals if d.receiver > 0):
        status['I1'] = 'warning'  # совпадение числа не означает утечку
    correct_deals = True
    if s.step >= 2:
        correct_deals = all(d.value == _eval_vector(s.polys[d.sender], d.receiver, p)
                            for d in s.all_deals if d.receiver > 0)
        status['I7'] = 'ok' if correct_deals else 'fail'
    if s.step >= n + 1:
        status['I3'] = 'ok' if all(len(s.received[j]) == n - 1 for j in range(1, n + 1)) else 'fail'
    if s.step >= n + 2:
        local_ok = True
        agreed_ok = True
        for j in range(1, n + 1):
            own = _eval_vector(s.polys[j], j, p)
            actual = tuple((own[k] + sum(vector[k] for vector in s.received[j].values())) % p
                           for k in range(m))
            correct = tuple(sum(_eval_vector(s.polys[i], j, p)[k]
                                for i in range(1, n + 1)) % p for k in range(m))
            if actual != s.shares[j]:
                local_ok = False
            if correct != s.shares[j]:
                agreed_ok = False
        status['I4'] = 'ok' if local_ok else 'fail'
        publications_ok = all(d.value == s.shares[d.sender]
                              for d in s.all_deals if d.receiver == -1)
        status['I7'] = 'ok' if correct_deals and agreed_ok and publications_ok else 'fail'
    if s.phase == 'reconstruct':
        results = list((s.reconstructed or {}).values())
        status['I5'] = ('fail' if not results else 'unknown' if len(results) == 1
                        else 'ok' if len(set(results)) == 1 else 'fail')
        if s.aggregate_F is not None and results:
            if tuple(poly[0] for poly in s.aggregate_F) != results[0]:
                status['I7'] = 'fail'
    return status
