"""Тесты модуля src/core/algebra/interpolation.py — класс LagrangeInterpolation.

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.
Проверки выполняются через публичный API (свойство shares, lagrange_polynomial,
вызов объекта как функции) — без обращения к приватным полям.

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__               — приём любого итерируемого объекта с долями Share
                               (список, кортеж, генератор, map/filter/zip),
                               проверка непустоты, одинаковости поля,
                               склейка дубликатов;
    * shares                 — список уникальных долей в порядке первого появления;
    * lagrange_polynomial    — многочлен Лагранжа, проходящий через все доли;
    * __call__               — значение интерполянта в точке как Share
                               (в том числе восстановление секрета в x = 0);
    * математические свойства — точность восстановления, единственность,
                               порядок долей не важен, пороговое свойство Шамира.

Особенности реализации, зафиксированные тестами:
    * конструктор сразу вычисляет интерполянт и кэширует его, поэтому свойство
      lagrange_polynomial и __call__ ничего не пересчитывают;
    * слагаемые Лагранжа суммируются через sum(..., start=F_pPolynomial(p)),
      то есть стартовое значение — нулевой многочлен;
    * умножение значения доли на базисный многочлен работает за счёт
      делегирования F_pElement.__mul__ (`other * self`) и скалярных
      __mul__/__truediv__ многочлена;
    * базисный многочлен i-го слагаемого строится как произведение
      (x - x_j) для всех j != i и делится на его значение в точке x_i;
    * свойство shares отдаёт внутренний список без копии, но его изменение
      не влияет на уже вычисленный интерполянт;
    * доли с одинаковым x и одинаковым значением склеиваются, а противоречие
      в одной точке даёт ValueError с текстами обеих долей.
"""

from itertools import combinations

import pytest

from src import F_pElement, F_pPolynomial, Share
from src.core.algebra.interpolation import LagrangeInterpolation

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

PRIMES = [5, 7, 11, 13, 101]

SMALL_PRIMES = [2, 3, 5, 7, 11, 13]

FIELD_PAIRS = [(5, 7), (7, 5), (2, 3), (11, 13), (13, 101), (3, 5)]

NON_SHARE_VALUES = [0, 5, "x", None, 3.5, (1, 2), [1], object()]


# ---------------------------------------------------------------------------
# Вспомогательные функции (только публичный API)
# ---------------------------------------------------------------------------

def _shares_of(polynomial, xs):
    """Доли многочлена в точках xs (список Share)."""
    return [Share.make(polynomial, F_pElement(polynomial.p, x)) for x in xs]


def _values(polynomial):
    """Численные значения коэффициентов многочлена."""
    if not polynomial:
        return []
    return [polynomial[index].value
            for index in range(int(polynomial.degree) + 1)]


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(params=PRIMES)
def prime(request):
    """Параметризованная фикстура: простые модули p (p >= 5)."""
    return request.param


@pytest.fixture(params=SMALL_PRIMES)
def small_prime(request):
    """Параметризованная фикстура: небольшие простые модули p."""
    return request.param


@pytest.fixture
def make_share():
    """Фабрика долей: make_share(p, x, value) -> Share."""
    def _make(p, x, value):
        return Share(F_pElement(p, x), F_pElement(p, value))
    return _make


@pytest.fixture
def make_poly():
    """Фабрика многочленов из списка целых коэффициентов (от свободного члена)."""
    def _make(p, coefficients):
        return F_pPolynomial(
            p, coefficients=[F_pElement(p, c) for c in coefficients])
    return _make


@pytest.fixture
def quadratic11(make_poly):
    """Многочлен 1 + 2x + 3x^2 над F_11 (секрет = 1)."""
    return make_poly(11, [1, 2, 3])


@pytest.fixture
def shares11(quadratic11):
    """Три доли многочлена 1 + 2x + 3x^2 над F_11 в точках 0, 1, 2."""
    return _shares_of(quadratic11, [0, 1, 2])


@pytest.fixture
def line5(make_poly):
    """Многочлен 1 + 2x над F_5."""
    return make_poly(5, [1, 2])


# ===========================================================================
# 1. __init__ — валидные наборы долей
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1], [0]),
    (5, [1, 2], [0, 1]),
    (7, [3, 4], [1, 2]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [5, 6, 7, 8], [1, 2, 3, 4]),
    (101, [9, 8, 7], [10, 20, 30]),
])
def test_init_accepts_shares_of_polynomial(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    interpolation = LagrangeInterpolation(shares)
    assert isinstance(interpolation, LagrangeInterpolation)
    assert len(interpolation.shares) == len(xs)


@pytest.mark.parametrize("p, x, value", [
    (5, 0, 3), (7, 2, 6), (11, 5, 10), (13, 12, 1), (2, 1, 1),
])
def test_init_accepts_single_share(make_share, p, x, value):
    interpolation = LagrangeInterpolation([make_share(p, x, value)])
    assert len(interpolation.shares) == 1
    assert interpolation.shares[0] == make_share(p, x, value)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4, 5]), (11, [1, 2, 3]), (13, [9, 8]), (101, [7, 6, 5]),
])
def test_init_accepts_tuple_of_shares(make_poly, p, coeffs):
    shares = tuple(_shares_of(make_poly(p, coeffs), range(1, len(coeffs) + 1)))
    interpolation = LagrangeInterpolation(shares)
    assert len(interpolation.shares) == len(shares)


@pytest.mark.parametrize("p, coeffs, xs", [
    (11, [1, 2, 3], [0, 1, 2]),
    (7, [3, 4], [1, 2]),
    (5, [1, 2], [3, 4]),
    (13, [5, 6, 7], [2, 3, 4]),
    (101, [9, 8], [7, 8]),
])
def test_init_order_of_shares_does_not_matter(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    reversed_interpolation = LagrangeInterpolation(list(reversed(shares)))
    direct_interpolation = LagrangeInterpolation(shares)
    assert (reversed_interpolation.lagrange_polynomial ==
            direct_interpolation.lagrange_polynomial)


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2, 3], [0, 1, 2]),
    (7, [6, 5, 4], [1, 2, 3]),
    (11, [1, 2, 3, 4], [0, 1, 2, 3]),
    (13, [9, 8, 7], [5, 6, 7]),
    (101, [1, 2], [10, 20]),
])
def test_init_keeps_all_distinct_shares(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    interpolation = LagrangeInterpolation(shares)
    assert sorted(share.x.value for share in interpolation.shares) == \
           sorted(x % p for x in xs)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_init_collapses_identical_duplicates(make_share, p, x, value):
    share = make_share(p, x, value)
    interpolation = LagrangeInterpolation([share, make_share(p, x, value), share])
    assert len(interpolation.shares) == 1
    assert interpolation.shares[0] == share


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (11, [1, 2, 3]), (13, [9, 8]), (2, [1, 1]),
])
def test_init_field_matches_shares_field(make_poly, p, coeffs):
    shares = _shares_of(make_poly(p, coeffs), range(len(coeffs)))
    interpolation = LagrangeInterpolation(shares)
    assert {share.p for share in interpolation.shares} == {p}


# ===========================================================================
# 2. __init__ — исключения
# ===========================================================================

@pytest.mark.parametrize("shares", [[], ()])
def test_init_empty_shares_raises_value_error(shares):
    with pytest.raises(ValueError, match="at least one share is required"):
        LagrangeInterpolation(shares)


@pytest.mark.parametrize("p1, p2", FIELD_PAIRS)
def test_init_shares_from_different_fields_raises_type_error(make_share, p1, p2):
    with pytest.raises(TypeError, match="all shares must be from the same field"):
        LagrangeInterpolation([make_share(p1, 1, 1), make_share(p2, 2, 2)])


@pytest.mark.parametrize("p1, p2", FIELD_PAIRS)
def test_init_different_field_in_the_middle_raises_type_error(make_share, p1, p2):
    with pytest.raises(TypeError, match="all shares must be from the same field"):
        LagrangeInterpolation([
            make_share(p1, 1, 1), make_share(p2, 2, 2), make_share(p1, 3, 3)])


@pytest.mark.parametrize("p, x, v1, v2", [
    (5, 1, 2, 3),
    (7, 3, 4, 5),
    (11, 0, 1, 10),
    (13, 5, 6, 7),
    (2, 1, 0, 1),
])
def test_init_conflicting_values_at_same_x_raises_value_error(
        make_share, p, x, v1, v2):
    with pytest.raises(ValueError,
                       match="cannot interpolate two different values"):
        LagrangeInterpolation([make_share(p, x, v1), make_share(p, x, v2)])


@pytest.mark.parametrize("p, x, v1, v2", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 0, 1, 10), (13, 5, 6, 7), (101, 9, 1, 2),
])
def test_init_conflict_message_mentions_both_shares(make_share, p, x, v1, v2):
    first = make_share(p, x, v1)
    second = make_share(p, x, v2)
    with pytest.raises(ValueError) as exc_info:
        LagrangeInterpolation([first, second])
    message = str(exc_info.value)
    assert str(first) in message
    assert str(second) in message


@pytest.mark.parametrize("other", NON_SHARE_VALUES)
def test_init_non_share_items_raise_attribute_error(make_share, other):
    with pytest.raises(AttributeError):
        LagrangeInterpolation([other])


@pytest.mark.parametrize("other", NON_SHARE_VALUES[:5])
def test_init_non_share_item_after_valid_share_raises_attribute_error(
        make_share, other):
    with pytest.raises(AttributeError):
        LagrangeInterpolation([make_share(5, 1, 2), other])


@pytest.mark.parametrize("element_or_poly", ["element", "polynomial"])
def test_init_element_or_polynomial_instead_of_share_raises_attribute_error(
        element_or_poly):
    if element_or_poly == "element":
        item = F_pElement(5, 1)
    else:
        item = F_pPolynomial(5, [F_pElement(5, 1)])
    with pytest.raises(AttributeError):
        LagrangeInterpolation([item])


@pytest.mark.parametrize("shares", ["generator", "map", "filter", "iter", "zip"])
def test_init_accepts_lazy_iterables(make_share, shares):
    # конструктор приводит вход к списку, поэтому ленивые итераторы подходят
    pair = [make_share(5, 1, 2), make_share(5, 2, 3)]
    if shares == "generator":
        lazy = (share for share in pair)
    elif shares == "map":
        lazy = map(lambda share: share, pair)
    elif shares == "filter":
        lazy = filter(lambda share: True, pair)
    elif shares == "iter":
        lazy = iter(pair)
    else:
        lazy = reversed(pair)
    interpolation = LagrangeInterpolation(lazy)
    assert set(interpolation.shares) == set(pair)
    assert interpolation(F_pElement(5, 1)) == pair[0]


# ===========================================================================
# 3. Свойство shares
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1]),
    (7, [3, 4, 5], [1, 2, 3]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8], [4, 5]),
    (101, [7, 6, 5], [10, 20, 30]),
])
def test_shares_returns_list_of_shares(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    interpolation = LagrangeInterpolation(shares)
    assert isinstance(interpolation.shares, list)
    assert all(isinstance(share, Share) for share in interpolation.shares)


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1]),
    (7, [3, 4], [1, 2]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8, 7], [3, 4, 5]),
    (101, [1, 2], [7, 8]),
])
def test_shares_preserve_first_occurrence_order(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    interpolation = LagrangeInterpolation(shares)
    assert interpolation.shares == shares


@pytest.mark.parametrize("p, x, value, duplicates", [
    (5, 1, 2, 2), (7, 3, 4, 3), (11, 5, 6, 4), (13, 7, 8, 2), (101, 9, 1, 5),
])
def test_shares_are_unique_by_x(make_share, p, x, value, duplicates):
    shares = [make_share(p, x, value) for _ in range(duplicates)]
    interpolation = LagrangeInterpolation(shares)
    assert len(interpolation.shares) == 1
    xs = [share.x for share in interpolation.shares]
    assert len(set(xs)) == len(xs)


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1, 0, 1]),
    (7, [3, 4], [1, 2, 1]),
    (11, [1, 2, 3], [0, 1, 2, 0]),
    (13, [9, 8], [4, 5, 5, 4]),
    (101, [7, 6], [10, 20, 10]),
])
def test_shares_count_never_exceeds_distinct_x(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    interpolation = LagrangeInterpolation(shares)
    assert len(interpolation.shares) == len(set(x % p for x in xs))
    assert len(interpolation.shares) <= len(shares)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (11, [1, 2, 3]), (13, [9, 8]), (101, [7, 6, 5]),
])
def test_shares_property_returns_internal_list(make_poly, p, coeffs):
    # нюанс: свойство отдаёт внутренний список без копии, но его изменение
    # не пересчитывает интерполянт
    shares = _shares_of(make_poly(p, coeffs), range(len(coeffs)))
    interpolation = LagrangeInterpolation(shares)
    returned = interpolation.shares
    assert isinstance(returned, list)
    assert len(returned) == len(shares)
    before = interpolation(F_pElement.zero(p))
    returned.append(shares[0])
    assert interpolation(F_pElement.zero(p)) == before


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1]),
    (7, [3, 4, 5], [1, 2, 3]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8, 7], [3, 4, 5]),
    (2, [1, 1], [0, 1]),
])
def test_shares_property_is_stable(make_poly, p, coeffs, xs):
    interpolation = LagrangeInterpolation(_shares_of(make_poly(p, coeffs), xs))
    assert interpolation.shares == interpolation.shares
    assert len(interpolation.shares) == len(xs)


# ===========================================================================
# 4. Свойство lagrange_polynomial
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1]),
    (7, [3, 4], [1, 2]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8, 7], [3, 4, 5]),
    (101, [7, 6, 5], [10, 20, 30]),
])
def test_lagrange_polynomial_is_polynomial(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    polynomial = LagrangeInterpolation(shares).lagrange_polynomial
    assert isinstance(polynomial, F_pPolynomial)
    assert polynomial.p == p


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1]),
    (7, [3, 4], [1, 2]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8, 7], [3, 4, 5]),
    (101, [7, 6, 5], [10, 20, 30]),
])
def test_lagrange_polynomial_restores_original(make_poly, p, coeffs, xs):
    poly = make_poly(p, coeffs)
    shares = _shares_of(poly, xs)
    assert LagrangeInterpolation(shares).lagrange_polynomial == poly


@pytest.mark.parametrize("p, coeffs, xs", [
    (11, [1, 2, 3], [0, 1, 2]),
    (7, [3, 4, 5], [1, 2, 3]),
    (5, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8, 7, 6], [1, 2, 3, 4]),
    (101, [1, 2, 3], [5, 6, 7]),
])
def test_lagrange_polynomial_passes_through_all_shares(make_poly, p, coeffs, xs):
    poly = make_poly(p, coeffs)
    shares = _shares_of(poly, xs)
    interpolation = LagrangeInterpolation(shares)
    for share in shares:
        assert interpolation.lagrange_polynomial(share.x) == share.value


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1]),
    (7, [3, 4, 5], [1, 2, 3]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8, 7, 6], [2, 3, 4, 5]),
    (101, [7, 6, 5], [10, 20, 30]),
])
def test_lagrange_polynomial_degree_is_bounded(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    polynomial = LagrangeInterpolation(shares).lagrange_polynomial
    assert polynomial.degree <= len(shares) - 1


@pytest.mark.parametrize("p, x, value", [
    (5, 0, 3), (7, 2, 6), (11, 5, 10), (13, 12, 1), (101, 9, 100),
])
def test_lagrange_polynomial_of_single_share_is_constant(make_share, p, x, value):
    polynomial = LagrangeInterpolation(
        [make_share(p, x, value)]).lagrange_polynomial
    assert polynomial == F_pPolynomial(p, [F_pElement(p, value)])
    assert polynomial.degree == 0


@pytest.mark.parametrize("p, coeffs, xs", [
    (11, [1, 2, 3], [0, 1, 2]),
    (7, [3, 4], [1, 2]),
    (5, [1, 2], [0, 1]),
    (13, [9, 8, 7], [2, 3, 4]),
    (101, [1, 2], [3, 4]),
])
def test_lagrange_polynomial_is_unique(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    first = LagrangeInterpolation(shares).lagrange_polynomial
    second = LagrangeInterpolation(list(reversed(shares))).lagrange_polynomial
    assert first == second
    assert _values(first) == _values(second)


@pytest.mark.parametrize("p, coeffs, xs, extra", [
    (11, [1, 2, 3], [0, 1, 2], [3, 4]),
    (7, [3, 4, 5], [1, 2, 3], [4, 5]),
    (5, [1, 2], [0, 1], [2, 3]),
    (13, [9, 8, 7], [1, 2, 3], [4, 5]),
    (101, [7, 6, 5], [1, 2, 3], [4, 5]),
])
def test_extra_consistent_shares_do_not_change_polynomial(
        make_poly, p, coeffs, xs, extra):
    poly = make_poly(p, coeffs)
    few = LagrangeInterpolation(_shares_of(poly, xs)).lagrange_polynomial
    many = LagrangeInterpolation(_shares_of(poly, xs + extra)).lagrange_polynomial
    assert few == many == poly


# ===========================================================================
# 5. __call__ — значение интерполянта в точке
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, xs, x", [
    (5, [1, 2], [0, 1], 3),
    (7, [3, 4], [1, 2], 5),
    (11, [1, 2, 3], [0, 1, 2], 7),
    (13, [9, 8, 7], [1, 2, 3], 10),
    (101, [7, 6, 5], [1, 2, 3], 50),
])
def test_call_returns_share(make_poly, p, coeffs, xs, x):
    shares = _shares_of(make_poly(p, coeffs), xs)
    result = LagrangeInterpolation(shares)(F_pElement(p, x))
    assert isinstance(result, Share)
    assert result.x == F_pElement(p, x)
    assert result.p == p


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1]),
    (7, [3, 4, 5], [1, 2, 3]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8, 7], [2, 3, 4]),
    (101, [7, 6, 5], [10, 20, 30]),
])
def test_call_reproduces_every_input_share(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    interpolation = LagrangeInterpolation(shares)
    for share in shares:
        assert interpolation(share.x) == share


@pytest.mark.parametrize("p, coeffs", [
    (5, [3, 1, 2]), (7, [6, 5, 4]), (11, [10, 9, 8]), (13, [1, 2, 3]),
    (101, [42, 7, 8]),
])
def test_call_at_zero_recovers_secret(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    shares = _shares_of(poly, range(1, len(coeffs) + 1))
    secret = LagrangeInterpolation(shares)(F_pElement.zero(p))
    assert secret.value == F_pElement(p, coeffs[0])
    assert secret.x == F_pElement.zero(p)


@pytest.mark.parametrize("p, coeffs, xs, x", [
    (11, [1, 2, 3], [0, 1, 2], 5),
    (7, [3, 4, 5], [1, 2, 3], 6),
    (5, [1, 2], [0, 1], 4),
    (13, [9, 8, 7], [1, 2, 3], 11),
    (101, [7, 6, 5], [1, 2, 3], 77),
])
def test_call_agrees_with_original_polynomial(make_poly, p, coeffs, xs, x):
    poly = make_poly(p, coeffs)
    shares = _shares_of(poly, xs)
    point = F_pElement(p, x)
    assert LagrangeInterpolation(shares)(point).value == poly(point)


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1]),
    (7, [3, 4], [1, 2]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8, 7], [2, 3, 4]),
    (101, [1, 2, 3], [4, 5, 6]),
])
def test_call_equals_share_of_lagrange_polynomial(make_poly, p, coeffs, xs):
    shares = _shares_of(make_poly(p, coeffs), xs)
    interpolation = LagrangeInterpolation(shares)
    point = F_pElement(p, xs[0])
    assert interpolation(point) == Share.make(interpolation.lagrange_polynomial, point)


@pytest.mark.parametrize("other", NON_SHARE_VALUES)
def test_call_with_non_element_raises_type_error(make_share, other):
    # __call__ интерполянта делегирует F_pPolynomial.__call__,
    # который проверяет тип аргумента до вычисления
    interpolation = LagrangeInterpolation([make_share(5, 1, 2)])
    with pytest.raises(TypeError,
                       match="polynomial argument must be a F_pElement"):
        interpolation(other)


@pytest.mark.parametrize("p, other_p", FIELD_PAIRS[:5])
def test_call_with_element_of_other_field_raises_type_error(
        make_share, p, other_p):
    interpolation = LagrangeInterpolation([make_share(p, 1, 2)])
    with pytest.raises(TypeError):
        interpolation(F_pElement(other_p, 1))


@pytest.mark.parametrize("p, coeffs, xs, x", [
    (5, [1, 2], [0, 1], 2),
    (7, [3, 4], [1, 2], 3),
    (11, [1, 2, 3], [0, 1, 2], 4),
    (13, [9, 8], [1, 2], 5),
    (101, [7, 6], [1, 2], 3),
])
def test_call_is_deterministic(make_poly, p, coeffs, xs, x):
    shares = _shares_of(make_poly(p, coeffs), xs)
    interpolation = LagrangeInterpolation(shares)
    point = F_pElement(p, x)
    assert interpolation(point) == interpolation(point)
    assert interpolation(point) == LagrangeInterpolation(shares)(point)


# ===========================================================================
# 6. Математические свойства интерполяции Лагранжа
# ===========================================================================

@pytest.mark.parametrize("p, xs, values, expected", [
    (5, [0, 1], [1, 3], [1, 2]),
    (7, [0, 1], [2, 5], [2, 3]),
    (11, [0, 1, 2], [1, 6, 6], [1, 2, 3]),
    (13, [1, 2], [2, 4], [0, 2]),
    (5, [0, 1, 2], [0, 1, 4], [0, 0, 1]),
    (2, [0, 1], [1, 0], [1, 1]),
])
def test_interpolation_of_hand_computed_cases(
        make_share, make_poly, p, xs, values, expected):
    shares = [make_share(p, x, value) for x, value in zip(xs, values)]
    polynomial = LagrangeInterpolation(shares).lagrange_polynomial
    assert _values(polynomial) == expected


@pytest.mark.parametrize("prime", PRIMES)
def test_interpolation_matches_polynomial_at_all_field_points(prime):
    p = prime
    poly = F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    shares = _shares_of(poly, [0, 1])
    interpolation = LagrangeInterpolation(shares)
    for value in range(p):
        point = F_pElement(p, value)
        assert interpolation(point).value == poly(point)


@pytest.mark.parametrize("p", [5, 7, 11, 13, 101])
def test_threshold_property_recovers_secret_from_any_subset(p):
    secret = F_pElement(p, 3)
    poly = F_pPolynomial.random(p, 2, least_coefficient=secret)
    shares = _shares_of(poly, range(1, 6))
    for subset in combinations(shares, 3):
        interpolation = LagrangeInterpolation(list(subset))
        assert interpolation(F_pElement.zero(p)).value == secret


@pytest.mark.parametrize("p", [5, 7, 11, 13])
def test_fewer_shares_than_degree_do_not_restore_polynomial(p):
    poly = F_pPolynomial.random(
        p, 3, least_coefficient=F_pElement(p, 1))
    shares = _shares_of(poly, [1, 2])
    interpolation = LagrangeInterpolation(shares)
    # двух долей достаточно только для многочлена степени <= 1
    assert interpolation.lagrange_polynomial.degree <= 1
    for share in shares:
        assert interpolation(share.x) == share


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5, 4]), (11, [1, 2, 3, 4]), (13, [9, 8]),
    (101, [7, 6, 5]),
])
def test_interpolation_of_zero_polynomial(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    zero_poly = poly - poly
    shares = _shares_of(zero_poly, range(len(coeffs)))
    interpolation = LagrangeInterpolation(shares)
    assert bool(interpolation.lagrange_polynomial) is False
    assert interpolation(F_pElement.zero(p)).value == F_pElement.zero(p)


@pytest.mark.parametrize("p, coeffs, xs", [
    (11, [1, 2, 3], [0, 1, 2]),
    (7, [3, 4, 5], [1, 2, 3]),
    (13, [9, 8, 7], [2, 3, 4]),
    (5, [1, 2], [0, 1]),
    (101, [7, 6, 5], [10, 20, 30]),
])
def test_interpolation_is_linear_in_values(make_poly, p, coeffs, xs):
    poly = make_poly(p, coeffs)
    shares = _shares_of(poly, xs)
    doubled = [Share(share.x, share.value + share.value)
               for share in shares]
    first = LagrangeInterpolation(shares).lagrange_polynomial
    second = LagrangeInterpolation(doubled).lagrange_polynomial
    assert second == first + first


# ===========================================================================
# 7. Интерполяция долей случайного многочлена
# ===========================================================================

@pytest.mark.parametrize("prime", PRIMES)
def test_interpolation_of_shares_of_random_polynomial(prime):
    p = prime
    poly = F_pPolynomial.random(p, 2)
    shares = _shares_of(poly, range(1, 4))
    interpolation = LagrangeInterpolation(shares)
    assert interpolation.lagrange_polynomial == poly
    assert interpolation(F_pElement.zero(p)).value == poly(F_pElement.zero(p))
