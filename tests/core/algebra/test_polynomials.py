"""Тесты модуля src/core/algebra/polynomial.py — класс F_pPolynomial (над F_p).

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.
Сравнение выполняется через публичный API (==, !=, hash(), str(), repr(),
свойства p / degree / degree_bound) — без обращения к приватным полям.

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__       — все валидные комбинации (p, degree_bound, coefficients)
                       и все исключения (ValueError / TypeError);
    * p, degree, degree_bound — свойства;
    * __eq__, __ne__, __hash__ — сравнение и хешируемость;
    * __str__        — печать "(k mod p)x^i", дополнение нулями до degree_bound;
    * __repr__       — отладочное представление (кортеж коэффициентов);
    * __pos__, __neg__ — унарные операции;
    * __add__, __sub__ — сложение/вычитание, их исключения и проверка типов;
    * служебные нюансы — отсутствие умножения/деления/степени, len(), итерации,
      индексации, __bool__; неизменяемость операндов и переданного списка.

Особенности реализации, зафиксированные тестами:
    * коэффициенты хранятся неизменяемым кортежем с отброшенными хвостовыми нулями
      (ведущие и средние нули сохраняются, поэтому degree их учитывает);
    * переданный список коэффициентов не изменяется (внутри делается копия);
    * проверка degree_bound выполняется по ИСХОДНОЙ длине списка коэффициентов,
      то есть до отбрасывания хвостовых нулей;
    * degree_bound участвует в __eq__/__hash__: многочлены с разной границей
      степени не равны даже при одинаковых коэффициентах;
    * __eq__ и арифметика используют строгую проверку type(self) is type(other):
      подкласс не равен родителю, а арифметика с подклассом даёт TypeError;
    * операции с объектами других типов дают TypeError
      ("unsupported operand type(s) ..."), а для разных полей — ValueError;
    * степень нулевого многочлена равна float('-inf');
    * при degree_bound = -1 str() возвращает пустую строку;
    * умножение, деление, возведение в степень, len(), итерация, индексация
      и abs() для многочленов не поддерживаются.
"""

import pytest

from src import F_pElement, F_pPolynomial

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

PRIMES = [2, 3, 5, 7, 11, 13, 101]

SMALL_PRIMES = [2, 3, 5, 7, 11, 13]

NON_PRIMES = [-5, 0, 1, 4, 6, 8, 9, 10, 12, 15, 21, 25, 100, 589423]

NON_POLYNOMIAL_VALUES = [0, 5, "x", None, 3.5, (1, 2), [1], object()]


class SubPolynomial(F_pPolynomial):
    """Подкласс для проверки строгой проверки type(self) is type(other)."""


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(params=PRIMES)
def prime(request):
    """Параметризованная фикстура: разные простые модули p."""
    return request.param


@pytest.fixture(params=SMALL_PRIMES)
def small_prime(request):
    """Параметризованная фикстура: небольшие простые модули p."""
    return request.param


@pytest.fixture
def make_element():
    """Фабрика элементов поля: make_element(p, value) -> F_pElement."""
    def _make(p, value=None):
        return F_pElement(p, value)
    return _make


@pytest.fixture
def make_poly():
    """Фабрика многочленов из списка целых коэффициентов (от свободного члена)."""
    def _make(p, coefficients, degree_bound=None):
        return F_pPolynomial(
            p, degree_bound,
            coefficients=[F_pElement(p, c) for c in coefficients])
    return _make


@pytest.fixture
def poly5(make_poly):
    """Многочлен 1 + 2x + 3x^2 над F_5."""
    return make_poly(5, [1, 2, 3])


@pytest.fixture
def poly5_short(make_poly):
    """Многочлен 4 + 4x над F_5 (короче poly5)."""
    return make_poly(5, [4, 4])


@pytest.fixture
def zero_poly5():
    """Нулевой многочлен над F_5."""
    return F_pPolynomial(5)


@pytest.fixture
def const_poly5(make_poly):
    """Константа (3 mod 5)."""
    return make_poly(5, [3])


# ===========================================================================
# 1. __init__ — валидные комбинации аргументов
# ===========================================================================

@pytest.mark.parametrize("p", PRIMES)
def test_init_with_p_only_creates_zero_polynomial(p):
    poly = F_pPolynomial(p)
    assert isinstance(poly, F_pPolynomial)
    assert poly.p == p
    assert poly.degree_bound is None
    assert poly.degree == float('-inf')
    assert str(poly) == f"(0 mod {p})"


@pytest.mark.parametrize("p, other_p", [
    (2, 3), (3, 5), (5, 7), (7, 11), (11, 13), (13, 101), (101, 2),
])
def test_zero_polynomials_of_different_fields_are_distinct(p, other_p):
    assert F_pPolynomial(p) != F_pPolynomial(other_p)
    assert F_pPolynomial(p).p == p
    assert F_pPolynomial(other_p).p == other_p


@pytest.mark.parametrize("p, coeffs", [
    (5, [1]),
    (5, [1, 2]),
    (7, [3, 0, 1]),
    (2, [1, 1]),
    (11, [10, 1]),
    (13, [0, 0, 1]),
    (101, [50, 60, 70]),
])
def test_init_with_coefficients_only_infers_p(p, coeffs):
    poly = F_pPolynomial(coefficients=[F_pElement(p, c) for c in coeffs])
    assert poly.p == p
    assert poly == F_pPolynomial(p, None,
                                 coefficients=[F_pElement(p, c) for c in coeffs])


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [6, 5, 4]), (2, [1, 0, 1]), (11, [1, 2, 3, 4]),
    (13, [12, 11]), (101, [1, 2, 3]),
])
def test_init_with_p_and_coefficients(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly.p == p
    assert poly.degree == len(coeffs) - 1
    assert poly.degree_bound is None


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [5, 6, 7]), (13, [1, 0, 1]),
])
def test_init_accepts_keyword_arguments(p, coeffs):
    poly = F_pPolynomial(p=p, degree_bound=None,
                         coefficients=[F_pElement(p, c) for c in coeffs])
    assert poly == F_pPolynomial(coefficients=[F_pElement(p, c) for c in coeffs])
    assert poly.p == p


@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 2, 0], [1, 2]),
    (5, [1, 0, 0], [1]),
    (5, [0, 0, 0], []),
    (7, [3, 0], [3]),
    (7, [0, 1, 0, 0], [0, 1]),
    (11, [1, 2, 3, 0, 0], [1, 2, 3]),
    (13, [0], []),
    (2, [1, 0, 0, 0], [1]),
])
def test_init_strips_trailing_zeros(make_poly, p, coeffs, expected):
    assert make_poly(p, coeffs) == make_poly(p, expected)


@pytest.mark.parametrize("p, coeffs", [
    (5, [0, 1]), (5, [0, 0, 1]), (7, [0, 3, 0, 4]), (11, [0, 5]), (13, [0, 0, 0, 1]),
])
def test_init_keeps_leading_and_middle_zeros(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly.degree == len(coeffs) - 1
    assert poly == make_poly(p, coeffs)
    assert poly != make_poly(p, coeffs[1:])


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 0]), (7, [3, 0, 0]), (2, [1, 0]), (11, [4, 5, 0, 0]), (13, [6, 0]),
])
def test_init_does_not_mutate_passed_list(p, coeffs):
    elements = [F_pElement(p, c) for c in coeffs]
    snapshot = list(elements)
    F_pPolynomial(p, None, coefficients=elements)
    assert elements == snapshot
    assert len(elements) == len(snapshot)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [5, 6]), (13, [7, 8, 9]),
])
def test_init_accepts_tuple_of_coefficients(p, coeffs):
    from_tuple = F_pPolynomial(p, None,
                               coefficients=tuple(F_pElement(p, c) for c in coeffs))
    from_list = F_pPolynomial(p, None,
                              coefficients=[F_pElement(p, c) for c in coeffs])
    assert from_tuple == from_list


@pytest.mark.parametrize("p, coeffs", [
    (5, []), (7, []), (2, []), (11, []), (13, []),
])
def test_init_empty_coefficients_with_p_is_zero_polynomial(p, coeffs):
    assert F_pPolynomial(p, None, coefficients=coeffs) == F_pPolynomial(p)


@pytest.mark.parametrize("bound", [0, 1, 2, 5, 10])
def test_init_degree_bound_with_empty_coefficients_list(bound):
    poly = F_pPolynomial(5, bound, coefficients=[])
    assert poly.degree_bound == bound
    assert poly.degree == float('-inf')
    assert poly == F_pPolynomial(5, bound)


def test_init_degree_bound_minus_one_with_empty_coefficients():
    poly = F_pPolynomial(5, -1, coefficients=[])
    assert poly.degree_bound == -1
    assert poly.degree == float('-inf')
    assert str(poly) == ""


@pytest.mark.parametrize("bound, coeffs", [
    (0, [1]), (1, [1, 2]), (2, [1, 2, 3]), (3, [1, 0, 0, 1]), (5, [2]), (10, [1, 2]),
])
def test_init_degree_bound_equal_or_greater_than_degree(make_poly, bound, coeffs):
    poly = make_poly(5, coeffs, bound)
    assert poly.degree_bound == bound
    assert poly.degree <= bound


@pytest.mark.parametrize("p", PRIMES)
def test_init_returns_instance_of_f_p_polynomial(p):
    assert type(F_pPolynomial(p)) is F_pPolynomial
    assert isinstance(
        F_pPolynomial(p, 3, coefficients=[F_pElement(p, 1)]), F_pPolynomial)


# ===========================================================================
# 2. __init__ — исключения
# ===========================================================================

@pytest.mark.parametrize("p", NON_PRIMES)
def test_init_non_prime_p_raises_value_error(p):
    with pytest.raises(ValueError, match=f"p must be prime, cannot be {p}"):
        F_pPolynomial(p)


@pytest.mark.parametrize("p", NON_PRIMES)
def test_init_non_prime_p_raises_even_with_coefficients(p):
    with pytest.raises(ValueError, match="p must be prime"):
        F_pPolynomial(p, None, coefficients=[F_pElement(5, 1)])


@pytest.mark.parametrize("kwargs", [
    {},
    {"coefficients": []},
    {"degree_bound": None},
    {"coefficients": [], "degree_bound": None},
    {"p": None, "coefficients": None},
    {"p": None},
])
def test_init_without_p_and_coefficients_raises_value_error(kwargs):
    with pytest.raises(ValueError, match="p or coefficients must be provided"):
        F_pPolynomial(**kwargs)


@pytest.mark.parametrize("p", PRIMES[:5])
def test_init_empty_coefficients_without_p_raises_value_error(p):
    # пустой список не позволяет определить поле -> p обязателен
    with pytest.raises(ValueError, match="p or coefficients must be provided"):
        F_pPolynomial(coefficients=[])


@pytest.mark.parametrize("p, other_p, values", [
    (5, 7, [1, 2]),
    (5, 11, [3, 1]),
    (7, 5, [1, 2]),
    (2, 3, [1, 1]),
    (11, 13, [5, 6]),
    (13, 101, [1, 2]),
])
def test_init_coefficients_from_other_field_raises_value_error(
        make_element, p, other_p, values):
    coefficients = [make_element(p, values[0]), make_element(other_p, values[1])]
    with pytest.raises(
            ValueError, match=f"coefficients must be from the field of order {p}"):
        F_pPolynomial(p, None, coefficients=coefficients)


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101), (3, 5),
])
def test_init_mixed_coefficients_without_p_raises_value_error(
        make_element, p1, p2):
    coefficients = [make_element(p1, 1), make_element(p2, 1)]
    with pytest.raises(
            ValueError, match=f"coefficients must be from the field of order {p1}"):
        F_pPolynomial(coefficients=coefficients)


@pytest.mark.parametrize("bound, coeffs", [
    (0, [1, 2]),
    (1, [1, 2, 3]),
    (2, [1, 2, 3, 4]),
    (3, [1, 2, 3, 4, 5]),
    (0, [0, 1]),
    (1, [1, 1, 1]),
])
def test_init_degree_bound_less_than_degree_raises_value_error(make_poly, bound, coeffs):
    with pytest.raises(ValueError,
                       match="degree_bound is less than coefficients degree"):
        make_poly(5, coeffs, bound)


@pytest.mark.parametrize("bound, coeffs", [
    (0, [1, 0]),
    (1, [1, 2, 0]),
    (1, [5, 0, 0]),
    (2, [1, 2, 0, 0]),
    (0, [3, 0, 0, 0]),
    (3, [1, 0, 0, 0, 0]),
])
def test_init_degree_bound_check_ignores_trailing_zero_stripping(
        make_poly, bound, coeffs):
    # нюанс: проверка len(coefficients) - 1 > degree_bound выполняется
    # по ИСХОДНОМУ списку, поэтому «лишние» хвостовые нули приводят к ValueError
    with pytest.raises(ValueError,
                       match="degree_bound is less than coefficients degree"):
        make_poly(5, coeffs, bound)


@pytest.mark.parametrize("bound", [0, 1, 2, 5, 10])
def test_init_degree_bound_with_none_coefficients_raises_type_error(bound):
    assert (F_pPolynomial(5, bound) ==
            F_pPolynomial(5, bound, coefficients=[
                F_pElement.zero(5) for _ in range(bound+1)
            ]))


@pytest.mark.parametrize("bound", [-2, -3, -5, -10, -100])
def test_init_negative_degree_bound_below_minus_one_raises_value_error(bound):
    with pytest.raises(ValueError,
                       match="degree_bound is less than coefficients degree"):
        F_pPolynomial(5, bound, coefficients=[])


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 11), (2, 5), (11, 13), (13, 3),
])
def test_init_field_check_precedes_degree_bound_check(p, other_p):
    # сначала проверяется поле коэффициентов, затем degree_bound
    with pytest.raises(ValueError, match="coefficients must be from the field"):
        F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(other_p, 1)])


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1]), (11, [4, 5]), (13, [6, 7, 8]),
])
def test_init_with_non_element_coefficients_raises(p, coeffs):
    # коэффициенты обязаны быть F_pElement: сравнение c.p у int -> AttributeError
    with pytest.raises((AttributeError, TypeError)):
        F_pPolynomial(p, None, coefficients=coeffs)


# ===========================================================================
# 3. Свойства p, degree, degree_bound
# ===========================================================================

@pytest.mark.parametrize("p", PRIMES)
def test_p_property_returns_field_order(p):
    assert F_pPolynomial(p).p == p


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_p_property_is_same_for_all_coefficients(make_poly, p, coeffs):
    assert make_poly(p, coeffs).p == p


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 1), (11, 5), (13, 12), (101, 7),
])
def test_p_property_is_read_only(make_poly, p, value):
    poly = make_poly(p, [value])
    with pytest.raises(AttributeError):
        poly.p = 7


@pytest.mark.parametrize("p", PRIMES)
def test_degree_of_zero_polynomial_is_minus_infinity(p):
    assert F_pPolynomial(p).degree == float('-inf')


@pytest.mark.parametrize("p", PRIMES)
def test_degree_of_zero_polynomial_is_negative_infinity_float(p):
    degree = F_pPolynomial(p).degree
    assert isinstance(degree, float)
    assert degree < 0
    assert degree == -float('inf')


@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1], 0),
    (5, [1, 2], 1),
    (5, [1, 2, 3], 2),
    (7, [1, 0, 0, 1], 3),
    (11, [1, 2, 3, 4, 5], 4),
    (13, [9], 0),
    (2, [1, 1, 1, 1], 3),
])
def test_degree_equals_number_of_coefficients_minus_one(make_poly, p, coeffs, expected):
    assert make_poly(p, coeffs).degree == expected


@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [0, 1], 1),
    (5, [0, 0, 1], 2),
    (7, [0, 0, 0, 3], 3),
    (11, [0, 5], 1),
    (13, [0, 0, 0, 0, 1], 4),
])
def test_degree_counts_leading_zeros(make_poly, p, coeffs, expected):
    assert make_poly(p, coeffs).degree == expected


@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 0], 0),
    (5, [1, 2, 0, 0], 1),
    (7, [3, 0, 0], 0),
    (11, [1, 2, 3, 0], 2),
    (13, [0, 1, 0, 0], 1),
])
def test_degree_after_trailing_zero_stripping(make_poly, p, coeffs, expected):
    assert make_poly(p, coeffs).degree == expected


@pytest.mark.parametrize("bound, coeffs", [
    (2, [1, 2]), (5, [1]), (3, [1, 2, 3]), (10, [0, 1]), (4, [1, 0, 0]),
])
def test_degree_never_exceeds_degree_bound(make_poly, bound, coeffs):
    poly = make_poly(5, coeffs, bound)
    assert poly.degree <= poly.degree_bound


@pytest.mark.parametrize("bound, coeffs, expected_degree", [
    (7, [1], 0),
    (5, [1, 2], 1),
    (10, [], float('-inf')),
    (3, [0, 0, 1], 2),
    (9, [1, 0], 0),
])
def test_degree_is_independent_of_degree_bound(make_poly, bound, coeffs, expected_degree):
    poly = make_poly(5, coeffs, bound)
    assert poly.degree == expected_degree
    assert poly.degree_bound == bound


@pytest.mark.parametrize("p, coeffs", [
    (5, [1]), (5, [1, 2, 3]), (7, [4]), (11, [1, 2]), (13, [0, 1]),
])
def test_degree_bound_default_is_none(make_poly, p, coeffs):
    assert make_poly(p, coeffs).degree_bound is None


@pytest.mark.parametrize("bound", [0, 1, 2, 3, 5, 17])
def test_degree_bound_is_stored(make_poly, bound):
    assert make_poly(5, [1], bound).degree_bound == bound


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 1), (11, 5), (13, 12), (101, 7),
])
def test_degree_bound_is_read_only(make_poly, p, value):
    poly = make_poly(p, [value], 3)
    with pytest.raises(AttributeError):
        poly.degree_bound = 5


# ===========================================================================
# 4. __eq__ / __ne__ / __hash__
# ===========================================================================

@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (5, []), (7, [3, 4]), (2, [1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_eq_is_reflexive(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly == poly
    assert not (poly != poly)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (5, []), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_eq_true_for_equal_content(make_poly, p, coeffs):
    assert make_poly(p, coeffs) == make_poly(p, coeffs)
    assert not (make_poly(p, coeffs) != make_poly(p, coeffs))


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [1, 2, 3]),
    (5, [1], [2]),
    (7, [3, 4], [4, 3]),
    (2, [1], [1, 1]),
    (11, [9, 8], [9, 8, 0, 1]),
    (13, [1, 0, 2], [1, 0, 3]),
])
def test_eq_false_for_different_coefficients(make_poly, p, a, b):
    assert make_poly(p, a) != make_poly(p, b)
    assert not (make_poly(p, a) == make_poly(p, b))


@pytest.mark.parametrize("p1, p2, coeffs", [
    (5, 7, [1]), (2, 3, [1]), (7, 11, [1, 2]), (5, 101, [3]), (13, 3, [1, 1]),
])
def test_eq_false_for_different_fields(make_poly, p1, p2, coeffs):
    assert make_poly(p1, coeffs) != make_poly(p2, coeffs)


@pytest.mark.parametrize("bound1, bound2", [
    (0, 1), (1, 2), (2, 5), (5, 3), (None, 0), (None, 4),
])
def test_eq_compares_degree_bound(make_poly, bound1, bound2):
    # нюанс: degree_bound входит в сравнение
    assert make_poly(5, [1], bound1) != make_poly(5, [1], bound2)
    assert make_poly(5, [1], bound1) == make_poly(5, [1], bound1)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 0]), (7, [3, 0]), (2, [1, 0, 0]), (11, [4, 5, 0]), (13, [6, 0, 0, 0]),
])
def test_eq_ignores_trailing_zeros(make_poly, p, coeffs):
    stripped = list(coeffs)
    while stripped and stripped[-1] % p == 0:
        stripped.pop()
    assert make_poly(p, coeffs) == make_poly(p, stripped)


@pytest.mark.parametrize("p, coeffs", [
    (5, [0, 1]), (7, [0, 0, 3]), (2, [0, 1, 1]), (11, [0, 5]), (13, [0, 0, 1]),
])
def test_eq_distinguishes_leading_zeros(make_poly, p, coeffs):
    assert make_poly(p, coeffs) != make_poly(p, coeffs[1:])


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES + [F_pElement(5, 1)])
def test_eq_with_non_polynomial_is_false(poly5, other):
    assert (poly5 == other) is False
    assert (poly5 != other) is True


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES[:6])
def test_eq_reflected_with_non_polynomial_is_false(poly5, other):
    assert (other == poly5) is False
    assert (other != poly5) is True


@pytest.mark.parametrize("p", PRIMES[:5])
def test_eq_with_subclass_is_false(p):
    # нюанс: type(self) is not type(other) -> False
    base = F_pPolynomial(p)
    sub = SubPolynomial(p)
    assert (base == sub) is False
    assert (sub == base) is False
    assert base != sub


@pytest.mark.parametrize("p", PRIMES[:5])
def test_eq_between_subclass_instances_works(p):
    assert SubPolynomial(p) == SubPolynomial(p)
    assert SubPolynomial(p, 2, coefficients=[]) != SubPolynomial(p, 3, coefficients=[])
    assert SubPolynomial(p, 2, coefficients=[]) == SubPolynomial(p, 2, coefficients=[])


@pytest.mark.parametrize("p, a, b, c", [
    (5, [1], [1], [2]),
    (7, [1, 2], [1, 2], [2, 1]),
    (2, [1], [1], [0, 1]),
    (11, [3], [3], [4]),
    (13, [1, 1], [1, 1], [1, 2]),
])
def test_eq_is_symmetric_and_transitive(make_poly, p, a, b, c):
    first, second, third = make_poly(p, a), make_poly(p, b), make_poly(p, c)
    assert (first == second) == (second == first)
    assert first == second
    assert second != third
    assert first != third


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (5, []), (7, [3]), (2, [1, 1]), (11, [4, 5, 6]), (13, [7]),
])
def test_hash_is_consistent_with_eq(make_poly, p, coeffs):
    first = make_poly(p, coeffs)
    second = make_poly(p, coeffs)
    assert first == second
    assert hash(first) == hash(second)
    assert isinstance(hash(first), int)


@pytest.mark.parametrize("p", PRIMES)
def test_zero_polynomials_are_hashable_and_equal(p):
    assert hash(F_pPolynomial(p)) == hash(F_pPolynomial(p))
    assert F_pPolynomial(p) == F_pPolynomial(p)
    assert len({F_pPolynomial(p), F_pPolynomial(p)}) == 1


@pytest.mark.parametrize("p, coeffs", [
    (5, [1]), (5, [1, 2]), (7, [3, 4]), (11, [1, 2, 3]), (13, [9]),
])
def test_polynomial_can_be_dict_key(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    mapping = {poly: "value"}
    assert mapping[make_poly(p, coeffs)] == "value"
    assert make_poly(p, coeffs) in mapping


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]),
    (7, [1], [2]),
    (2, [1, 1], [1]),
    (11, [5, 6], [7, 8]),
    (13, [1, 2, 3], [4, 5]),
])
def test_distinct_polynomials_coexist_in_set(make_poly, p, a, b):
    collection = {make_poly(p, a), make_poly(p, b), make_poly(p, a)}
    assert len(collection) == 2


@pytest.mark.parametrize("bound", [0, 1, 2, 5, 9])
def test_hash_distinguishes_degree_bound(make_poly, bound):
    assert len({make_poly(5, [1], bound), make_poly(5, [1], bound + 1)}) == 2


# ===========================================================================
# 5. __str__
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 2, 3], "(3 mod 5)x^2 + (2 mod 5)x^1 + (1 mod 5)"),
    (5, [3], "(3 mod 5)"),
    (7, [0, 0, 4], "(4 mod 7)x^2 + (0 mod 7)x^1 + (0 mod 7)"),
    (2, [1, 1], "(1 mod 2)x^1 + (1 mod 2)"),
    (11, [10, 0, 1], "(1 mod 11)x^2 + (0 mod 11)x^1 + (10 mod 11)"),
    (13, [0, 1], "(1 mod 13)x^1 + (0 mod 13)"),
])
def test_str_without_degree_bound(make_poly, p, coeffs, expected):
    assert str(make_poly(p, coeffs)) == expected


@pytest.mark.parametrize("p, coeffs, bound, expected", [
    (5, [1, 2, 3], 4,
     "(0 mod 5)x^4 + (0 mod 5)x^3 + (3 mod 5)x^2 + (2 mod 5)x^1 + (1 mod 5)"),
    (5, [3], 1, "(0 mod 5)x^1 + (3 mod 5)"),
    (5, [3], 0, "(3 mod 5)"),
    (7, [1], 2, "(0 mod 7)x^2 + (0 mod 7)x^1 + (1 mod 7)"),
    (11, [10], 1, "(0 mod 11)x^1 + (10 mod 11)"),
])
def test_str_with_degree_bound_pads_zeros(make_poly, p, coeffs, bound, expected):
    assert str(make_poly(p, coeffs, bound)) == expected


@pytest.mark.parametrize("p", PRIMES)
def test_str_of_zero_polynomial(p):
    assert str(F_pPolynomial(p)) == f"(0 mod {p})"


@pytest.mark.parametrize("p, bound, expected", [
    (5, 2, "(0 mod 5)x^2 + (0 mod 5)x^1 + (0 mod 5)"),
    (5, 1, "(0 mod 5)x^1 + (0 mod 5)"),
    (5, 0, "(0 mod 5)"),
    (7, 3, "(0 mod 7)x^3 + (0 mod 7)x^2 + (0 mod 7)x^1 + (0 mod 7)"),
    (11, 1, "(0 mod 11)x^1 + (0 mod 11)"),
])
def test_str_of_zero_polynomial_with_degree_bound(p, bound, expected):
    # degree_bound без coefficients даёт TypeError, поэтому передаём пустой список
    assert str(F_pPolynomial(p, bound, coefficients=[])) == expected


@pytest.mark.parametrize("p, value, expected", [
    (5, 0, "(0 mod 5)"),
    (5, 4, "(4 mod 5)"),
    (7, 6, "(6 mod 7)"),
    (11, 0, "(0 mod 11)"),
    (13, 12, "(12 mod 13)"),
])
def test_str_of_constant_polynomial(make_poly, p, value, expected):
    assert str(make_poly(p, [value])) == expected


@pytest.mark.parametrize("p", PRIMES)
def test_str_with_negative_degree_bound_is_empty(p):
    # нюанс: range(-1, -1, -1) пуст -> join даёт пустую строку
    assert str(F_pPolynomial(p, -1, coefficients=[])) == ""


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4, 5]), (2, [1, 0, 1]), (11, [9, 8]), (13, [1, 0, 0, 2]),
])
def test_str_contains_every_coefficient(make_poly, p, coeffs):
    text = str(make_poly(p, coeffs))
    for value in coeffs:
        assert f"({value % p} mod {p})" in text
    assert text.count("x^") == max(len(coeffs) - 1, 0)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [1, 2]), (2, [1, 1]), (11, [1, 2, 3, 4]), (13, [5, 6]),
])
def test_str_joins_terms_with_plus(make_poly, p, coeffs):
    text = str(make_poly(p, coeffs))
    terms = text.split(" + ")
    assert len(terms) == len(coeffs)
    assert terms[-1] == f"({coeffs[0] % p} mod {p})"
    assert terms[0].endswith(f"x^{len(coeffs) - 1}")


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4, 5]), (13, [6, 7, 8]),
])
def test_str_uses_element_str_representation(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert str(poly) == " + ".join(
        f"{F_pElement(p, c)}" + (f"x^{i}" if i else "")
        for i, c in reversed(list(enumerate(coeffs))))


# ===========================================================================
# 6. __repr__
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, bound, expected", [
    (5, [1, 2], None,
     "F_pPolynomial(5, None, coefficients=(F_pElement(p=5, value=1), "
     "F_pElement(p=5, value=2)))"),
    (5, [], 3, "F_pPolynomial(5, 3, coefficients=())"),
    (7, [], None, "F_pPolynomial(7, None, coefficients=())"),
    (2, [1], 1, "F_pPolynomial(2, 1, coefficients=(F_pElement(p=2, value=1),))"),
    (13, [0, 12], None,
     "F_pPolynomial(13, None, coefficients=(F_pElement(p=13, value=0), "
     "F_pElement(p=13, value=12)))"),
])
def test_repr_format(make_poly, p, coeffs, bound, expected):
    assert repr(make_poly(p, coeffs, bound)) == expected


@pytest.mark.parametrize("p", PRIMES)
def test_repr_contains_field_order(p):
    assert repr(F_pPolynomial(p)) == f"F_pPolynomial({p}, None, coefficients=())"


@pytest.mark.parametrize("bound", [0, 1, 2, 5, 12])
def test_repr_contains_degree_bound(make_poly, bound):
    assert repr(make_poly(5, [1], bound)).startswith(f"F_pPolynomial(5, {bound}, ")


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4, 5, 6]), (13, [0, 1]),
])
def test_repr_differs_from_str(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert repr(poly) != str(poly)
    assert "F_pPolynomial(" in repr(poly)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1]), (11, [9, 9]), (13, [1, 0, 2]),
])
def test_repr_of_equal_polynomials_is_equal(make_poly, p, coeffs):
    assert repr(make_poly(p, coeffs)) == repr(make_poly(p, coeffs))


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 0]), (7, [3, 0]), (2, [1, 0, 0]), (11, [4, 0]), (13, [6, 0, 0]),
])
def test_repr_shows_stripped_coefficients(make_poly, p, coeffs):
    # хвостовые нули отброшены, поэтому в repr их нет
    stripped = list(coeffs)
    while stripped and stripped[-1] % p == 0:
        stripped.pop()
    text = repr(make_poly(p, coeffs))
    assert text.count("F_pElement(") == len(stripped)
    assert text.count("value=0") == sum(1 for c in stripped if c % p == 0)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [5, 6]), (13, [7, 8, 9]),
])
def test_repr_of_list_of_polynomials(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert repr([poly]) == f"[{repr(poly)}]"


# ===========================================================================
# 7. __pos__
# ===========================================================================

@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (5, []), (7, [3]), (2, [1, 1]), (11, [1, 2, 3]), (13, [0, 1]),
])
def test_pos_returns_same_object(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert (+poly) is poly


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6]), (2, [1, 0, 1]), (11, [4, 5]), (13, [0, 0, 1]),
])
def test_pos_preserves_value(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert +poly == poly


@pytest.mark.parametrize("bound", [None, 0, 1, 3, 10])
def test_pos_preserves_degree_bound(make_poly, bound):
    assert (+make_poly(5, [1], bound)).degree_bound == bound


@pytest.mark.parametrize("p, coeffs", [
    (5, [1]), (7, [2, 3]), (2, [1]), (11, [0, 1]), (13, [4, 5, 6]),
])
def test_pos_is_idempotent(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert +(+(+poly)) is poly


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [7, 8]), (13, [1, 2, 3]),
])
def test_pos_does_not_change_str_degree_and_p(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert str(+poly) == str(poly)
    assert (+poly).degree == poly.degree
    assert (+poly).p == p


@pytest.mark.parametrize("p", PRIMES)
def test_pos_of_zero_polynomial(p):
    poly = F_pPolynomial(p)
    assert (+poly) is poly
    assert +poly == poly


# ===========================================================================
# 8. __neg__
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 2], [4, 3]),
    (5, [0], []),
    (5, [], []),
    (7, [3, 5], [4, 2]),
    (2, [1], [1]),
    (2, [1, 1], [1, 1]),
    (11, [10, 1], [1, 10]),
    (13, [0, 0, 1], [0, 0, 12]),
])
def test_neg_coefficients(make_poly, p, coeffs, expected):
    assert -make_poly(p, coeffs) == make_poly(p, expected)


@pytest.mark.parametrize("p", PRIMES)
def test_neg_of_zero_polynomial_is_zero_polynomial(p):
    negated = -F_pPolynomial(p)
    assert negated == F_pPolynomial(p)
    assert str(negated) == f"(0 mod {p})"
    assert negated.degree == float('-inf')


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [4, 5]), (2, [1, 0, 1]), (11, [9]), (13, [0, 1, 2]),
])
def test_double_negation_restores_polynomial(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert -(-poly) == poly
    assert str(-(-poly)) == str(poly)


@pytest.mark.parametrize("bound", [None, 0, 1, 4, 9])
def test_neg_preserves_degree_bound(make_poly, bound):
    assert (-make_poly(5, [1], bound)).degree_bound == bound


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1]), (11, [4, 5, 6]), (13, [7]),
])
def test_neg_preserves_field(make_poly, p, coeffs):
    assert (-make_poly(p, coeffs)).p == p


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [5]), (13, [6, 7]),
])
def test_neg_returns_new_object(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    negated = -poly
    assert negated is not poly
    assert poly == make_poly(p, coeffs)


@pytest.mark.parametrize("p, zeros", [
    (5, 1), (5, 3), (7, 2), (11, 4), (13, 5),
])
def test_neg_of_all_zero_coefficients_is_zero_polynomial(make_poly, p, zeros):
    assert -make_poly(p, [0] * zeros) == F_pPolynomial(p)


@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 2], "(3 mod 5)x^1 + (4 mod 5)"),
    (7, [3, 5], "(2 mod 7)x^1 + (4 mod 7)"),
    (2, [1, 1], "(1 mod 2)x^1 + (1 mod 2)"),
    (11, [10], "(1 mod 11)"),
    (13, [0, 1], "(12 mod 13)x^1 + (0 mod 13)"),
])
def test_neg_string_representation(make_poly, p, coeffs, expected):
    assert str(-make_poly(p, coeffs)) == expected


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4, 5]), (11, [6, 7]), (13, [8, 9, 10]), (2, [1, 0, 1]),
])
def test_neg_preserves_degree(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert (-poly).degree == poly.degree


# ===========================================================================
# 9. __add__
# ===========================================================================

@pytest.mark.parametrize("p, a, b, expected", [
    (5, [1, 2, 3], [4, 4], [0, 1, 3]),
    (5, [1, 2], [3, 4], [4, 1]),
    (5, [1, 1], [4, 4], []),
    (5, [3], [4], [2]),
    (5, [1, 0, 2], [4, 3], [0, 3, 2]),
    (5, [1, 2], [3], [4, 2]),
    (5, [3], [1, 2], [4, 2]),
    (7, [3, 5], [4, 2], []),
    (7, [2, 3, 4], [5, 4], [0, 0, 4]),
    (7, [6], [1], []),
    (11, [10, 10], [2, 2], [1, 1]),
    (13, [12], [1], []),
    (2, [1], [1], []),
    (2, [1, 0], [0, 1], [1, 1]),
    (3, [2, 2], [1, 1, 1], [0, 0, 1]),
])
def test_add_coefficients(make_poly, p, a, b, expected):
    assert make_poly(p, a) + make_poly(p, b) == make_poly(p, expected)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [0, 1]), (11, [10]), (13, [1, 0, 0, 2]), (2, [1, 1]), (3, [2]),
])
def test_add_zero_polynomial_is_identity(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    zero = F_pPolynomial(p)
    assert poly + zero == poly
    assert zero + poly == poly


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4, 1]),
    (7, [6, 1], [2]),
    (11, [1, 2, 3], [4, 5]),
    (13, [0, 1], [1, 0, 2]),
    (2, [1, 1], [1]),
    (3, [2, 2], [1, 1, 1]),
])
def test_add_is_commutative(make_poly, p, a, b):
    assert make_poly(p, a) + make_poly(p, b) == make_poly(p, b) + make_poly(p, a)
    assert str(make_poly(p, a) + make_poly(p, b)) == \
           str(make_poly(p, b) + make_poly(p, a))


@pytest.mark.parametrize("p, a, b, c", [
    (5, [1], [2], [3]),
    (5, [1, 2], [3, 4], [1, 1]),
    (7, [2, 3], [4], [5, 6, 1]),
    (11, [1, 2, 3], [9, 8], [7]),
    (13, [0, 1], [1, 0, 2], [3, 3]),
])
def test_add_is_associative(make_poly, p, a, b, c):
    left = (make_poly(p, a) + make_poly(p, b)) + make_poly(p, c)
    right = make_poly(p, a) + (make_poly(p, b) + make_poly(p, c))
    assert left == right


@pytest.mark.parametrize("p, a, b, expected_degree", [
    (5, [1, 2, 3], [4, 4], 2),
    (5, [1, 2], [3], 1),
    (7, [2, 3, 4], [5, 4], 2),
    (11, [1], [2, 3], 1),
    (13, [0, 1], [1], 1),
])
def test_add_degree_is_max_of_degrees(make_poly, p, a, b, expected_degree):
    assert (make_poly(p, a) + make_poly(p, b)).degree == expected_degree


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]), (7, [1], [2, 3]), (2, [1, 1], [1]), (11, [5, 6], [7]),
    (13, [1, 2, 3], [4, 5, 6]),
])
def test_add_returns_new_polynomial_of_same_field(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    result = first + second
    assert isinstance(result, F_pPolynomial)
    assert result is not first and result is not second
    assert result.p == p


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]), (7, [6, 1], [2]), (2, [1], [1]), (11, [3, 4], [5, 6]),
    (13, [1], [2, 3]),
])
def test_add_does_not_mutate_operands(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    _ = first + second
    assert first == make_poly(p, a)
    assert second == make_poly(p, b)


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 17), (3, 5),
])
def test_add_different_fields_raises_value_error(make_poly, p1, p2):
    with pytest.raises(ValueError,
                       match="Cannot add two polynomials in different fields"):
        make_poly(p1, [1]) + make_poly(p2, [1])


@pytest.mark.parametrize("b1, b2", [
    (2, 3), (0, 1), (5, 4), (1, 10), (3, 2), (7, 8),
])
def test_add_different_degree_bound_raises_value_error(make_poly, b1, b2):
    with pytest.raises(ValueError,
                       match="Cannot add two polynomials with different degree_bound"):
        make_poly(5, [1], b1) + make_poly(5, [1], b2)


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES)
def test_add_non_polynomial_raises_type_error(make_poly, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_poly(5, [1]) + other


@pytest.mark.parametrize("other", [1, 2, 5, 0, -3])
def test_int_plus_polynomial_raises_type_error(make_poly, other):
    with pytest.raises(TypeError):
        other + make_poly(5, [1])


@pytest.mark.parametrize("p", PRIMES[:5])
def test_add_with_subclass_raises_type_error(p):
    with pytest.raises(TypeError, match="unsupported operand type"):
        F_pPolynomial(p) + SubPolynomial(p)


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 1], [4, 4]),
    (5, [2], [3]),
    (7, [3, 5], [4, 2]),
    (13, [12], [1]),
    (2, [1, 0, 1], [1, 0, 1]),
])
def test_add_of_opposite_polynomials_is_zero(make_poly, p, a, b):
    result = make_poly(p, a) + make_poly(p, b)
    assert result == F_pPolynomial(p)
    assert str(result) == f"(0 mod {p})"


@pytest.mark.parametrize("bound", [0, 1, 2, 3, 7])
def test_add_preserves_equal_degree_bound(make_poly, bound):
    assert (make_poly(5, [1], bound) + make_poly(5, [2], bound)).degree_bound == bound


@pytest.mark.parametrize("bound", [0, 1, 2, 5, 11])
def test_add_degree_bound_becomes_none_if_any_operand_unbound(make_poly, bound):
    bounded = make_poly(5, [1], bound)
    unbounded = make_poly(5, [1, 2])
    assert (unbounded + bounded).degree_bound is None
    assert (bounded + unbounded).degree_bound is None


@pytest.mark.parametrize("p, a, bound", [
    (5, [1, 2], 2), (5, [1], 4), (7, [3, 4], 3), (11, [1, 2, 3], 5), (13, [9], 1),
])
def test_add_result_respects_degree_bound(make_poly, p, a, bound):
    result = make_poly(p, a, bound) + make_poly(p, a, bound)
    assert result.degree_bound == bound
    assert result.degree <= bound


# ===========================================================================
# 10. __sub__
# ===========================================================================

@pytest.mark.parametrize("p, a, b, expected", [
    (5, [1, 2], [3, 4], [3, 3]),
    (5, [3], [1, 2], [2, 3]),
    (5, [1, 2, 3], [1, 2, 3], []),
    (7, [6], [1], [5]),
    (7, [0], [1], [6]),
    (11, [3], [7], [7]),
    (13, [2], [5], [10]),
    (2, [1], [1], []),
    (2, [1, 1], [0, 1], [1]),
    (3, [0, 1], [1, 2], [2, 2]),
])
def test_sub_coefficients(make_poly, p, a, b, expected):
    assert make_poly(p, a) - make_poly(p, b) == make_poly(p, expected)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [4, 5]), (2, [1, 1]), (11, [9, 8, 7]), (13, [0, 1]),
])
def test_sub_self_is_zero_polynomial(make_poly, p, coeffs):
    result = make_poly(p, coeffs) - make_poly(p, coeffs)
    assert result == F_pPolynomial(p)
    assert result.degree == float('-inf')


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]), (7, [1, 2, 3], [4]), (2, [1, 1], [1]), (11, [5], [6, 7]),
    (13, [3, 4], [1, 2, 3]),
])
def test_sub_equals_add_of_negation(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    assert first - second == first + (-second)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 0]), (11, [4, 5, 6]), (13, [0, 1]),
])
def test_sub_zero_polynomial_is_identity(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly - F_pPolynomial(p) == poly


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [5, 6]), (13, [7, 8, 9]),
])
def test_zero_minus_polynomial_is_negation(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert F_pPolynomial(p) - poly == -poly


@pytest.mark.parametrize("p, a, b", [
    (5, [4, 1], [1, 3]), (7, [2, 5], [6, 1]), (11, [3, 9], [10, 2]),
    (13, [1, 6], [5, 12]), (2, [1, 0], [0, 1]),
])
def test_sub_is_anti_commutative(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    assert first - second == -(second - first)


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 17), (3, 5),
])
def test_sub_different_fields_raises_value_error(make_poly, p1, p2):
    # нюанс: __sub__ делегирует __add__, поэтому сообщение — про сложение
    with pytest.raises(ValueError,
                       match="Cannot add two polynomials in different fields"):
        make_poly(p1, [1]) - make_poly(p2, [1])


@pytest.mark.parametrize("b1, b2", [
    (2, 3), (0, 1), (5, 4), (1, 10), (3, 2), (7, 8),
])
def test_sub_different_degree_bound_raises_value_error(make_poly, b1, b2):
    with pytest.raises(ValueError,
                       match="Cannot add two polynomials with different degree_bound"):
        make_poly(5, [1], b1) - make_poly(5, [1], b2)


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES)
def test_sub_non_polynomial_raises_type_error(make_poly, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_poly(5, [1]) - other


@pytest.mark.parametrize("other", [1, 2, 5, 0])
def test_int_minus_polynomial_raises_type_error(make_poly, other):
    with pytest.raises(TypeError):
        other - make_poly(5, [1])


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]), (7, [6, 1], [2]), (2, [1], [1]), (11, [3, 4], [5, 6]),
    (13, [1], [2, 3]),
])
def test_sub_does_not_mutate_operands(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    _ = first - second
    assert first == make_poly(p, a)
    assert second == make_poly(p, b)


@pytest.mark.parametrize("bound", [0, 1, 2, 4, 8])
def test_sub_preserves_degree_bound(make_poly, bound):
    result = make_poly(5, [1], bound) - make_poly(5, [2], bound)
    assert result.degree_bound == bound


# ===========================================================================
# 11. Неподдерживаемые операции и прочие нюансы
# ===========================================================================

@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1]), (11, [5]), (13, [1, 2, 3]),
])
def test_multiplication_is_not_supported(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    with pytest.raises(TypeError):
        poly * poly
    with pytest.raises(TypeError):
        poly * 2
    with pytest.raises(TypeError):
        2 * poly


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4, 5]), (13, [6]),
])
def test_power_is_not_supported(make_poly, p, coeffs):
    with pytest.raises(TypeError):
        make_poly(p, coeffs) ** 2


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1]), (11, [4]), (13, [5, 6]),
])
def test_division_is_not_supported(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    with pytest.raises(TypeError):
        poly / poly
    with pytest.raises(TypeError):
        poly / 2


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [9]), (13, [1, 2]),
])
def test_floordiv_mod_matmul_are_not_supported(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    with pytest.raises(TypeError):
        poly // poly
    with pytest.raises(TypeError):
        poly % poly
    with pytest.raises(TypeError):
        poly @ poly


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1]), (11, [5, 6, 7]), (13, [8]),
])
def test_len_iteration_and_indexing_are_not_supported(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    with pytest.raises(TypeError):
        len(poly)
    with pytest.raises(TypeError):
        iter(poly)
    with pytest.raises(TypeError):
        poly[0]


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4]), (13, [5, 6]),
])
def test_abs_is_not_supported_and_bool_is_true(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    with pytest.raises(TypeError):
        abs(poly)
    # __bool__ отсутствует -> любой многочлен (даже нулевой) истинен
    assert bool(poly) is True


@pytest.mark.parametrize("p", PRIMES)
def test_zero_polynomial_is_truthy(p):
    assert bool(F_pPolynomial(p)) is True


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1]), (11, [4, 5]), (13, [6, 7]),
])
def test_unary_invert_is_not_supported(make_poly, p, coeffs):
    with pytest.raises(TypeError):
        ~make_poly(p, coeffs)


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]), (7, [1], [2]), (2, [1, 0], [0, 1]), (11, [3, 4], [5]),
    (13, [1, 2, 3], [4, 5]),
])
def test_sum_is_new_object_not_sharing_state(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    result = first + second
    assert result is not first
    assert result is not second
    assert first == make_poly(p, a)
    assert second == make_poly(p, b)


@pytest.mark.parametrize("prime", PRIMES)
def test_zero_polynomial_public_state(prime):
    p = prime
    poly = F_pPolynomial(p)
    assert poly.p == p
    assert poly.degree_bound is None
    assert poly.degree == float('-inf')
    assert str(poly) == f"(0 mod {p})"
    assert repr(poly) == f"F_pPolynomial({p}, None, coefficients=())"


@pytest.mark.parametrize("prime", PRIMES)
def test_unary_operations_on_zero_polynomial(prime):
    p = prime
    zero = F_pPolynomial(p)
    assert (-zero) == zero
    assert (+zero) is zero
    assert -(-zero) == zero


@pytest.mark.parametrize("prime", PRIMES)
def test_additive_laws_with_zero_polynomial(prime):
    p = prime
    zero = F_pPolynomial(p)
    poly = F_pPolynomial(p, None, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    assert poly + zero == poly
    assert zero + poly == poly
    assert poly + (-poly) == zero
    assert poly - poly == zero
