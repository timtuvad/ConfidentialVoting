"""Тесты модуля src/core/algebra/polynomial.py — класс F_pPolynomial (над F_p).

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.
Сравнение выполняется через публичный API (==, !=, hash(), str(), repr(),
свойства p / degree / degree_bound) — без обращения к приватным полям.

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__       — все валидные комбинации (p, degree_bound, coefficients)
                       и все исключения (ValueError);
    * p, degree, degree_bound — свойства;
    * random         — случайный многочлен заданной степени (с фиксированным
                       свободным членом least_coefficient или без него);
    * __eq__, __ne__, __hash__ — сравнение и хешируемость;
    * __str__        — печать "(k mod p)x^i", дополнение нулями до degree_bound;
    * __repr__       — отладочное представление (кортеж коэффициентов);
    * __pos__, __neg__ — унарные операции;
    * __add__, __sub__ — сложение/вычитание, их исключения и проверка типов;
    * __call__       — вычисление значения многочлена в точке поля;
    * share_at       — доля секрета SecretShare в точке x;
    * служебные нюансы — отсутствие умножения/деления/степени, len(), итерации,
      индексации, __bool__; неизменяемость операндов и переданного списка.

Особенности реализации, зафиксированные тестами:
    * коэффициенты хранятся неизменяемым кортежем с отброшенными хвостовыми нулями
      (ведущие и средние нули сохраняются, поэтому degree их учитывает);
    * переданный список коэффициентов не изменяется (внутри делается копия);
    * degree_bound обязан быть неотрицательным (ValueError
      "degree_bound must be non-negative"), проверка выполняется раньше,
      чем «p or coefficients must be provided», но позже проверки простоты p;
    * degree_bound допускается без coefficients — получается нулевой многочлен,
      а str() дополняется нулевыми слагаемыми до degree_bound;
    * соответствие degree_bound длине списка проверяется по ИСХОДНОМУ списку,
      то есть до отбрасывания хвостовых нулей;
    * degree_bound участвует в __eq__/__hash__: многочлены с разной границей
      степени не равны даже при одинаковых коэффициентах;
    * __eq__ и арифметика используют строгую проверку type(self) is type(other):
      подкласс не равен родителю, а арифметика с подклассом даёт TypeError;
    * операции с объектами других типов дают TypeError
      ("unsupported operand type(s) ..."), а для разных полей — ValueError;
    * степень нулевого многочлена равна float('-inf');
    * умножение, деление, возведение в степень, len(), итерация, индексация
      и abs() для многочленов не поддерживаются;
    * __call__ для нулевого многочлена не проверяет тип аргумента
      (цикл по коэффициентам не выполняется).

Доля секрета SecretShare (src/core/secret_share.py), возвращаемая share_at:
    * публичные свойства x и value — точка и значение многочлена в ней;
    * __eq__/__ne__ со строгой проверкой типа и __hash__ по паре (x, value),
      поэтому доли можно сравнивать, класть в множества и использовать как ключи;
    * конструктор бросает ValueError("x and value must be in the same field"),
      если точка и значение принадлежат разным полям;
    * str() -> "?(x)=value", repr() -> "SecretShare(x, value)".
"""

import pytest

from src import F_pElement, F_pPolynomial, SecretShare

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

PRIMES = [2, 3, 5, 7, 11, 13, 101]

SMALL_PRIMES = [2, 3, 5, 7, 11, 13]

NON_PRIMES = [-5, 0, 1, 4, 6, 8, 9, 10, 12, 15, 21, 25, 100, 589423]

NON_POLYNOMIAL_VALUES = [0, 5, "x", None, 3.5, (1, 2), [1], object()]


class SubPolynomial(F_pPolynomial):
    """Подкласс многочлена для проверки type(self) is type(other)."""


class SubElement(F_pElement):
    """Подкласс элемента поля для проверки type(self) is type(other)."""


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


@pytest.mark.parametrize("bound", [-1, -2, -7])
def test_init_negative_degree_bound_raises_value_error(bound):
    with pytest.raises(ValueError, match="degree_bound must be non-negative"):
        F_pPolynomial(5, bound, coefficients=[])


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
def test_init_degree_bound_without_coefficients_creates_zero_polynomial(bound):
    poly = F_pPolynomial(5, bound)
    assert poly.p == 5
    assert poly.degree_bound == bound
    assert poly.degree == float('-inf')
    assert (poly ==
            F_pPolynomial(5, bound, coefficients=[
                F_pElement.zero(5) for _ in range(bound + 1)
            ]))


@pytest.mark.parametrize("bound", [-1, -2, -3, -5, -10, -100])
def test_init_negative_degree_bound_message(bound):
    with pytest.raises(ValueError, match="degree_bound must be non-negative"):
        F_pPolynomial(5, bound, coefficients=[])


@pytest.mark.parametrize("kwargs", [
    {"degree_bound": -1},
    {"degree_bound": -2, "coefficients": []},
    {"p": 5, "degree_bound": -5},
    {"p": None, "degree_bound": -3},
    {"p": 5, "degree_bound": -1, "coefficients": None},
    {"p": 7, "degree_bound": -4, "coefficients": [F_pElement(7, 1)]},
])
def test_init_negative_degree_bound_checked_before_missing_arguments(kwargs):
    # нюанс: проверка degree_bound < 0 выполняется раньше, чем
    # «p or coefficients must be provided»
    with pytest.raises(ValueError, match="degree_bound must be non-negative"):
        F_pPolynomial(**kwargs)


@pytest.mark.parametrize("p", [4, 6, 9, 15, 100])
def test_init_non_prime_p_checked_before_degree_bound(p):
    # нюанс: непростое p важнее отрицательного degree_bound
    with pytest.raises(ValueError, match="p must be prime"):
        F_pPolynomial(p, -1, coefficients=[])


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
    assert str(F_pPolynomial(p, bound, coefficients=[])) == expected
    # то же самое без явной передачи списка коэффициентов
    assert str(F_pPolynomial(p, bound)) == expected


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
def test_str_of_zero_polynomial_created_without_coefficients(p):
    assert str(F_pPolynomial(p)) == f"(0 mod {p})"
    assert str(F_pPolynomial(p, 0)) == f"(0 mod {p})"
    assert str(F_pPolynomial(p, 1)) == f"(0 mod {p})x^1 + (0 mod {p})"
    assert str(F_pPolynomial(p, 2)) == (
        f"(0 mod {p})x^2 + (0 mod {p})x^1 + (0 mod {p})")


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


# ===========================================================================
# 12. __call__ — вычисление значения многочлена в точке
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, x, expected", [
    (5, [3], 0, 3),
    (5, [3], 4, 3),
    (5, [1, 2], 3, 2),          # 1 + 2*3 = 7 = 2 (mod 5)
    (5, [1, 2, 3], 2, 2),       # 1 + 4 + 12 = 17 = 2 (mod 5)
    (5, [1, 2, 3], 1, 1),       # 1 + 2 + 3 = 6 = 1 (mod 5)
    (5, [0, 0, 1], 3, 4),       # 9 = 4 (mod 5)
    (7, [6, 5, 4], 2, 4),       # 6 + 10 + 16 = 32 = 4 (mod 7)
    (7, [0, 1], 0, 0),
    (11, [1, 2, 3], 3, 1),      # 1 + 6 + 27 = 34 = 1 (mod 11)
    (11, [10, 10, 10], 10, 10),  # 1110 = 10 (mod 11)
    (13, [12, 1], 12, 11),      # 12 + 12 = 24 = 11 (mod 13)
    (13, [1, 0, 0, 1], 3, 2),   # 1 + 27 = 28 = 2 (mod 13)
    (2, [1, 1, 1], 1, 1),       # 3 = 1 (mod 2)
    (3, [2, 2, 2], 2, 2),       # 2 + 4 + 8 = 14 = 2 (mod 3)
    (101, [1, 2], 50, 0),       # 1 + 100 = 101 = 0 (mod 101)
])
def test_call_computes_polynomial_value(make_poly, p, coeffs, x, expected):
    assert make_poly(p, coeffs)(F_pElement(p, x)) == F_pElement(p, expected)


@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 2, 3], 1),
    (7, [6, 5], 6),
    (2, [1, 1, 1], 1),
    (11, [10], 10),
    (13, [0, 1], 0),
    (101, [50, 60], 50),
])
def test_call_at_zero_returns_least_coefficient(make_poly, p, coeffs, expected):
    poly = make_poly(p, coeffs)
    assert poly(F_pElement.zero(p)) == F_pElement(p, expected)


@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 2, 3], 1),
    (7, [6, 5], 4),
    (2, [1, 1, 1], 1),
    (11, [10], 10),
    (13, [0, 1], 1),
    (101, [50, 60], 9),
])
def test_call_at_one_returns_sum_of_coefficients(make_poly, p, coeffs, expected):
    poly = make_poly(p, coeffs)
    assert poly(F_pElement.unit(p)) == F_pElement(p, expected)
    assert poly(F_pElement(p, 1)) == F_pElement(p, sum(coeffs) % p)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5]), (2, [1, 1]), (11, [10, 1, 2]), (13, [0, 1]),
])
def test_call_returns_element_of_same_field(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(min(p, 5)):
        result = poly(F_pElement(p, value))
        assert isinstance(result, F_pElement)
        assert result.p == p
        assert 0 <= result.value < p


@pytest.mark.parametrize("prime", PRIMES)
def test_call_of_zero_polynomial_is_zero(prime):
    p = prime
    poly = F_pPolynomial(p)
    for value in range(min(p, 5)):
        assert poly(F_pElement(p, value)) == F_pElement.zero(p)


@pytest.mark.parametrize("argument", [3, 0, None, "1", 2.5, [1]])
def test_call_of_zero_polynomial_ignores_argument_type(argument):
    # нюанс: у нулевого многочлена цикл не выполняется, тип x не проверяется
    assert F_pPolynomial(5)(argument) == F_pElement.zero(5)


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 6), (2, 1), (11, 10), (13, 12),
])
def test_call_of_constant_polynomial(make_poly, p, value):
    poly = make_poly(p, [value])
    for argument in range(min(p, 5)):
        assert poly(F_pElement(p, argument)) == F_pElement(p, value)


@pytest.mark.parametrize("p, x, expected", [
    (5, 0, 0), (5, 1, 1), (5, 2, 4), (5, 3, 4), (5, 4, 1),
    (7, 3, 2), (7, 6, 1),
])
def test_call_of_monomial_x_squared(make_poly, p, x, expected):
    poly = make_poly(p, [0, 0, 1])
    assert poly(F_pElement(p, x)) == F_pElement(p, expected)


@pytest.mark.parametrize("prime", PRIMES)
def test_call_of_linear_polynomial(prime):
    # f(x) = 1 + 2x
    p = prime
    poly = F_pPolynomial(
        p, None, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    for value in range(min(p, 7)):
        assert poly(F_pElement(p, value)) == F_pElement(p, (1 + 2 * value) % p)


@pytest.mark.parametrize("p, a, b, x", [
    (5, [1, 2], [3, 4], 2),
    (7, [6, 1], [2, 3], 5),
    (11, [1, 2, 3], [9, 8], 4),
    (13, [0, 1], [1, 0, 2], 7),
    (2, [1, 1], [1], 1),
])
def test_call_is_additive(make_poly, p, a, b, x):
    point = F_pElement(p, x)
    left = (make_poly(p, a) + make_poly(p, b))(point)
    right = make_poly(p, a)(point) + make_poly(p, b)(point)
    assert left == right


@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2, 3], 4), (7, [6, 5], 2), (11, [10, 1], 9), (13, [1, 2], 5),
    (2, [1, 1], 1),
])
def test_call_respects_negation(make_poly, p, coeffs, x):
    point = F_pElement(p, x)
    poly = make_poly(p, coeffs)
    assert (-poly)(point) == -(poly(point))


@pytest.mark.parametrize("p, coeffs, bound", [
    (5, [1, 2], 4), (5, [1, 2], 2), (7, [3], 5), (11, [1, 2, 3], 7), (13, [9], 3),
])
def test_call_ignores_degree_bound(make_poly, p, coeffs, bound):
    bounded = make_poly(p, coeffs, bound)
    unbounded = make_poly(p, coeffs)
    for value in range(min(p, 5)):
        point = F_pElement(p, value)
        assert bounded(point) == unbounded(point)


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES)
def test_call_with_non_element_raises_type_error(make_poly, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_poly(5, [1, 2])(other)


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101),
])
def test_call_with_element_of_other_field_raises_type_error(make_poly, p, other_p):
    with pytest.raises(TypeError, match="cannot *"):
        make_poly(p, [1, 2])(F_pElement(other_p, 1))


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_call_with_subclass_element_raises_type_error(make_poly, p, value):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_poly(p, [1, 2])(SubElement(p, value))


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5]), (2, [1, 1]), (11, [10, 1]), (13, [1, 2, 3]),
])
def test_call_does_not_mutate_polynomial(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(min(p, 5)):
        poly(F_pElement(p, value))
    assert poly == make_poly(p, coeffs)
    assert str(poly) == str(make_poly(p, coeffs))
    assert poly.degree == make_poly(p, coeffs).degree


@pytest.mark.parametrize("p, coeffs", [
    (2, [1, 1]), (3, [2, 0, 1]), (5, [1, 2, 3]), (7, [4, 0, 3]), (11, [1, 2]),
])
def test_call_matches_manual_sum_over_all_points(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(p):
        x = F_pElement(p, value)
        expected = F_pElement.zero(p)
        for i, coefficient in enumerate(coeffs):
            expected = expected + F_pElement(p, coefficient) * (x ** i)
        assert poly(x) == expected


@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2], 3), (7, [3, 4], 5), (11, [9, 1], 2), (13, [5, 6], 7),
    (101, [50, 51], 100),
])
def test_call_result_supports_field_arithmetic(make_poly, p, coeffs, x):
    value = make_poly(p, coeffs)(F_pElement(p, x))
    assert isinstance(value + value, F_pElement)
    assert isinstance(value * value, F_pElement)
    assert isinstance(-value, F_pElement)
    assert isinstance(value ** 3, F_pElement)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5]), (11, [10, 1, 2]), (13, [1, 0, 2]), (101, [7, 8]),
])
def test_call_is_deterministic(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(min(p, 5)):
        point = F_pElement(p, value)
        assert poly(point) == poly(point)
        assert poly(point) == make_poly(p, coeffs)(point)


@pytest.mark.parametrize("prime", PRIMES)
def test_call_of_zero_polynomial_with_degree_bound(prime):
    p = prime
    poly = F_pPolynomial(p, 3)
    assert poly(F_pElement.zero(p)) == F_pElement.zero(p)
    assert poly(F_pElement.unit(p)) == F_pElement.zero(p)


# ===========================================================================
# 13. random(p, degree_bound[, least_coefficient]) — случайный многочлен
# ===========================================================================

@pytest.mark.parametrize("p, bound", [
    (5, 1), (5, 3), (7, 2), (11, 4), (13, 0), (101, 5), (2, 2),
])
def test_random_returns_polynomial_of_field(p, bound):
    poly = F_pPolynomial.random(p, bound)
    assert isinstance(poly, F_pPolynomial)
    assert poly.p == p


@pytest.mark.parametrize("bound", [0, 1, 2, 3, 5, 9])
def test_random_stores_degree_bound(bound):
    assert F_pPolynomial.random(101, bound).degree_bound == bound


@pytest.mark.parametrize("p, bound", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5), (2, 1),
])
def test_random_degree_does_not_exceed_degree_bound(p, bound):
    for _ in range(10):
        poly = F_pPolynomial.random(p, bound)
        assert poly.degree <= bound


@pytest.mark.parametrize("p, bound, secret", [
    (5, 1, 3), (7, 2, 6), (11, 3, 10), (13, 0, 12), (101, 4, 42), (2, 2, 1),
])
def test_random_uses_provided_least_coefficient(p, bound, secret):
    least = F_pElement(p, secret)
    poly = F_pPolynomial.random(p, bound, least_coefficient=least)
    assert poly(F_pElement.zero(p)) == least


@pytest.mark.parametrize("p, bound, secret", [
    (5, 2, 1), (7, 3, 5), (11, 1, 9), (13, 2, 7), (101, 3, 33),
])
def test_random_keeps_least_coefficient_across_calls(p, bound, secret):
    least = F_pElement(p, secret)
    zero = F_pElement.zero(p)
    values = {F_pPolynomial.random(p, bound, least_coefficient=least)(zero).value
              for _ in range(20)}
    assert values == {secret % p}


@pytest.mark.parametrize("p", [5, 7, 11, 13, 101])
def test_random_with_zero_least_coefficient(p):
    poly = F_pPolynomial.random(p, 2, least_coefficient=F_pElement.zero(p))
    assert poly(F_pElement.zero(p)) == F_pElement.zero(p)


@pytest.mark.parametrize("p, bound", [
    (5, 2), (7, 3), (11, 2), (13, 4), (101, 3),
])
def test_random_without_least_coefficient_has_random_constant_term(p, bound):
    zero = F_pElement.zero(p)
    values = {F_pPolynomial.random(p, bound)(zero).value for _ in range(40)}
    assert values <= set(range(p))
    if p > 3:
        assert len(values) > 1


@pytest.mark.parametrize("p, bound", [
    (11, 3), (13, 2), (101, 4), (7, 3), (5, 2),
])
def test_random_produces_different_polynomials(p, bound):
    samples = {str(F_pPolynomial.random(p, bound)) for _ in range(30)}
    assert len(samples) > 1


@pytest.mark.parametrize("bound", [-1, -2, -3, -5, -10])
def test_random_negative_degree_bound_raises_value_error(bound):
    with pytest.raises(ValueError, match="degree_bound must be non-negative"):
        F_pPolynomial.random(5, bound)


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101), (101, 5),
])
def test_random_least_coefficient_from_other_field_raises_value_error(p, other_p):
    with pytest.raises(
            ValueError,
            match=f"least_coefficient must be from the field of order {p}"):
        F_pPolynomial.random(p, 2, least_coefficient=F_pElement(other_p, 1))


@pytest.mark.parametrize("p", [4, 6, 9, 15, 100, 1])
def test_random_non_prime_p_raises_value_error(p):
    with pytest.raises(ValueError, match="p must be prime"):
        F_pPolynomial.random(p, 2)


@pytest.mark.parametrize("p", PRIMES)
def test_random_with_degree_bound_zero_is_constant(p):
    poly = F_pPolynomial.random(p, 0)
    assert poly.degree_bound == 0
    assert poly.degree <= 0
    assert "x^" not in str(poly)
    assert poly(F_pElement.zero(p)) == poly(F_pElement.unit(p))


@pytest.mark.parametrize("p, bound", [
    (5, 1), (7, 2), (11, 3), (13, 1), (101, 2),
])
def test_random_returns_new_object_each_call(p, bound):
    assert F_pPolynomial.random(p, bound) is not F_pPolynomial.random(p, bound)


@pytest.mark.parametrize("p, bound", [
    (5, 2), (7, 3), (11, 2), (13, 4), (101, 3),
])
def test_random_evaluation_gives_elements_of_same_field(p, bound):
    poly = F_pPolynomial.random(p, bound)
    for value in range(min(p, 6)):
        result = poly(F_pElement(p, value))
        assert isinstance(result, F_pElement)
        assert result.p == p


@pytest.mark.parametrize("p, secret", [
    (5, 3), (7, 6), (11, 10), (13, 1), (101, 42),
])
def test_random_share_at_zero_equals_secret(p, secret):
    least = F_pElement(p, secret)
    poly = F_pPolynomial.random(p, 3, least_coefficient=least)
    zero = F_pElement.zero(p)
    share = poly.share_at(zero)
    assert share.x == zero
    assert share.value == least
    assert share == SecretShare(zero, least)


# ===========================================================================
# 14. share_at(x) — доля секрета SecretShare в точке x
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2, 3], 0),
    (5, [1, 2, 3], 4),
    (7, [6, 5], 3),
    (11, [10], 7),
    (13, [0, 1], 12),
    (2, [1, 1], 1),
])
def test_share_at_returns_secret_share(make_poly, p, coeffs, x):
    share = make_poly(p, coeffs).share_at(F_pElement(p, x))
    assert isinstance(share, SecretShare)


@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2, 3], 1),
    (5, [4], 2),
    (7, [6, 5], 3),
    (11, [10, 1], 5),
    (13, [0, 1], 12),
])
def test_share_at_holds_polynomial_value(make_poly, p, coeffs, x):
    poly = make_poly(p, coeffs)
    point = F_pElement(p, x)
    share = poly.share_at(point)
    assert share.x == point
    assert share.value == poly(point)
    assert share == SecretShare(point, poly(point))
    assert share != SecretShare(point, poly(point) + F_pElement.unit(p))


@pytest.mark.parametrize("p, coeffs, x, expected_value", [
    (5, [1, 2], 3, 2),
    (5, [3], 4, 3),
    (7, [6, 5, 4], 2, 4),
    (11, [1, 2, 3], 3, 1),
    (13, [12, 1], 12, 11),
])
def test_share_at_value_matches_expected(
        make_poly, p, coeffs, x, expected_value):
    share = make_poly(p, coeffs).share_at(F_pElement(p, x))
    assert share == SecretShare(F_pElement(p, x), F_pElement(p, expected_value))
    assert share.x == F_pElement(p, x)
    assert share.value == F_pElement(p, expected_value)


@pytest.mark.parametrize("prime", PRIMES)
def test_share_at_zero_returns_least_coefficient(prime):
    p = prime
    least = F_pElement(p, 1)
    poly = F_pPolynomial(
        p, None, coefficients=[least, F_pElement(p, 2), F_pElement(p, 3)])
    zero = F_pElement.zero(p)
    assert poly.share_at(zero) == SecretShare(zero, least)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (11, [9, 8, 7]), (13, [1, 0, 2]), (2, [1, 1]),
])
def test_share_at_different_points_give_different_shares(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    first = poly.share_at(F_pElement(p, 1))
    second = poly.share_at(F_pElement(p, 2))
    assert first != second


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5]), (2, [1, 1]), (11, [10, 1]), (13, [1, 2, 3]),
])
def test_share_at_does_not_mutate_polynomial(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(min(p, 5)):
        poly.share_at(F_pElement(p, value))
    assert poly == make_poly(p, coeffs)
    assert poly.degree == make_poly(p, coeffs).degree


@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2], 1), (7, [3], 2), (11, [9, 8], 3), (13, [5], 4), (101, [1, 2], 5),
])
def test_share_at_returns_new_object_each_call(make_poly, p, coeffs, x):
    poly = make_poly(p, coeffs)
    point = F_pElement(p, x)
    assert poly.share_at(point) is not poly.share_at(point)


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES)
def test_share_at_with_non_element_raises_type_error(make_poly, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_poly(5, [1, 2]).share_at(other)


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101),
])
def test_share_at_with_element_of_other_field_raises_type_error(
        make_poly, p, other_p):
    with pytest.raises(TypeError, match="cannot *"):
        make_poly(p, [1, 2]).share_at(F_pElement(other_p, 1))


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_share_at_with_subclass_element_raises_type_error(make_poly, p, value):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_poly(p, [1, 2]).share_at(SubElement(p, value))


@pytest.mark.parametrize("prime", PRIMES)
def test_share_at_of_zero_polynomial(prime):
    p = prime
    poly = F_pPolynomial(p)
    point = F_pElement.unit(p)
    share = poly.share_at(point)
    assert isinstance(share, SecretShare)
    assert share == SecretShare(point, F_pElement.zero(p))
    assert share.value == F_pElement.zero(p)


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 6), (11, 10), (13, 12), (2, 1),
])
def test_share_at_of_constant_polynomial_has_same_value(make_poly, p, value):
    poly = make_poly(p, [value])
    points = [F_pElement(p, v) for v in range(min(p, 4))]
    shares = [poly.share_at(point) for point in points]
    expected = [SecretShare(point, F_pElement(p, value)) for point in points]
    assert shares == expected


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (11, [9, 8]), (13, [5, 6]), (101, [1, 2]),
])
def test_share_at_result_is_hashable(make_poly, p, coeffs):
    share = make_poly(p, coeffs).share_at(F_pElement(p, 1))
    assert isinstance(hash(share), int)
    assert len({share, share}) == 1


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5]), (11, [10, 1, 2]), (13, [1, 0, 2]), (101, [7, 8]),
])
def test_share_at_agrees_with_call(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(min(p, 5)):
        point = F_pElement(p, value)
        assert poly.share_at(point) == SecretShare(point, poly(point))


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4, 5]), (11, [9, 8]), (13, [1, 0, 2]), (101, [7, 8, 9]),
])
def test_share_at_of_equal_polynomials_is_equal(make_poly, p, coeffs):
    first = make_poly(p, coeffs)
    second = make_poly(p, coeffs)
    point = F_pElement.unit(p)
    assert first.share_at(point) == second.share_at(point)
    assert hash(first.share_at(point)) == hash(second.share_at(point))


@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2, 3], 2), (7, [6, 5], 4), (11, [10, 1], 9), (13, [1, 0, 2], 5),
    (101, [7, 8], 100),
])
def test_share_at_exposes_point_and_value(make_poly, p, coeffs, x):
    poly = make_poly(p, coeffs)
    point = F_pElement(p, x)
    share = poly.share_at(point)
    assert share.x == point
    assert share.x.p == p
    assert share.value == poly(point)
    assert share.value.p == p


@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2], 3), (7, [3, 4], 5), (11, [9, 1], 2), (13, [5, 6], 7),
    (101, [1, 2], 50),
])
def test_share_at_string_representations(make_poly, p, coeffs, x):
    poly = make_poly(p, coeffs)
    point = F_pElement(p, x)
    share = poly.share_at(point)
    assert str(share) == f"?({point})={poly(point)}"
    assert repr(share) == f"SecretShare({point}, {poly(point)})"


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (11, [9, 8]), (13, [5, 6]), (101, [7, 8]),
])
def test_share_at_value_matches_manual_evaluation(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(min(p, 5)):
        point = F_pElement(p, value)
        manual = F_pElement.zero(p)
        for i, coefficient in enumerate(coeffs):
            manual = manual + F_pElement(p, coefficient) * (point ** i)
        assert poly.share_at(point).value == manual


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5]), (11, [10, 1, 2]), (13, [1, 0, 2]), (101, [7, 8]),
])
def test_share_at_distinct_points_have_distinct_x(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    points = [F_pElement(p, value) for value in range(min(p, 5))]
    shares = [poly.share_at(point) for point in points]
    assert {share.x for share in shares} == set(points)
    assert len({share.x for share in shares}) == len(points)


@pytest.mark.parametrize("prime", PRIMES)
def test_share_at_always_produces_same_field_share(prime):
    p = prime
    poly = F_pPolynomial(
        p, None, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    for value in range(min(p, 5)):
        share = poly.share_at(F_pElement(p, value))
        assert share.x.p == p
        assert share.value.p == p


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (11, [9, 8]), (13, [5, 6]), (101, [7, 8]),
])
def test_share_at_equality_is_reflexive_and_symmetric(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    point = F_pElement(p, 1)
    first = poly.share_at(point)
    second = poly.share_at(point)
    assert first == first
    assert first == second
    assert second == first
    assert not (first != second)
    assert (first == 5) is False
    assert (first != 5) is True


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5]), (11, [10, 1]), (13, [1, 0, 2]), (2, [1, 1]),
])
def test_share_at_supports_set_and_dict(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    shares = [poly.share_at(F_pElement(p, value)) for value in range(min(p, 4))]
    mapping = {share: index for index, share in enumerate(shares)}
    for index, share in enumerate(shares):
        assert mapping[poly.share_at(share.x)] == index
    assert len({poly.share_at(share.x) for share in shares}) == len(shares)
