"""Тесты модуля src/core/algebra/polynomial.py — класс F_pPolynomial (над F_p).

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.
Проверки выполняются через публичный API (==, !=, hash(), bool(), repr(),
str()/str_format(), индексация poly[i], свойства p / degree) — без обращения
к приватным полям.

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__       — все валидные комбинации (p, coefficients) и все исключения;
                       принимает любой итерируемый объект с коэффициентами;
    * p, degree, coefficients — свойства;
    * __getitem__    — доступ к коэффициенту по индексу;
    * __eq__, __ne__, __hash__ — сравнение и хешируемость;
    * str_format(), __str__ — печать "(k mod p)x^i" и дополнение нулями
                       до явно переданной границы степени;
    * __repr__       — отладочное представление (кортеж коэффициентов);
    * __pos__, __neg__ — унарные операции;
    * __add__, __sub__, __mul__ — сложение, вычитание, умножение и их исключения;
    * one            — константа 1 (мультипликативная единица кольца);
    * __mul__/__truediv__ с элементом поля — умножение и деление на константу,
                       включая __rmul__;
    * __bool__       — истинность: нулевой многочлен ложен;
    * random         — случайный многочлен заданной степени (с фиксированным
                       свободным членом least_coefficient или без него);
    * __call__       — вычисление значения многочлена в точке поля;
    * share_at       — удалённый метод: бросает NotImplementedError и отправляет
                       к статическому методу Share.make(polynomial, x);
    * служебные нюансы — неизменяемость операндов и переданного списка.

Особенности реализации, зафиксированные тестами:
    * коэффициенты хранятся неизменяемым кортежем с отброшенными хвостовыми нулями
      (ведущие и средние нули сохраняются, поэтому degree их учитывает);
    * конструктор принимает любой итерируемый объект (список, кортеж,
      генератор, map/filter/zip) — вход приводится к списку и копируется;
    * публичное свойство coefficients отдаёт кортеж коэффициентов;
    * переданный список коэффициентов не изменяется (внутри делается копия);
    * коэффициенты из другого поля дают TypeError
      "coefficients must be from the field of order p";
    * __eq__ и арифметика проверяют типы операндов через isinstance;
    * арифметика с объектами других типов даёт TypeError
      ("unsupported operand type(s) for + ..."), с многочленами из другого поля —
      TypeError ("cannot + with two polynomials in different fields: F_p and F_q");
    * степень нулевого многочлена равна float('-inf');
    * __bool__: нулевой многочлен ложен, ненулевой истинен;
    * __getitem__: индекс вне степени даёт ноль поля, отрицательный индекс —
      IndexError, нецелочисленный — TypeError; bool принимается (подкласс int);
    * __iter__ отсутствует, но из-за __getitem__ объект итерируем по старому
      протоколу, причём итерация не ограничена (IndexError никогда не возникает);
    * str_format(degree_bound) при границе МЕНЬШЕ степени обрезает старшие
      слагаемые, при отрицательной границе возвращает пустую строку;
    * __mul__ и __truediv__ с элементом поля дают умножение/деление всех
      коэффициентов на константу (деление — умножение на обратный элемент);
    * __call__ проверяет аргумент ДО цикла: объект не-F_pElement даёт
      TypeError "polynomial argument must be a F_pElement, not '<тип>'",
      а элемент другого поля — TypeError "polynomial argument must be in the
      same field as the polynomial, not F_p" (в том числе для нулевого
      многочлена).

Доля секрета Share (src/core/share.py) теперь создаётся через Share.make(poly, x):
    * Share.make(f, x) == Share(x, f(x)) — точка и значение многочлена в ней;
    * публичные свойства x, value и p; арифметика — только + и -;
    * подробные тесты класса Share находятся в tests/core/test_share.py.

Строковое представление строится через __getitem__, поэтому выход за степень
даёт нулевые слагаемые, а str_format(отрицательное) — пустую строку.
Сообщения об ошибках для разных полей дополнены именами полей:
"cannot + with two polynomials in different fields: F_5 and F_7".

Проверки типов в арифметике (__check_types и ветка с F_pElement в __mul__),
а также в __eq__/__ne__ выполняются через isinstance.

Умножение и деление многочлена на элемент поля работают; выражение
`element * poly` тоже работает, поскольку F_pElement.__mul__ для объекта
другого типа выполняет `other * self`.
"""

import pytest

from src import F_pElement, F_pPolynomial, Share

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

PRIMES = [2, 3, 5, 7, 11, 13, 101]

SMALL_PRIMES = [2, 3, 5, 7, 11, 13]

NON_PRIMES = [-5, 0, 1, 4, 6, 8, 9, 10, 12, 15, 21, 25, 100, 589423]

NON_POLYNOMIAL_VALUES = [0, 5, "x", None, 3.5, (1, 2), [1], object()]


# ---------------------------------------------------------------------------
# Вспомогательные функции (только публичный API: bool(), degree, poly[i])
# ---------------------------------------------------------------------------

def _coefficients(poly):
    """Список коэффициентов многочлена в порядке возрастания степеней."""
    if not poly:
        return []
    return [poly[index] for index in range(int(poly.degree) + 1)]


def _values(poly):
    """Численные значения коэффициентов многочлена."""
    return [coefficient.value for coefficient in _coefficients(poly)]


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
    def _make(p, coefficients):
        return F_pPolynomial(
            p, coefficients=[F_pElement(p, c) for c in coefficients])
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
    assert poly.degree == float('-inf')
    assert bool(poly) is False
    assert _values(poly) == []


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
    assert poly == F_pPolynomial(p, coefficients=[F_pElement(p, c) for c in coeffs])
    assert _values(poly) == [c % p for c in coeffs]


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [6, 5, 4]), (2, [1, 0, 1]), (11, [1, 2, 3, 4]),
    (13, [12, 11]), (101, [1, 2, 3]),
])
def test_init_with_p_and_coefficients(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly.p == p
    assert poly.degree == len(coeffs) - 1
    assert _values(poly) == coeffs


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1]), (11, [5, 6]), (13, [7, 8, 9]),
])
def test_init_second_positional_argument_is_coefficients(p, coeffs):
    elements = [F_pElement(p, c) for c in coeffs]
    assert F_pPolynomial(p, elements) == F_pPolynomial(p, coefficients=elements)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [5, 6, 7]), (13, [1, 0, 1]),
])
def test_init_accepts_keyword_arguments(p, coeffs):
    poly = F_pPolynomial(p=p, coefficients=[F_pElement(p, c) for c in coeffs])
    assert poly == F_pPolynomial(coefficients=[F_pElement(p, c) for c in coeffs])
    assert poly.p == p


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [1, 2, 3, 4]), (2, [1, 1, 1]), (11, [9, 8]), (13, [5, 4, 3]),
])
def test_init_stores_coefficients_as_f_p_elements(p, coeffs):
    poly = F_pPolynomial(p, coefficients=[F_pElement(p, c) for c in coeffs])
    for index, value in enumerate(coeffs):
        assert isinstance(poly[index], F_pElement)
        assert poly[index].p == p
        assert poly[index].value == value % p


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
    assert _values(make_poly(p, coeffs)) == expected


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
    F_pPolynomial(p, coefficients=elements)
    assert elements == snapshot
    assert len(elements) == len(snapshot)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [5, 6]), (13, [7, 8, 9]),
])
def test_init_accepts_tuple_of_coefficients(p, coeffs):
    from_tuple = F_pPolynomial(
        p, coefficients=tuple(F_pElement(p, c) for c in coeffs))
    from_list = F_pPolynomial(p, coefficients=[F_pElement(p, c) for c in coeffs])
    assert from_tuple == from_list
    assert _values(from_tuple) == _values(from_list)


@pytest.mark.parametrize("p", PRIMES)
def test_init_empty_coefficients_with_p_is_zero_polynomial(p):
    assert F_pPolynomial(p, coefficients=[]) == F_pPolynomial(p)
    assert F_pPolynomial(p, []).degree == float('-inf')
    assert bool(F_pPolynomial(p, [])) is False


@pytest.mark.parametrize("p", PRIMES)
def test_init_returns_instance_of_f_p_polynomial(p):
    assert type(F_pPolynomial(p)) is F_pPolynomial
    assert isinstance(F_pPolynomial(p, [F_pElement(p, 1)]), F_pPolynomial)


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
        F_pPolynomial(p, coefficients=[F_pElement(5, 1)])


@pytest.mark.parametrize("kwargs", [
    {},
    {"coefficients": []},
    {"p": None, "coefficients": None},
    {"p": None},
    {"p": None, "coefficients": []},
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
def test_init_coefficients_from_other_field_raises_type_error(
        make_element, p, other_p, values):
    coefficients = [make_element(p, values[0]), make_element(other_p, values[1])]
    with pytest.raises(
            TypeError, match=f"coefficients must be from the same field"):
        F_pPolynomial(p, coefficients=coefficients)


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101), (3, 5),
])
def test_init_mixed_coefficients_without_p_raises_type_error(
        make_element, p1, p2):
    coefficients = [make_element(p1, 1), make_element(p2, 1)]
    with pytest.raises(
            TypeError, match=f"coefficients must be from the same field"):
        F_pPolynomial(coefficients=coefficients)


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 11), (2, 5), (11, 13), (13, 3),
])
def test_init_field_check_uses_stripped_coefficients(p, other_p):
    with pytest.raises(TypeError, match="coefficients must be from the same field"):
        F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(other_p, 1)])


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 11), (2, 5), (11, 13), (13, 3),
])
def test_init_zero_from_other_field_is_not_stripped(p, other_p):
    # нюанс: хвостовой ноль из чужого поля НЕ отбрасывается, так как
    # F_pElement.__eq__ требует совпадения p, -> срабатывает проверка поля
    with pytest.raises(TypeError, match="coefficients must be from the same field"):
        F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(other_p, 0)])


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1]), (11, [4, 5]), (13, [6, 7, 8]),
])
def test_init_with_non_element_coefficients_raises(p, coeffs):
    # коэффициенты обязаны быть F_pElement: обращение к c.p у int -> AttributeError
    with pytest.raises((AttributeError, TypeError)):
        F_pPolynomial(p, coefficients=coeffs)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, "2"]), (7, [None]), (2, [1, 2.5]), (11, [[1]]), (13, [{}]),
])
def test_init_with_invalid_coefficient_types_raises(p, coeffs):
    with pytest.raises((AttributeError, TypeError)):
        F_pPolynomial(p, coefficients=coeffs)


@pytest.mark.parametrize("p", [4, 6, 9, 15, 100])
def test_init_non_prime_p_checked_before_coefficients(p):
    # нюанс: непростое p важнее прочих проверок
    with pytest.raises(ValueError, match="p must be prime"):
        F_pPolynomial(p, coefficients=[])
    with pytest.raises(ValueError, match="p must be prime"):
        F_pPolynomial(p, coefficients=[F_pElement(5, 1)])


# ===========================================================================
# 3. Свойства p и degree
# ===========================================================================

@pytest.mark.parametrize("p", PRIMES)
def test_p_property_returns_field_order(p):
    assert F_pPolynomial(p).p == p


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_p_property_matches_coefficients_field(make_poly, p, coeffs):
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


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [1]), (2, [1, 1]), (11, [3, 4, 5, 6]), (13, [0, 1]),
])
def test_degree_matches_number_of_accessible_coefficients(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly.degree == poly.degree
    assert len(_coefficients(poly)) == int(poly.degree) + 1


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 1), (11, 5), (13, 12), (101, 7),
])
def test_degree_property_is_read_only(make_poly, p, value):
    poly = make_poly(p, [value])
    with pytest.raises(AttributeError):
        poly.degree = 5


@pytest.mark.parametrize("p, coeffs", [
    (5, [0]), (5, [0, 0]), (7, [0, 0, 0]), (11, [0]), (13, [0, 0]),
])
def test_degree_of_all_zero_coefficients_is_minus_infinity(make_poly, p, coeffs):
    assert make_poly(p, coeffs).degree == float('-inf')
    assert make_poly(p, coeffs) == F_pPolynomial(p)


# ===========================================================================
# 4. __getitem__ — доступ к коэффициентам по индексу
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, index, expected", [
    (5, [1, 2, 3], 0, 1),
    (5, [1, 2, 3], 1, 2),
    (5, [1, 2, 3], 2, 3),
    (7, [6, 5, 4], 0, 6),
    (7, [6, 5, 4], 2, 4),
    (11, [10], 0, 10),
    (13, [0, 1], 0, 0),
    (13, [0, 1], 1, 1),
    (2, [1, 1], 1, 1),
])
def test_getitem_returns_coefficient(make_poly, p, coeffs, index, expected):
    assert make_poly(p, coeffs)[index] == F_pElement(p, expected)


@pytest.mark.parametrize("p, coeffs, index", [
    (5, [1, 2, 3], 3),
    (5, [1, 2, 3], 10),
    (7, [6], 1),
    (7, [6], 100),
    (11, [1, 2], 5),
])
def test_getitem_beyond_degree_returns_zero(make_poly, p, coeffs, index):
    assert make_poly(p, coeffs)[index] == F_pElement.zero(p)


@pytest.mark.parametrize("prime", PRIMES)
def test_getitem_of_zero_polynomial_returns_zero(prime):
    poly = F_pPolynomial(prime)
    for index in range(5):
        assert poly[index] == F_pElement.zero(prime)


@pytest.mark.parametrize("p, coeffs, index", [
    (5, [1, 2], -1),
    (5, [1], -2),
    (7, [3, 4], -5),
    (11, [1], -1),
    (13, [1, 2, 3], -3),
])
def test_getitem_negative_index_raises_index_error(make_poly, p, coeffs, index):
    with pytest.raises(IndexError, match="index cannot be negative"):
        make_poly(p, coeffs)[index]


@pytest.mark.parametrize("index", [
    "0", 1.5, None, [0], (0,), {}, slice(0, 2), F_pElement(5, 0),
])
def test_getitem_non_int_index_raises_type_error(make_poly, index):
    with pytest.raises(TypeError, match="index must be an integer"):
        make_poly(5, [1, 2])[index]


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [0, 5]), (13, [1, 0, 2]),
])
def test_getitem_accepts_bool_index(make_poly, p, coeffs):
    # нюанс: bool — подкласс int, поэтому poly[True] == poly[1]
    poly = make_poly(p, coeffs)
    assert poly[False] == poly[0]
    assert poly[True] == poly[1]


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4, 5]), (2, [1, 0, 1]), (11, [9, 8]), (13, [1, 0, 0, 2]),
])
def test_getitem_reconstructs_all_coefficients(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert _values(poly) == [c % p for c in coeffs]


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4, 5]), (11, [9, 8]), (13, [1, 0, 2]), (2, [1, 1]),
])
def test_getitem_last_coefficient_is_nonzero(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    degree = int(poly.degree)
    assert poly[degree] != F_pElement.zero(p)
    assert poly[degree + 1] == F_pElement.zero(p)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4, 5]), (13, [6, 7, 8]),
])
def test_getitem_of_leading_zeros(make_poly, p, coeffs):
    poly = make_poly(p, [0] + coeffs)
    assert poly[0] == F_pElement.zero(p)
    assert poly[1] == F_pElement(p, coeffs[0])
    assert poly.degree == len(coeffs)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [4, 5]), (13, [6, 7]),
])
def test_iteration_is_unbounded(make_poly, p, coeffs):
    # нюанс: итерация работает по старому протоколу через __getitem__;
    # IndexError для больших индексов не возникает, поэтому list(poly) зависает
    poly = make_poly(p, coeffs)
    iterator = iter(poly)
    for index, value in enumerate(coeffs):
        assert next(iterator) == F_pElement(p, value)
    assert next(iterator) == F_pElement.zero(p)
    assert next(iterator) == F_pElement.zero(p)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4, 5]), (11, [9, 8]), (13, [1, 0, 2]), (2, [1, 0, 1]),
])
def test_getitem_of_negation(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    negated = -poly
    for index in range(len(coeffs)):
        assert negated[index] == -poly[index]


# ===========================================================================
# 5. __eq__ / __ne__ / __hash__
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


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 3),
])
def test_equal_coefficients_in_different_fields_are_distinct_in_set(make_poly, p1, p2):
    assert len({make_poly(p1, [1]), make_poly(p2, [1])}) == 2


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 0]), (7, [3, 0, 0]), (2, [1, 0]), (11, [4, 0]), (13, [6, 0]),
])
def test_hash_ignores_trailing_zeros(make_poly, p, coeffs):
    stripped = list(coeffs)
    while stripped and stripped[-1] % p == 0:
        stripped.pop()
    assert hash(make_poly(p, coeffs)) == hash(make_poly(p, stripped))


# ===========================================================================
# 6. str_format() и __str__
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 2, 3], "(3 mod 5)x^2 + (2 mod 5)x^1 + (1 mod 5)"),
    (5, [3], "(3 mod 5)"),
    (7, [0, 0, 4], "(4 mod 7)x^2 + (0 mod 7)x^1 + (0 mod 7)"),
    (2, [1, 1], "(1 mod 2)x^1 + (1 mod 2)"),
    (11, [10, 0, 1], "(1 mod 11)x^2 + (0 mod 11)x^1 + (10 mod 11)"),
    (13, [0, 1], "(1 mod 13)x^1 + (0 mod 13)"),
])
def test_str_without_explicit_bound(make_poly, p, coeffs, expected):
    assert str(make_poly(p, coeffs)) == expected


@pytest.mark.parametrize("p, coeffs, bound, expected", [
    (5, [1, 2, 3], 4,
     "(0 mod 5)x^4 + (0 mod 5)x^3 + (3 mod 5)x^2 + (2 mod 5)x^1 + (1 mod 5)"),
    (5, [3], 1, "(0 mod 5)x^1 + (3 mod 5)"),
    (5, [3], 0, "(3 mod 5)"),
    (7, [1], 2, "(0 mod 7)x^2 + (0 mod 7)x^1 + (1 mod 7)"),
    (11, [10], 1, "(0 mod 11)x^1 + (10 mod 11)"),
])
def test_str_format_pads_with_zeros(make_poly, p, coeffs, bound, expected):
    assert make_poly(p, coeffs).str_format(bound) == expected


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6]), (2, [1, 0, 1]), (11, [4, 5]), (13, [0, 0, 1]),
])
def test_str_format_default_equals_str(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly.str_format() == str(poly)
    assert poly.str_format(None) == str(poly)


@pytest.mark.parametrize("p, coeffs, bound, expected", [
    (5, [1, 2, 3], 1, "(2 mod 5)x^1 + (1 mod 5)"),
    (5, [1, 2, 3], 0, "(1 mod 5)"),
    (7, [6, 5, 4], 1, "(5 mod 7)x^1 + (6 mod 7)"),
    (11, [1, 2, 3], 2, "(3 mod 11)x^2 + (2 mod 11)x^1 + (1 mod 11)"),
    (13, [9, 8], 0, "(9 mod 13)"),
])
def test_str_format_truncates_when_bound_below_degree(
        make_poly, p, coeffs, bound, expected):
    # нюанс: старшие слагаемые молча отбрасываются
    assert make_poly(p, coeffs).str_format(bound) == expected


@pytest.mark.parametrize("p, coeffs, bound", [
    (5, [1, 2, 3], 2), (7, [6, 5, 4], 2), (11, [1, 2], 1), (13, [9], 0),
    (2, [1, 1], 1),
])
def test_str_format_with_bound_equal_to_degree_is_full(make_poly, p, coeffs, bound):
    poly = make_poly(p, coeffs)
    assert poly.str_format(bound) == str(poly)


@pytest.mark.parametrize("p, bound", [
    (5, -1), (5, -3), (7, -1), (11, -5), (13, -2),
])
def test_str_format_negative_bound_gives_empty_string(p, bound):
    # нюанс: range(bound, -1, -1) пуст -> join возвращает пустую строку
    assert F_pPolynomial(p).str_format(bound) == ""
    assert F_pPolynomial(p, [F_pElement(p, 1)]).str_format(bound) == ""


@pytest.mark.parametrize("p", PRIMES)
def test_str_of_zero_polynomial(p):
    assert str(F_pPolynomial(p)) == f"(0 mod {p})"
    assert F_pPolynomial(p).str_format() == f"(0 mod {p})"
    assert F_pPolynomial(p).str_format(0) == f"(0 mod {p})"


@pytest.mark.parametrize("p, bound, expected", [
    (5, 2, "(0 mod 5)x^2 + (0 mod 5)x^1 + (0 mod 5)"),
    (5, 1, "(0 mod 5)x^1 + (0 mod 5)"),
    (7, 3, "(0 mod 7)x^3 + (0 mod 7)x^2 + (0 mod 7)x^1 + (0 mod 7)"),
    (11, 1, "(0 mod 11)x^1 + (0 mod 11)"),
    (13, 0, "(0 mod 13)"),
])
def test_str_format_of_zero_polynomial_pads_zeros(p, bound, expected):
    assert F_pPolynomial(p).str_format(bound) == expected


@pytest.mark.parametrize("p, value, expected", [
    (5, 0, "(0 mod 5)"),
    (5, 4, "(4 mod 5)"),
    (7, 6, "(6 mod 7)"),
    (11, 0, "(0 mod 11)"),
    (13, 12, "(12 mod 13)"),
])
def test_str_of_constant_polynomial(make_poly, p, value, expected):
    assert str(make_poly(p, [value])) == expected


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
    terms = str(make_poly(p, coeffs)).split(" + ")
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


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5]), (2, [1, 1]), (11, [10, 1]), (13, [1, 2, 3]),
])
def test_str_format_does_not_mutate_polynomial(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for bound in (None, 0, 1, 5):
        poly.str_format(bound)
    assert poly == make_poly(p, coeffs)
    assert poly.degree == len(coeffs) - 1
    assert _values(poly) == coeffs


@pytest.mark.parametrize("bound", [2.5, "2", [2], 1.0, {}])
def test_str_format_non_int_bound_raises_type_error(make_poly, bound):
    with pytest.raises(TypeError):
        make_poly(5, [1, 2]).str_format(bound)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1]), (11, [5, 6]), (13, [7, 8, 9]),
])
def test_str_of_equal_polynomials_is_equal(make_poly, p, coeffs):
    assert str(make_poly(p, coeffs)) == str(make_poly(p, list(coeffs) + [0]))


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4, 5]), (11, [9, 8]), (13, [1, 0, 2]), (101, [7, 8, 9]),
])
def test_str_format_with_large_bound_pads_many_zeros(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    bound = len(coeffs) + 2
    assert len(poly.str_format(bound).split(" + ")) == bound + 1


# ===========================================================================
# 7. __repr__
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 2],
     "F_pPolynomial(5, coefficients=(F_pElement(p=5, value=1), "
     "F_pElement(p=5, value=2)))"),
    (5, [], "F_pPolynomial(5, coefficients=())"),
    (7, [], "F_pPolynomial(7, coefficients=())"),
    (2, [1], "F_pPolynomial(2, coefficients=(F_pElement(p=2, value=1),))"),
    (13, [0, 12],
     "F_pPolynomial(13, coefficients=(F_pElement(p=13, value=0), "
     "F_pElement(p=13, value=12)))"),
])
def test_repr_format(make_poly, p, coeffs, expected):
    assert repr(make_poly(p, coeffs)) == expected


@pytest.mark.parametrize("p", PRIMES)
def test_repr_of_zero_polynomial(p):
    assert repr(F_pPolynomial(p)) == f"F_pPolynomial({p}, coefficients=())"


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4, 5, 6]), (13, [0, 1]),
])
def test_repr_contains_field_and_coefficients(make_poly, p, coeffs):
    text = repr(make_poly(p, coeffs))
    assert text.startswith(f"F_pPolynomial({p}, coefficients=")
    assert text.count("F_pElement(") == len(coeffs)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4, 5, 6]), (13, [0, 1]),
])
def test_repr_differs_from_expected_str(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert repr(poly) != poly.str_format()
    assert "x^" not in repr(poly)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1]), (11, [9, 9]), (13, [1, 0, 2]),
])
def test_repr_of_equal_polynomials_is_equal(make_poly, p, coeffs):
    assert repr(make_poly(p, coeffs)) == repr(make_poly(p, coeffs))


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 0]), (7, [3, 0]), (2, [1, 0, 0]), (11, [4, 0]), (13, [6, 0, 0]),
])
def test_repr_shows_stripped_coefficients(make_poly, p, coeffs):
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
# 8. __pos__ / __neg__
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
    assert _values(+poly) == _values(poly)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1]), (7, [2, 3]), (2, [1]), (11, [0, 1]), (13, [4, 5, 6]),
])
def test_pos_is_idempotent(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert +(+(+poly)) is poly


@pytest.mark.parametrize("prime", PRIMES)
def test_pos_of_zero_polynomial(prime):
    poly = F_pPolynomial(prime)
    assert (+poly) is poly
    assert +poly == poly


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
    assert _values(-make_poly(p, coeffs)) == expected


@pytest.mark.parametrize("p", PRIMES)
def test_neg_of_zero_polynomial_is_zero_polynomial(p):
    negated = -F_pPolynomial(p)
    assert negated == F_pPolynomial(p)
    assert negated.degree == float('-inf')
    assert bool(negated) is False
    assert _values(negated) == []


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [4, 5]), (2, [1, 0, 1]), (11, [9]), (13, [0, 1, 2]),
])
def test_double_negation_restores_polynomial(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert -(-poly) == poly
    assert _values(-(-poly)) == _values(poly)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1]), (11, [4, 5, 6]), (13, [7]),
])
def test_neg_preserves_field_and_degree(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert (-poly).p == p
    assert (-poly).degree == poly.degree


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
    (5, [1, 2], [4, 3]),
    (7, [3, 5], [4, 2]),
    (2, [1, 1], [1, 1]),
    (11, [10], [1]),
    (13, [0, 1], [0, 12]),
])
def test_neg_coefficients_elementwise(make_poly, p, coeffs, expected):
    assert _values(-make_poly(p, coeffs)) == expected


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4, 5]), (11, [6, 7]), (13, [8, 9, 10]), (2, [1, 0, 1]),
])
def test_neg_of_sum_equals_sum_of_negs(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    twin = make_poly(p, coeffs)
    assert -(poly + twin) == (-poly) + (-twin)


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
    assert _values(make_poly(p, a) + make_poly(p, b)) == expected


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
    assert _values(make_poly(p, a) + make_poly(p, b)) == \
           _values(make_poly(p, b) + make_poly(p, a))


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
def test_add_different_fields_raises_type_error(make_poly, p1, p2):
    with pytest.raises(
            TypeError, match="cannot \\+ with two polynomials in different fields"):
        make_poly(p1, [1]) + make_poly(p2, [1])


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES)
def test_add_non_polynomial_raises_type_error(make_poly, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_poly(5, [1]) + other


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
    assert bool(result) is False
    assert result.degree == float('-inf')
    assert _values(result) == []


@pytest.mark.parametrize("prime", PRIMES)
def test_additive_laws_with_zero_polynomial(prime):
    p = prime
    zero = F_pPolynomial(p)
    poly = F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    assert poly + zero == poly
    assert zero + poly == poly
    assert poly + (-poly) == zero
    assert poly - poly == zero


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
    assert _values(make_poly(p, a) - make_poly(p, b)) == expected


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [4, 5]), (2, [1, 1]), (11, [9, 8, 7]), (13, [0, 1]),
])
def test_sub_self_is_zero_polynomial(make_poly, p, coeffs):
    result = make_poly(p, coeffs) - make_poly(p, coeffs)
    assert result == F_pPolynomial(p)
    assert result.degree == float('-inf')
    assert bool(result) is False


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
def test_sub_different_fields_raises_type_error(make_poly, p1, p2):
    with pytest.raises(
            TypeError, match="cannot - with two polynomials in different fields"):
        make_poly(p1, [1]) - make_poly(p2, [1])


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES)
def test_sub_non_polynomial_raises_type_error(make_poly, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_poly(5, [1]) - other


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


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]), (7, [1], [2, 3]), (2, [1, 1], [1]), (11, [5, 6], [7]),
    (13, [1, 2, 3], [4, 5, 6]),
])
def test_sub_returns_new_polynomial_of_same_field(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    result = first - second
    assert isinstance(result, F_pPolynomial)
    assert result is not first and result is not second
    assert result.p == p


# ===========================================================================
# 11. __mul__
# ===========================================================================

@pytest.mark.parametrize("p, a, b, expected", [
    (5, [1, 2], [3, 4], [3, 0, 3]),
    (5, [2], [3], [1]),
    (5, [1, 2, 3], [4, 4], [4, 2, 0, 2]),
    (5, [0, 1], [0, 1], [0, 0, 1]),
    (7, [1, 1], [1, 1], [1, 2, 1]),
    (7, [3, 5], [2], [6, 3]),
    (7, [1, 2, 3, 4], [1], [1, 2, 3, 4]),
    (11, [1, 2, 3], [1, 1], [1, 3, 5, 3]),
    (13, [12], [12], [1]),
    (2, [1, 1], [1, 1], [1, 0, 1]),
    (3, [2, 2], [2], [1, 1]),
    (101, [1, 1], [1, 100], [1, 0, 100]),
])
def test_mul_coefficients(make_poly, p, a, b, expected):
    assert make_poly(p, a) * make_poly(p, b) == make_poly(p, expected)
    assert _values(make_poly(p, a) * make_poly(p, b)) == expected


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_mul_by_zero_polynomial_is_zero(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    zero = F_pPolynomial(p)
    assert poly * zero == zero
    assert zero * poly == zero
    assert bool(poly * zero) is False
    assert (poly * zero).degree == float('-inf')


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_mul_by_one_is_identity(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly * make_poly(p, [1]) == poly
    assert make_poly(p, [1]) * poly == poly
    assert poly * F_pPolynomial.one(p) == poly
    assert F_pPolynomial.one(p) * poly == poly


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]),
    (7, [1, 1], [2, 3, 4]),
    (2, [1, 1], [1]),
    (11, [1, 2, 3], [9, 8]),
    (13, [4], [5, 6]),
])
def test_mul_is_commutative(make_poly, p, a, b):
    assert make_poly(p, a) * make_poly(p, b) == make_poly(p, b) * make_poly(p, a)


@pytest.mark.parametrize("p, a, b, c", [
    (5, [1, 2], [3], [1, 1]),
    (7, [2, 3], [4], [1, 2]),
    (11, [1, 2], [3, 4], [5]),
    (13, [1], [2, 3], [4, 5]),
    (2, [1, 1], [1], [1, 1]),
])
def test_mul_is_associative(make_poly, p, a, b, c):
    left = (make_poly(p, a) * make_poly(p, b)) * make_poly(p, c)
    right = make_poly(p, a) * (make_poly(p, b) * make_poly(p, c))
    assert left == right


@pytest.mark.parametrize("p, a, b, c", [
    (5, [1, 2], [3], [1, 1]),
    (7, [2, 3], [4], [1, 2]),
    (11, [1, 2], [3, 4], [5]),
    (13, [1], [2, 3], [4, 5]),
    (2, [1, 1], [1], [1, 1]),
])
def test_mul_distributes_over_addition(make_poly, p, a, b, c):
    first = make_poly(p, a)
    second = make_poly(p, b)
    third = make_poly(p, c)
    assert first * (second + third) == first * second + first * third
    assert (second + third) * first == second * first + third * first


@pytest.mark.parametrize("p, a, b, expected_degree", [
    (5, [1, 2], [3, 4], 2),
    (7, [1, 1], [1, 1], 2),
    (11, [1, 2, 3], [1, 1], 3),
    (13, [12], [12], 0),
    (2, [1, 1], [1], 1),
    (5, [0, 1], [0, 1], 2),
])
def test_mul_degree_is_sum_of_degrees(make_poly, p, a, b, expected_degree):
    assert (make_poly(p, a) * make_poly(p, b)).degree == expected_degree


@pytest.mark.parametrize("p, a, b, x", [
    (5, [1, 2], [3, 4], 2),
    (7, [6, 1], [2, 3], 5),
    (11, [1, 2, 3], [9, 8], 4),
    (13, [0, 1], [1, 0, 2], 7),
    (2, [1, 1], [1], 1),
])
def test_mul_agrees_with_evaluation(make_poly, p, a, b, x):
    point = F_pElement(p, x)
    product = (make_poly(p, a) * make_poly(p, b))(point)
    assert product == make_poly(p, a)(point) * make_poly(p, b)(point)


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 17),
])
def test_mul_different_fields_raises_type_error(make_poly, p1, p2):
    with pytest.raises(
            TypeError, match="cannot \\* with two polynomials in different fields"):
        make_poly(p1, [1]) * make_poly(p2, [1])


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES)
def test_mul_non_polynomial_raises_type_error(make_poly, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_poly(5, [1, 2]) * other


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]), (7, [1], [2, 3]), (2, [1, 1], [1]), (11, [3, 4], [5]),
    (13, [1, 2, 3], [4, 5]),
])
def test_mul_does_not_mutate_operands(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    _ = first * second
    assert first == make_poly(p, a)
    assert second == make_poly(p, b)


@pytest.mark.parametrize("p, coeffs, times", [
    (5, [1, 1], 2), (7, [1, 2], 2), (11, [3], 3), (13, [1, 0, 1], 2), (2, [1, 1], 3),
])
def test_repeated_multiplication_raises_degree(make_poly, p, coeffs, times):
    poly = make_poly(p, coeffs)
    power = make_poly(p, [1])
    for _ in range(times):
        power = power * poly
    assert power.degree == poly.degree * times


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]), (7, [1], [2, 3]), (2, [1, 1], [1]), (11, [5, 6], [7]),
    (13, [1, 2, 3], [4, 5]),
])
def test_mul_returns_new_polynomial_of_same_field(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    result = first * second
    assert isinstance(result, F_pPolynomial)
    assert result is not first and result is not second
    assert result.p == p


@pytest.mark.parametrize("prime", PRIMES)
def test_no_zero_divisors_in_polynomial_ring(prime):
    p = prime
    first = F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    second = F_pPolynomial(p, coefficients=[F_pElement(p, 3), F_pElement(p, 1)])
    assert bool(first * second) is True
    assert (first * second).degree == first.degree + second.degree


# ===========================================================================
# 12. __bool__
# ===========================================================================

@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4]), (13, [5, 6]),
])
def test_nonzero_polynomial_is_truthy(make_poly, p, coeffs):
    assert bool(make_poly(p, coeffs)) is True
    assert make_poly(p, coeffs)


@pytest.mark.parametrize("p", PRIMES)
def test_zero_polynomial_is_falsy(p):
    assert bool(F_pPolynomial(p)) is False
    assert not F_pPolynomial(p)


@pytest.mark.parametrize("p, coeffs", [
    (5, [0]), (5, [0, 0, 0]), (7, [0, 0]), (11, [0]), (13, [0, 0, 0, 0]),
])
def test_polynomial_of_only_zero_coefficients_is_falsy(make_poly, p, coeffs):
    # хвостовые нули отбрасываются -> многочлен становится нулевым
    assert bool(make_poly(p, coeffs)) is False


@pytest.mark.parametrize("p", PRIMES)
def test_polynomial_from_empty_coefficients_is_falsy(p):
    assert bool(F_pPolynomial(p, coefficients=[])) is False


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 1], [4, 4]), (5, [2], [3]), (7, [3, 5], [4, 2]), (13, [12], [1]),
    (2, [1, 0, 1], [1, 0, 1]),
])
def test_bool_of_cancelled_sum_is_false(make_poly, p, a, b):
    assert bool(make_poly(p, a) + make_poly(p, b)) is False


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1]), (11, [0, 5]), (13, [6, 7]),
])
def test_bool_of_negation_and_unary_plus(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert bool(-poly) is bool(poly)
    assert bool(+poly) is bool(poly)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4, 5]), (13, [6]),
])
def test_bool_is_used_in_conditional(make_poly, p, coeffs):
    assert ("nonzero" if make_poly(p, coeffs) else "zero") == "nonzero"
    assert ("nonzero" if F_pPolynomial(p) else "zero") == "zero"


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (11, [0, 5]), (13, [6, 0, 1]), (2, [1, 1]),
])
def test_bool_agrees_with_degree(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert bool(poly) is (poly.degree != float('-inf'))


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (11, [9, 8]), (13, [5, 6]), (2, [1, 1]),
])
def test_bool_of_product(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert bool(poly * poly) is True
    assert bool(poly * F_pPolynomial(p)) is False


# ===========================================================================
# 13. random(p, degree_bound[, least_coefficient])
# ===========================================================================

@pytest.mark.parametrize("p, bound", [
    (5, 1), (5, 3), (7, 2), (11, 4), (13, 0), (101, 5), (2, 2),
])
def test_random_returns_polynomial_of_field(p, bound):
    poly = F_pPolynomial.random(p, bound)
    assert isinstance(poly, F_pPolynomial)
    assert poly.p == p


@pytest.mark.parametrize("bound", [0, 1, 2, 3, 5, 9])
def test_random_degree_does_not_exceed_degree_bound(bound):
    for _ in range(10):
        assert F_pPolynomial.random(101, bound).degree <= bound


@pytest.mark.parametrize("p, bound, secret", [
    (5, 1, 3), (7, 2, 6), (11, 3, 10), (13, 0, 12), (101, 4, 42), (2, 2, 1),
])
def test_random_uses_provided_least_coefficient(p, bound, secret):
    least = F_pElement(p, secret)
    poly = F_pPolynomial.random(p, bound, least_coefficient=least)
    assert poly[0] == least


@pytest.mark.parametrize("p, bound, secret", [
    (5, 2, 1), (7, 3, 5), (11, 1, 9), (13, 2, 7), (101, 3, 33),
])
def test_random_keeps_least_coefficient_across_calls(p, bound, secret):
    least = F_pElement(p, secret)
    for _ in range(20):
        assert F_pPolynomial.random(p, bound, least_coefficient=least)[0] == least


@pytest.mark.parametrize("p", [5, 7, 11, 13, 101])
def test_random_with_zero_least_coefficient(p):
    poly = F_pPolynomial.random(p, 2, least_coefficient=F_pElement.zero(p))
    assert poly[0] == F_pElement.zero(p)


@pytest.mark.parametrize("p, bound", [
    (5, 2), (7, 3), (11, 2), (13, 4), (101, 3),
])
def test_random_without_least_coefficient_has_random_constant_term(p, bound):
    constants = {F_pPolynomial.random(p, bound)[0].value for _ in range(40)}
    assert constants <= set(range(p))
    if p > 3:
        assert len(constants) > 1


@pytest.mark.parametrize("p, bound", [
    (11, 3), (13, 2), (101, 4), (7, 3), (5, 2),
])
def test_random_produces_different_polynomials(p, bound):
    samples = {repr(F_pPolynomial.random(p, bound)) for _ in range(30)}
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


@pytest.mark.parametrize("p", [5, 7, 11, 13, 101])
def test_random_with_degree_bound_zero_is_constant(p):
    poly = F_pPolynomial.random(p, 0)
    assert poly.degree <= 0
    assert poly[1] == F_pElement.zero(p)
    assert len(_coefficients(poly)) <= 1


@pytest.mark.parametrize("p, bound", [
    (5, 1), (7, 2), (11, 3), (13, 1), (101, 2),
])
def test_random_returns_new_object_each_call(p, bound):
    assert F_pPolynomial.random(p, bound) is not F_pPolynomial.random(p, bound)


@pytest.mark.parametrize("p, bound", [
    (5, 2), (7, 3), (11, 2), (13, 4), (101, 3),
])
def test_random_result_is_usable_in_arithmetic(p, bound):
    first = F_pPolynomial.random(p, bound)
    second = F_pPolynomial.random(p, bound)
    assert isinstance(first + second, F_pPolynomial)
    assert isinstance(first - second, F_pPolynomial)
    assert isinstance(first * second, F_pPolynomial)
    assert isinstance(-first, F_pPolynomial)
    assert (+first) is first
    assert (first + second).p == p


@pytest.mark.parametrize("p, bound", [
    (5, 2), (7, 2), (11, 3), (13, 1), (101, 2),
])
def test_random_coefficients_are_elements_of_field(p, bound):
    poly = F_pPolynomial.random(p, bound)
    for index in range(bound + 1):
        coefficient = poly[index]
        assert isinstance(coefficient, F_pElement)
        assert coefficient.p == p


# ===========================================================================
# 14. __call__ — вычисление значения многочлена в точке
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
    assert poly(F_pElement.zero(p)) == poly[0]


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
    assert poly(F_pElement.one(p)) == F_pElement(p, expected)
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
def test_call_of_zero_polynomial_validates_argument_type(argument):
    # нюанс: проверка типа выполняется до цикла, поэтому даже нулевой
    # многочлен отклоняет аргумент не-F_pElement
    with pytest.raises(TypeError,
                       match="polynomial argument must be a F_pElement"):
        F_pPolynomial(5)(argument)


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
    poly = F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
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


@pytest.mark.parametrize("other", NON_POLYNOMIAL_VALUES)
def test_call_with_non_element_raises_type_error(make_poly, other):
    with pytest.raises(TypeError) as exc_info:
        make_poly(5, [1, 2])(other)
    message = str(exc_info.value)
    assert "polynomial argument must be a F_pElement" in message
    assert f"not '{other.__class__.__name__}'" in message


@pytest.mark.parametrize("argument, class_name", [
    (3, "int"),
    (3.5, "float"),
    ("1", "str"),
    (None, "NoneType"),
    ([1], "list"),
    ((1, 2), "tuple"),
    ({1}, "set"),
])
def test_call_error_message_mentions_argument_type(
        make_poly, argument, class_name):
    with pytest.raises(TypeError) as exc_info:
        make_poly(5, [1, 2])(argument)
    assert str(exc_info.value) == (
        f"polynomial argument must be a F_pElement, not '{class_name}'")


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_call_with_polynomial_argument_raises_type_error(make_poly, p, coeffs):
    with pytest.raises(TypeError,
                       match="polynomial argument must be a F_pElement, "
                             "not 'F_pPolynomial'"):
        make_poly(p, [1, 2])(make_poly(p, coeffs))


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4, 5]), (13, [6, 7]),
])
def test_call_validation_precedes_coefficient_loop(make_poly, p, coeffs):
    # ошибка типа возникает до обращения к коэффициентам
    with pytest.raises(TypeError,
                       match="polynomial argument must be a F_pElement"):
        make_poly(p, coeffs)(object())


@pytest.mark.parametrize("prime", PRIMES)
def test_call_accepts_any_element_of_own_field(prime):
    p = prime
    poly = F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    for value in range(min(p, 5)):
        assert isinstance(poly(F_pElement(p, value)), F_pElement)
    assert isinstance(poly(F_pElement.zero(p)), F_pElement)
    assert isinstance(poly(F_pElement.one(p)), F_pElement)


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101),
])
def test_call_with_element_of_other_field_raises_type_error(make_poly, p, other_p):
    with pytest.raises(TypeError, match="polynomial argument must be in the " \
                                        "same field as the polynomial, not "
                                        f"F_{other_p}"):
        make_poly(p, [1, 2])(F_pElement(other_p, 1))


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [6, 5]), (2, [1, 1]), (11, [10, 1]), (13, [1, 2, 3]),
])
def test_call_does_not_mutate_polynomial(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(min(p, 5)):
        poly(F_pElement(p, value))
    assert poly == make_poly(p, coeffs)
    assert _values(poly) == coeffs


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
def test_call_result_supports_field_arithmetic(prime):
    p = prime
    poly = F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    value = poly(F_pElement.one(p))
    assert isinstance(value + value, F_pElement)
    assert isinstance(value * value, F_pElement)
    assert isinstance(-value, F_pElement)
    assert isinstance(value ** 3, F_pElement)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (11, [9, 8]), (13, [5, 6]), (101, [7, 8]),
])
def test_call_uses_indexed_coefficients(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(min(p, 4)):
        x = F_pElement(p, value)
        expected = F_pElement.zero(p)
        for index in range(len(coeffs)):
            expected = expected + poly[index] * (x ** index)
        assert poly(x) == expected


# ===========================================================================
# 15. share_at (удалён) и Share.make как замена
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2, 3], 0),
    (5, [1, 2], 4),
    (7, [6, 5], 3),
    (11, [10], 7),
    (13, [0, 1], 12),
    (2, [1, 1], 1),
])
def test_share_at_is_removed_and_raises_not_implemented(make_poly, p, coeffs, x):
    with pytest.raises(NotImplementedError):
        make_poly(p, coeffs).share_at(F_pElement(p, x))


@pytest.mark.parametrize("prime", PRIMES)
def test_share_at_of_zero_polynomial_raises_not_implemented(prime):
    with pytest.raises(NotImplementedError):
        F_pPolynomial(prime).share_at(F_pElement.zero(prime))


@pytest.mark.parametrize("argument", [3, None, "1", 2.5, [1]])
def test_share_at_raises_not_implemented_before_argument_check(argument):
    # нюанс: метод сразу бросает NotImplementedError, тип аргумента не проверяется
    with pytest.raises(NotImplementedError):
        F_pPolynomial(5).share_at(argument)


def test_share_at_docstring_points_to_share_make():
    docstring = F_pPolynomial.share_at.__doc__
    assert docstring is not None
    assert "Share.make(polynomial, x)" in docstring
    assert "был удален" in docstring


@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2, 3], 0),
    (5, [1, 2], 4),
    (7, [6, 5], 3),
    (11, [10], 7),
    (13, [0, 1], 12),
    (2, [1, 1], 1),
])
def test_share_make_replaces_share_at(make_poly, p, coeffs, x):
    poly = make_poly(p, coeffs)
    point = F_pElement(p, x)
    share = Share.make(poly, point)
    assert isinstance(share, Share)
    assert share.x == point
    assert share.value == poly(point)
    assert share.p == p
    assert share == Share(point, poly(point))


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [3, 1, 2], [1, 2, 3]),
    (7, [6, 5, 4], [1, 2, 3]),
    (11, [10, 9, 8], [1, 2, 3]),
    (13, [1, 2, 3], [4, 5, 6]),
    (101, [42, 7, 8], [1, 2, 3]),
])
def test_share_make_at_zero_returns_least_coefficient(make_poly, p, coeffs, xs):
    poly = make_poly(p, coeffs)
    secret = Share.make(poly, F_pElement.zero(p))
    assert secret.value == F_pElement(p, coeffs[0])
    assert secret.x == F_pElement.zero(p)
    for x in xs:
        point = F_pElement(p, x)
        assert Share.make(poly, point) == Share(point, poly(point))


@pytest.mark.parametrize("other", [3, None, "1", 2.5, [1], object()])
def test_share_make_with_non_element_raises_type_error(make_poly, other):
    # Share.make делегирует вычисление __call__, который проверяет тип аргумента
    with pytest.raises(TypeError,
                       match="polynomial argument must be a F_pElement"):
        Share.make(make_poly(5, [1, 2]), other)


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101),
])
def test_share_make_with_element_of_other_field_raises_type_error(
        make_poly, p, other_p):
    with pytest.raises(TypeError, match="polynomial argument must be in the " \
                                        "same field as the polynomial, not "
                                        f"F_{other_p}"):
        Share.make(make_poly(p, [1, 2]), F_pElement(other_p, 1))


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_share_make_does_not_mutate_polynomial(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for value in range(min(p, 5)):
        Share.make(poly, F_pElement(p, value))
    assert poly == make_poly(p, coeffs)
    assert _values(poly) == [c % p for c in coeffs]


@pytest.mark.parametrize("prime", PRIMES)
def test_share_make_of_one_polynomial(prime):
    p = prime
    one = F_pPolynomial.one(p)
    for value in range(min(p, 5)):
        point = F_pElement(p, value)
        share = Share.make(one, point)
        assert share.x == point
        assert share.value == F_pElement.one(p)
        assert share.p == p


@pytest.mark.parametrize("p, bound, secret", [
    (5, 3, 2), (7, 2, 4), (11, 4, 7), (13, 1, 11), (101, 2, 99),
])
def test_share_make_of_random_polynomial_returns_secret(p, bound, secret):
    least = F_pElement(p, secret)
    poly = F_pPolynomial.random(p, bound, least_coefficient=least)
    share = Share.make(poly, F_pElement.zero(p))
    assert share.value == least
    assert share.x == F_pElement.zero(p)


# ===========================================================================
# 16. Свойство coefficients и итерируемые конструкторы
# ===========================================================================

@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_coefficients_property_returns_tuple(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    coefficients = poly.coefficients
    assert isinstance(coefficients, tuple)
    assert len(coefficients) == len(coeffs)
    for index, value in enumerate(coeffs):
        assert coefficients[index] == F_pElement(p, value)


@pytest.mark.parametrize("p, coeffs, expected", [
    (5, [1, 2, 0], (F_pElement(5, 1), F_pElement(5, 2))),
    (7, [3, 0, 0], (F_pElement(7, 3),)),
    (5, [0], ()),
    (11, [1, 2, 3, 0, 0], (F_pElement(11, 1), F_pElement(11, 2),
                           F_pElement(11, 3))),
])
def test_coefficients_property_ignores_trailing_zeros(
        make_poly, p, coeffs, expected):
    assert make_poly(p, coeffs).coefficients == expected


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_coefficients_of_zero_polynomial_is_empty_tuple(p):
    poly = F_pPolynomial(p)
    assert poly.coefficients == ()
    assert isinstance(poly.coefficients, tuple)


@pytest.mark.parametrize("p, coeffs", [
    (5, [0, 1, 2]), (7, [0, 3, 0, 4]), (11, [0, 5]), (13, [0, 0, 0, 1]),
])
def test_coefficients_property_keeps_leading_zeros(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly.coefficients == tuple(F_pElement(p, c) for c in coeffs)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_coefficients_property_is_read_only(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    with pytest.raises(AttributeError):
        poly.coefficients = ()


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 3]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_coefficients_match_getitem_and_degree(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert _values(poly) == [c.value for c in poly.coefficients]
    assert len(poly.coefficients) == int(poly.degree) + 1


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_constructor_accepts_generator_of_elements(make_poly, p, coeffs):
    elements = (F_pElement(p, c) for c in coeffs)
    poly = F_pPolynomial(p, coefficients=elements)
    assert poly == make_poly(p, coeffs)
    assert poly.coefficients == tuple(F_pElement(p, c) for c in coeffs)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_constructor_accepts_map_and_filter(make_poly, p, coeffs):
    elements = [F_pElement(p, c) for c in coeffs]
    poly_map = F_pPolynomial(p, coefficients=map(lambda e: e, elements))
    poly_filter = F_pPolynomial(p, coefficients=filter(lambda e: True, elements))
    assert poly_map == make_poly(p, coeffs)
    assert poly_filter == make_poly(p, coeffs)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_constructor_accepts_iterator(make_poly, p, coeffs):
    elements = iter([F_pElement(p, c) for c in coeffs])
    poly = F_pPolynomial(p, coefficients=elements)
    assert poly == make_poly(p, coeffs)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_constructor_accepts_reversed_iterable(make_poly, p, coeffs):
    # хвостовые нули корректно отбрасываются и для ленивых источников
    elements = reversed([F_pElement(p, c) for c in coeffs])
    elements = reversed(list(elements))
    poly = F_pPolynomial(p, coefficients=elements)
    assert poly == make_poly(p, coeffs)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_constructor_does_not_consume_source_list_twice(p, coeffs):
    source = [F_pElement(p, c) for c in coeffs]
    poly = F_pPolynomial(p, coefficients=source)
    assert len(source) == len(coeffs)
    assert poly.degree == len(coeffs) - 1


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2, 0]), (7, [3, 0, 0]), (11, [4, 0]),
])
def test_constructor_with_generator_strips_trailing_zeros(p, coeffs):
    elements = (F_pElement(p, c) for c in coeffs)
    poly = F_pPolynomial(p, coefficients=elements)
    stripped = list(coeffs)
    while stripped and stripped[-1] % p == 0:
        stripped.pop()
    assert _values(poly) == [c % p for c in stripped]
    assert poly.degree == len(stripped) - 1


# ===========================================================================
# 16. Прочие нюансы и ошибки существующих операций
# ===========================================================================

@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1]), (11, [4]), (13, [5, 6]),
])
def test_truediv_by_unsupported_operand_raises_type_error(make_poly, p, coeffs):
    # __truediv__ определён только для F_pElement: прочие операнды дают TypeError
    poly = make_poly(p, coeffs)
    with pytest.raises(TypeError):
        poly / poly      # ~poly -> bad operand type for unary ~
    with pytest.raises(TypeError):
        poly / 2         # ~2 = -3 -> не F_pElement -> TypeError
    with pytest.raises(TypeError):
        poly / None


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3]), (2, [1, 1]), (11, [4, 5]), (13, [6, 7, 8]),
])
def test_properties_are_read_only(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    for attribute in ("p", "degree"):
        with pytest.raises(AttributeError):
            setattr(poly, attribute, 1)


@pytest.mark.parametrize("prime", PRIMES)
def test_zero_polynomial_public_state(prime):
    p = prime
    poly = F_pPolynomial(p)
    assert poly.p == p
    assert poly.degree == float('-inf')
    assert repr(poly) == f"F_pPolynomial({p}, coefficients=())"
    assert bool(poly) is False
    assert _values(poly) == []


@pytest.mark.parametrize("prime", PRIMES)
def test_unary_operations_on_zero_polynomial(prime):
    p = prime
    zero = F_pPolynomial(p)
    assert (-zero) == zero
    assert (+zero) is zero
    assert -(-zero) == zero
    assert bool(-zero) is False


@pytest.mark.parametrize("p, a, b", [
    (5, [1, 2], [3, 4]), (7, [1], [2, 3]), (2, [1, 1], [1]), (11, [5, 6], [7]),
    (13, [1, 2, 3], [4, 5]),
])
def test_ring_laws(make_poly, p, a, b):
    first = make_poly(p, a)
    second = make_poly(p, b)
    assert first * second == second * first
    assert (first + second) - second == first
    assert first * make_poly(p, [1]) == first
    assert first * F_pPolynomial(p) == F_pPolynomial(p)
    assert -(first + second) == (-first) + (-second)


# ===========================================================================
# 17. one(p) — константа 1 (мультипликативная единица)
# ===========================================================================

@pytest.mark.parametrize("p", PRIMES)
def test_one_returns_constant_polynomial(p):
    poly = F_pPolynomial.one(p)
    assert isinstance(poly, F_pPolynomial)
    assert poly.p == p
    assert poly.degree == 0
    assert _values(poly) == [1]
    assert poly[0] == F_pElement.one(p)
    assert poly[1] == F_pElement.zero(p)


@pytest.mark.parametrize("p", PRIMES)
def test_one_equals_constructed_constant(make_poly, p):
    assert F_pPolynomial.one(p) == make_poly(p, [1])
    assert F_pPolynomial.one(p) == F_pPolynomial(p, [F_pElement(p, 1)])
    assert F_pPolynomial.one(p) != make_poly(p, [1, 0, 1])


@pytest.mark.parametrize("p", PRIMES)
def test_one_string_and_repr(p):
    poly = F_pPolynomial.one(p)
    assert str(poly) == f"(1 mod {p})"
    assert repr(poly) == (f"F_pPolynomial({p}, "
                          f"coefficients=(F_pElement(p={p}, value=1),))")


@pytest.mark.parametrize("p", PRIMES)
def test_one_is_truthy_and_hashable(p):
    poly = F_pPolynomial.one(p)
    assert bool(poly) is True
    assert hash(poly) == hash(F_pPolynomial.one(p))
    assert len({poly, F_pPolynomial.one(p)}) == 1


@pytest.mark.parametrize("p", NON_PRIMES[:6])
def test_one_non_prime_p_raises_value_error(p):
    with pytest.raises(ValueError, match="p must be prime"):
        F_pPolynomial.one(p)


@pytest.mark.parametrize("p", PRIMES[:5])
def test_one_returns_new_object_each_call(p):
    assert F_pPolynomial.one(p) is not F_pPolynomial.one(p)


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_one_is_multiplicative_identity(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly * F_pPolynomial.one(p) == poly
    assert F_pPolynomial.one(p) * poly == poly


@pytest.mark.parametrize("p", PRIMES)
def test_one_times_one_is_one(p):
    assert F_pPolynomial.one(p) * F_pPolynomial.one(p) == F_pPolynomial.one(p)
    assert (F_pPolynomial.one(p) * F_pPolynomial.one(p)).degree == 0


@pytest.mark.parametrize("p", PRIMES)
def test_one_evaluation_is_one(p):
    one = F_pPolynomial.one(p)
    for value in range(min(p, 5)):
        assert one(F_pElement(p, value)) == F_pElement.one(p)


@pytest.mark.parametrize("p", PRIMES)
def test_negation_of_one(p):
    assert -F_pPolynomial.one(p) == F_pPolynomial(p, [F_pElement(p, -1)])
    assert _values(-F_pPolynomial.one(p)) == [(p - 1) % p]


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (11, 13), (13, 101),
])
def test_one_of_different_fields_are_distinct(p1, p2):
    assert F_pPolynomial.one(p1) != F_pPolynomial.one(p2)
    assert len({F_pPolynomial.one(p1), F_pPolynomial.one(p2)}) == 2


@pytest.mark.parametrize("p", PRIMES[1:])
def test_one_plus_one(p):
    result = F_pPolynomial.one(p) + F_pPolynomial.one(p)
    assert result == F_pPolynomial(p, [F_pElement(p, 2)])
    assert result.degree == 0


@pytest.mark.parametrize("p", PRIMES)
def test_one_minus_one_is_zero_polynomial(p):
    result = F_pPolynomial.one(p) - F_pPolynomial.one(p)
    assert result == F_pPolynomial(p)
    assert bool(result) is False


# ===========================================================================
# 18. Умножение и деление на константу (элемент поля), __rmul__
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, scalar, expected", [
    (5, [1, 2], 2, [2, 4]),
    (5, [3, 4], 3, [4, 2]),
    (5, [1, 2, 3], 1, [1, 2, 3]),
    (7, [1, 2, 3], 2, [2, 4, 6]),
    (7, [6, 5], 6, [1, 2]),
    (11, [1, 2], 10, [10, 9]),
    (13, [1, 0, 2], 12, [12, 0, 11]),
    (2, [1, 1], 1, [1, 1]),
])
def test_mul_by_element_scales_coefficients(make_poly, p, coeffs, scalar, expected):
    result = make_poly(p, coeffs) * F_pElement(p, scalar)
    assert _values(result) == expected
    assert result.p == p


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_mul_by_zero_element_gives_zero_polynomial(make_poly, p, coeffs):
    result = make_poly(p, coeffs) * F_pElement.zero(p)
    assert result == F_pPolynomial(p)
    assert bool(result) is False
    assert result.degree == float('-inf')


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_mul_by_one_element_is_identity(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert poly * F_pElement.one(p) == poly


@pytest.mark.parametrize("p, coeffs, scalar", [
    (5, [1, 2], 3), (7, [3, 4], 5), (11, [9, 8], 7), (13, [1, 0, 2], 6),
    (2, [1, 1], 1),
])
def test_element_times_polynomial_is_scalar_multiplication(make_poly, p, coeffs, scalar):
    # нюанс: F_pElement.__mul__ для чужого типа выполняет `other * self`,
    # поэтому element * poly даёт то же, что poly * element
    poly = make_poly(p, coeffs)
    element = F_pElement(p, scalar)
    assert element * poly == poly * element


@pytest.mark.parametrize("p, coeffs, scalar, expected", [
    (5, [1, 2], 2, [3, 1]),
    (5, [3, 4], 4, [2, 1]),
    (7, [1, 2, 3], 2, [4, 1, 5]),
    (11, [1, 2], 10, [10, 9]),
    (13, [1, 0, 2], 12, [12, 0, 11]),
])
def test_truediv_by_element_scales_by_inverse(
        make_poly, p, coeffs, scalar, expected):
    result = make_poly(p, coeffs) / F_pElement(p, scalar)
    assert _values(result) == expected
    assert result.p == p


@pytest.mark.parametrize("p, coeffs, scalar", [
    (5, [1, 2], 2), (7, [3, 4], 5), (11, [9, 8], 7), (13, [1, 0, 2], 6),
    (101, [7, 8], 50),
])
def test_mul_then_div_restores_polynomial(make_poly, p, coeffs, scalar):
    poly = make_poly(p, coeffs)
    element = F_pElement(p, scalar)
    assert (poly * element) / element == poly
    assert (poly / element) * element == poly


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_truediv_by_zero_element_raises_zero_division_error(make_poly, p, coeffs):
    with pytest.raises(ZeroDivisionError, match="cannot divide by zero"):
        make_poly(p, coeffs) / F_pElement.zero(p)


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101),
])
def test_mul_by_element_of_other_field_raises_type_error(make_poly, p, other_p):
    with pytest.raises(
            TypeError,
            match="cannot multiply polynomial to element in different field"):
        make_poly(p, [1, 2]) * F_pElement(other_p, 2)


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101),
])
def test_div_by_element_of_other_field_raises_type_error(make_poly, p, other_p):
    with pytest.raises(
            TypeError,
            match="cannot multiply polynomial to element in different field"):
        make_poly(p, [1, 2]) / F_pElement(other_p, 2)


@pytest.mark.parametrize("p, a, b, scalar", [
    (5, [1, 2], [3, 4], 2), (7, [1], [2, 3], 5), (11, [1, 2], [3, 4], 9),
    (13, [4], [5, 6], 7), (2, [1, 1], [1], 1),
])
def test_scalar_mul_distributes_over_addition(make_poly, p, a, b, scalar):
    first = make_poly(p, a)
    second = make_poly(p, b)
    element = F_pElement(p, scalar)
    assert (first + second) * element == first * element + second * element


@pytest.mark.parametrize("p, a, b, scalar", [
    (5, [1, 2], [3, 4], 3), (7, [1], [2, 3], 6), (11, [1, 2], [3], 4),
    (13, [4], [5, 6], 12), (2, [1, 1], [1], 1),
])
def test_scalar_mul_commutes_with_polynomial_mul(make_poly, p, a, b, scalar):
    first = make_poly(p, a)
    second = make_poly(p, b)
    element = F_pElement(p, scalar)
    assert (first * element) * second == (first * second) * element


@pytest.mark.parametrize("p, coeffs, scalar, x", [
    (5, [1, 2], 3, 2), (7, [3, 4], 5, 1), (11, [9, 8], 7, 4),
    (13, [1, 0, 2], 6, 3), (2, [1, 1], 1, 1),
])
def test_scalar_mul_agrees_with_evaluation(make_poly, p, coeffs, scalar, x):
    poly = make_poly(p, coeffs)
    element = F_pElement(p, scalar)
    point = F_pElement(p, x)
    assert (poly * element)(point) == poly(point) * element


@pytest.mark.parametrize("p, coeffs", [
    (5, [1, 2]), (7, [3, 4]), (2, [1, 1]), (11, [9, 8]), (13, [1, 0, 2]),
])
def test_scalar_mul_preserves_degree(make_poly, p, coeffs):
    poly = make_poly(p, coeffs)
    assert (poly * F_pElement.one(p)).degree == poly.degree


@pytest.mark.parametrize("p, coeffs, scalar", [
    (5, [1, 2], 2), (7, [3, 4], 5), (11, [9, 8], 7), (13, [1, 0, 2], 6),
    (101, [7, 8], 50),
])
def test_scalar_operations_do_not_mutate_polynomial(make_poly, p, coeffs, scalar):
    poly = make_poly(p, coeffs)
    element = F_pElement(p, scalar)
    _ = poly * element
    _ = poly / element
    assert poly == make_poly(p, coeffs)
    assert _values(poly) == [c % p for c in coeffs]


@pytest.mark.parametrize("prime", PRIMES)
def test_scalar_operations_on_zero_polynomial(prime):
    p = prime
    zero = F_pPolynomial(p)
    assert zero * F_pElement.one(p) == zero
    assert zero / F_pElement.one(p) == zero


@pytest.mark.parametrize("other", [0, 5, "x", None, 3.5, (1, 2), [1]])
def test_scalar_division_by_non_element_raises_type_error(make_poly, other):
    with pytest.raises(TypeError):
        make_poly(5, [1, 2]) / other


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 3), (11, 13),
])
def test_different_fields_message_mentions_both_fields(make_poly, p1, p2):
    cases = (
        ("+", lambda: make_poly(p1, [1]) + make_poly(p2, [1])),
        ("-", lambda: make_poly(p1, [1]) - make_poly(p2, [1])),
        ("*", lambda: make_poly(p1, [1]) * make_poly(p2, [1])),
    )
    for operation, expression in cases:
        with pytest.raises(TypeError) as exc_info:
            expression()
        message = str(exc_info.value)
        assert (f"cannot {operation} with two polynomials in "
                "different fields") in message
        assert f"F_{p1} and F_{p2}" in message


@pytest.mark.parametrize("p, other_p", [
    (5, 7), (7, 5), (2, 3), (11, 13), (13, 101),
])
def test_scalar_mul_message_mentions_different_field(make_poly, p, other_p):
    with pytest.raises(TypeError) as exc_info:
        make_poly(p, [1, 2]) * F_pElement(other_p, 2)
    assert "cannot multiply polynomial to element in different field" in str(
        exc_info.value)
