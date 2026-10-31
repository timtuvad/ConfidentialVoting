"""Тесты модуля src/core/algebra/field.py — класс F_pElement (элемент поля F_p).

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__          — валидация простоты p, приведение value по модулю, value=None;
    * p, value          — свойства только для чтения;
    * zero, unit        — статические конструкторы;
    * randomGenerating  — случайный элемент, частный случай p == 2, непростое p;
    * __str__, __repr__ — строковое и отладочное представление;
    * __eq__, __ne__    — сравнение, в т.ч. с чужими типами, подклассом и чужим полем;
    * __hash__          — хешируемость, согласованность с __eq__;
    * __pos__, __neg__  — унарные операции;
    * __add__, __sub__, __mul__, __truediv__ — арифметика, TypeError для чужих типов
      и для разных полей;
    * __pow__, __invert__ — возведение в степень (включая отрицательную и проверку
      типа показателя) и мультипликативное обращение;
    * служебные нюансы — отсутствие __bool__/__int__/__radd__, неизменяемость
      операндов, замкнутость операций, законы поля F_p.
"""

import pytest

from src import F_pElement

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

PRIMES = [2, 3, 5, 7, 11, 13, 101, 10 ** 9 + 7]

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 101]

NON_PRIMES = [
    -7, -1, 0, 1, 4, 6, 8, 9, 10, 12, 15, 21, 25, 27, 100, 589423, 954529,
]

NON_ELEMENT_VALUES = [0, 5, -3, "5", None, 3.5, (5,), [5], {5}, object()]


class SubElement(F_pElement):
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
def a5(make_element):
    """Элемент (2 mod 5)."""
    return make_element(5, 2)


@pytest.fixture
def b5(make_element):
    """Элемент (3 mod 5)."""
    return make_element(5, 3)


@pytest.fixture
def a7(make_element):
    """Элемент (3 mod 7) — поле, отличное от F_5."""
    return make_element(7, 3)


@pytest.fixture
def zero5():
    """Ноль поля F_5."""
    return F_pElement.zero(5)


@pytest.fixture
def unit5():
    """Единица поля F_5."""
    return F_pElement.unit(5)


# ===========================================================================
# 1. __init__ — создание элемента
# ===========================================================================

@pytest.mark.parametrize("p, value, expected", [
    (2, 0, 0),
    (2, 1, 1),
    (2, 2, 0),
    (2, 3, 1),
    (2, -1, 1),
    (2, -2, 0),
    (5, 0, 0),
    (5, 5, 0),
    (5, 7, 2),
    (5, 12, 2),
    (5, -1, 4),
    (5, -7, 3),
    (7, 14, 0),
    (7, 15, 1),
    (7, -1, 6),
    (7, -8, 6),
    (11, 22, 0),
    (11, 23, 1),
    (11, -1, 10),
    (13, 26, 0),
    (13, 27, 1),
    (13, -1, 12),
    (101, 202, 0),
    (101, 10 ** 6, 100),
    (10 ** 9 + 7, 10 ** 9 + 8, 1),
    (10 ** 9 + 7, -1, 10 ** 9 + 6),
])
def test_init_reduces_value_modulo_p(make_element, p, value, expected):
    assert make_element(p, value).value == expected


@pytest.mark.parametrize("p, value", [
    (2, -100), (2, 12345), (3, -7), (5, -1), (5, 10 ** 6), (7, -13),
    (11, 12345), (13, -987), (101, 10 ** 9), (10 ** 9 + 7, -(10 ** 9)),
])
def test_init_value_always_lies_in_range(make_element, p, value):
    el = make_element(p, value)
    assert 0 <= el.value < p


@pytest.mark.parametrize("p, value", [
    (2, 1), (3, -4), (5, 17), (7, -29), (11, 123), (13, -456),
    (101, 99999), (10 ** 9 + 7, 10 ** 12 + 5),
])
def test_init_value_equals_python_mod(make_element, p, value):
    assert make_element(p, value).value == value % p


@pytest.mark.parametrize("p", PRIMES)
def test_init_default_value_is_zero(make_element, p):
    el = make_element(p)
    assert el.value == 0
    assert el == F_pElement(p, 0)
    assert el == F_pElement.zero(p)


@pytest.mark.parametrize("p", PRIMES)
def test_init_explicit_none_value_is_zero(make_element, p):
    assert make_element(p, None) == F_pElement(p, 0)


@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 3), (7, 0), (11, 10), (13, 5), (101, 99), (10 ** 9 + 7, 1),
])
def test_init_stores_prime_unchanged(make_element, p, value):
    assert make_element(p, value).p == p


@pytest.mark.parametrize("p", NON_PRIMES)
def test_init_non_prime_p_raises_value_error(make_element, p):
    with pytest.raises(ValueError, match=f"p must be prime, cannot be {p}"):
        make_element(p)


@pytest.mark.parametrize("p", NON_PRIMES)
def test_init_non_prime_p_raises_value_error_even_with_value(make_element, p):
    with pytest.raises(ValueError, match="p must be prime"):
        make_element(p, 1)


@pytest.mark.parametrize("p", [4, 6, 9, 15, 100])
def test_init_error_message_mentions_bad_p(make_element, p):
    with pytest.raises(ValueError) as exc_info:
        make_element(p, 3)
    assert str(p) in str(exc_info.value)


@pytest.mark.parametrize("p, value", [
    (2, 0), (5, 0), (5, 4), (7, 6), (11, 0), (13, 12), (101, 100),
])
def test_init_returns_instance_of_f_p_element(make_element, p, value):
    el = make_element(p, value)
    assert isinstance(el, F_pElement)
    assert type(el) is F_pElement


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 2), (11, 10), (13, 1), (101, 55),
])
def test_init_does_not_mutate_passed_value(make_element, p, value):
    original = value
    make_element(p, value)
    assert value == original


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 2), (11, 10), (13, 1), (101, 55),
])
def test_init_accepts_keyword_value(make_element, p, value):
    assert F_pElement(p, value=value) == make_element(p, value)


@pytest.mark.parametrize("value", ["2", [1], object()])
def test_init_non_numeric_value_raises_type_error(make_element, value):
    # нюанс: тип value не проверяется, "2" % p -> TypeError
    with pytest.raises(TypeError):
        make_element(5, value)


@pytest.mark.parametrize("value", [2.5, -1.5, 7.0])
def test_init_float_value_is_not_validated(make_element, value):
    # нюанс: value не проверяется на принадлежность int — float принимается как есть
    el = make_element(5, value)
    assert el.value == value % 5


# ===========================================================================
# 2. Свойства p и value — только для чтения
# ===========================================================================

@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 3), (7, 0), (11, 10), (13, 5), (101, 99),
])
def test_p_property_is_read_only(make_element, p, value):
    el = make_element(p, value)
    with pytest.raises(AttributeError):
        el.p = 3


@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 3), (7, 0), (11, 10), (13, 5), (101, 99),
])
def test_value_property_is_read_only(make_element, p, value):
    el = make_element(p, value)
    with pytest.raises(AttributeError):
        el.value = 0


@pytest.mark.parametrize("p, value", [
    (5, 0), (5, 4), (7, 6), (11, 0), (13, 12), (101, 7),
])
def test_p_property_is_stable_across_calls(make_element, p, value):
    el = make_element(p, value)
    assert el.p == el.p == p


@pytest.mark.parametrize("p, value", [
    (5, 1), (5, 4), (7, 3), (11, 9), (13, 0), (101, 100),
])
def test_value_property_is_stable_across_calls(make_element, p, value):
    el = make_element(p, value)
    assert el.value == el.value == value % p


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_public_api_is_properties_and_static_constructors(make_element, p, value):
    # нюанс инкапсуляции: состояние скрыто, публично доступны только свойства
    # p/value и статические конструкторы zero/unit/randomGenerating
    el = make_element(p, value)
    public_names = {name for name in dir(el) if not name.startswith("_")}
    assert public_names == {"p", "value", "zero", "unit", "randomGenerating"}
    assert not hasattr(el, "__p")
    assert not hasattr(el, "__value")


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_properties_agree_with_str(make_element, p, value):
    el = make_element(p, value)
    assert str(el) == f"({el.value} mod {el.p})"


# ===========================================================================
# 3. zero(p) — статический конструктор нуля
# ===========================================================================

@pytest.mark.parametrize("p", PRIMES)
def test_zero_has_zero_value(p):
    zero = F_pElement.zero(p)
    assert isinstance(zero, F_pElement)
    assert zero.value == 0


@pytest.mark.parametrize("p", PRIMES)
def test_zero_stores_prime(p):
    assert F_pElement.zero(p).p == p


@pytest.mark.parametrize("p", PRIMES)
def test_zero_equals_constructor_zero(p):
    assert F_pElement.zero(p) == F_pElement(p, 0)
    assert F_pElement.zero(p) == F_pElement(p, p)


@pytest.mark.parametrize("p", PRIMES)
def test_zero_string_representation(p):
    assert str(F_pElement.zero(p)) == f"(0 mod {p})"


@pytest.mark.parametrize("p", NON_PRIMES[:6])
def test_zero_non_prime_raises_value_error(p):
    with pytest.raises(ValueError, match="p must be prime"):
        F_pElement.zero(p)


@pytest.mark.parametrize("p", [2, 5, 7, 11, 13])
def test_zero_returns_new_object_each_call(p):
    assert F_pElement.zero(p) is not F_pElement.zero(p)


@pytest.mark.parametrize("p, value", [
    (5, 1), (5, 4), (7, 0), (11, 6), (13, 12), (2, 1),
])
def test_zero_is_additive_identity(make_element, p, value):
    a = make_element(p, value)
    zero = F_pElement.zero(p)
    assert a + zero == a
    assert zero + a == a


@pytest.mark.parametrize("p1, p2", [
    (2, 3), (3, 5), (5, 7), (7, 11), (11, 13), (13, 101),
])
def test_zero_and_unit_of_different_fields_are_not_equal(p1, p2):
    assert F_pElement.zero(p1) != F_pElement.zero(p2)
    assert F_pElement.unit(p1) != F_pElement.unit(p2)
    assert F_pElement.zero(p1) != F_pElement.unit(p2)


# ===========================================================================
# 4. unit(p) — статический конструктор единицы
# ===========================================================================

@pytest.mark.parametrize("p", PRIMES)
def test_unit_has_value_one(p):
    unit = F_pElement.unit(p)
    assert isinstance(unit, F_pElement)
    assert unit.value == 1 % p


@pytest.mark.parametrize("p", PRIMES)
def test_unit_stores_prime(p):
    assert F_pElement.unit(p).p == p


@pytest.mark.parametrize("p", PRIMES)
def test_unit_equals_constructor_one(p):
    assert F_pElement.unit(p) == F_pElement(p, 1)
    assert F_pElement.unit(p) == F_pElement(p, p + 1)


@pytest.mark.parametrize("p", PRIMES)
def test_unit_string_representation(p):
    assert str(F_pElement.unit(p)) == f"(1 mod {p})"


@pytest.mark.parametrize("p", NON_PRIMES[:6])
def test_unit_non_prime_raises_value_error(p):
    with pytest.raises(ValueError, match="p must be prime"):
        F_pElement.unit(p)


@pytest.mark.parametrize("p, value", [
    (5, 0), (5, 3), (7, 6), (11, 1), (13, 7), (2, 1),
])
def test_unit_is_multiplicative_identity(make_element, p, value):
    a = make_element(p, value)
    unit = F_pElement.unit(p)
    assert a * unit == a
    assert unit * a == a


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_unit_inverse_is_itself(p):
    assert ~F_pElement.unit(p) == F_pElement.unit(p)


@pytest.mark.parametrize("p", [2, 5, 7, 11, 13])
def test_unit_returns_new_object_each_call(p):
    assert F_pElement.unit(p) is not F_pElement.unit(p)


# ===========================================================================
# 5. randomGenerating(p) — случайный элемент
# ===========================================================================

@pytest.mark.parametrize("p", [3, 5, 7, 11, 13, 101, 10 ** 9 + 7])
def test_random_generating_returns_element_of_field(p):
    for _ in range(15):
        el = F_pElement.randomGenerating(p)
        assert isinstance(el, F_pElement)
        assert el.p == p


@pytest.mark.parametrize("p", [3, 5, 7, 11, 13, 101, 10 ** 9 + 7])
def test_random_generating_value_in_range_from_two_to_p(p):
    # нюанс реализации: randrange(2, p) — значения 0 и 1 не генерируются
    for _ in range(30):
        value = F_pElement.randomGenerating(p).value
        assert 2 <= value < p


@pytest.mark.parametrize("p", [5, 7, 11, 13, 101])
def test_random_generating_never_returns_zero_or_one(p):
    values = {F_pElement.randomGenerating(p).value for _ in range(200)}
    assert 0 not in values
    assert 1 not in values


@pytest.mark.parametrize("count", [1, 5, 20, 50])
def test_random_generating_for_p_two_returns_unit(count):
    for _ in range(count):
        el = F_pElement.randomGenerating(2)
        assert el == F_pElement.unit(2)
        assert el.value == 1


@pytest.mark.parametrize("p", NON_PRIMES)
def test_random_generating_non_prime_raises_value_error(p):
    with pytest.raises(ValueError, match=f"p must be prime, cannot be {p}"):
        F_pElement.randomGenerating(p)


def test_random_generating_produces_different_values():
    values = {F_pElement.randomGenerating(101).value for _ in range(300)}
    assert len(values) > 1


@pytest.mark.parametrize("p", [5, 7, 11, 13, 101])
def test_random_generating_returns_new_object_each_call(p):
    assert F_pElement.randomGenerating(p) is not F_pElement.randomGenerating(p)


@pytest.mark.parametrize("p", [5, 7, 11, 13])
def test_random_generating_result_is_usable_in_arithmetic(p):
    el = F_pElement.randomGenerating(p)
    assert isinstance(el + el, F_pElement)
    assert isinstance(el * el, F_pElement)
    assert isinstance(-el, F_pElement)
    assert isinstance(el ** 3, F_pElement)
    assert isinstance(~el, F_pElement)


@pytest.mark.parametrize("p", [5, 7, 11, 13, 101])
def test_random_generating_covers_many_residues(p):
    values = {F_pElement.randomGenerating(p).value for _ in range(400)}
    assert values <= set(range(2, p))
    assert len(values) > 1
    if p == 101:
        # 400 выборок из 99 возможных значений покрывают заметную часть поля
        assert len(values) > 20


# ===========================================================================
# 6. __str__ / __repr__
# ===========================================================================

@pytest.mark.parametrize("p, value, expected", [
    (2, 0, "(0 mod 2)"),
    (2, 1, "(1 mod 2)"),
    (5, 0, "(0 mod 5)"),
    (5, 4, "(4 mod 5)"),
    (5, 7, "(2 mod 5)"),
    (7, -1, "(6 mod 7)"),
    (13, 27, "(1 mod 13)"),
    (101, 10 ** 6, "(100 mod 101)"),
])
def test_str_format(make_element, p, value, expected):
    assert str(make_element(p, value)) == expected


@pytest.mark.parametrize("p, value, expected", [
    (5, 0, "(0 mod 5)"),
    (7, 0, "(0 mod 7)"),
    (11, 11, "(0 mod 11)"),
    (13, -13, "(0 mod 13)"),
    (2, 2, "(0 mod 2)"),
])
def test_str_of_zero_elements(make_element, p, value, expected):
    assert str(make_element(p, value)) == expected


@pytest.mark.parametrize("p, value, expected", [
    (5, 12, "(2 mod 5)"),
    (7, 20, "(6 mod 7)"),
    (11, 34, "(1 mod 11)"),
    (13, 100, "(9 mod 13)"),
    (101, 12345, "(23 mod 101)"),
])
def test_str_uses_reduced_value(make_element, p, value, expected):
    assert str(make_element(p, value)) == expected


@pytest.mark.parametrize("p, value, expected", [
    (5, -1, "(4 mod 5)"),
    (7, -2, "(5 mod 7)"),
    (11, -12, "(10 mod 11)"),
    (13, -1, "(12 mod 13)"),
    (2, -5, "(1 mod 2)"),
])
def test_str_of_negative_values(make_element, p, value, expected):
    assert str(make_element(p, value)) == expected


@pytest.mark.parametrize("p, value, expected", [
    (2, 0, "F_pElement(p=2, value=0)"),
    (2, 1, "F_pElement(p=2, value=1)"),
    (5, 3, "F_pElement(p=5, value=3)"),
    (5, 7, "F_pElement(p=5, value=2)"),
    (7, -1, "F_pElement(p=7, value=6)"),
    (13, 27, "F_pElement(p=13, value=1)"),
    (101, 10 ** 6, "F_pElement(p=101, value=100)"),
])
def test_repr_format(make_element, p, value, expected):
    assert repr(make_element(p, value)) == expected


@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 3), (7, 6), (11, 10), (13, 0), (101, 42),
])
def test_repr_is_consistent_with_properties(make_element, p, value):
    el = make_element(p, value)
    assert repr(el) == f"F_pElement(p={el.p}, value={el.value})"


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_repr_differs_from_str(make_element, p, value):
    el = make_element(p, value)
    assert repr(el) != str(el)


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_str_used_by_format(make_element, p, value):
    el = make_element(p, value)
    assert f"{el}" == f"({el.value} mod {p})"
    assert "{}".format(el) == f"({el.value} mod {p})"


@pytest.mark.parametrize("p, value", [
    (5, 1), (5, 4), (7, 0), (11, 6), (13, 12),
])
def test_str_of_equal_elements_is_equal(make_element, p, value):
    a = make_element(p, value)
    b = make_element(p, value + p)
    assert a == b
    assert str(a) == str(b)
    assert repr(a) == repr(b)


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 3), (11, 4), (13, 5), (101, 9),
])
def test_repr_of_list_of_elements(make_element, p, value):
    elements = [make_element(p, value), F_pElement.zero(p)]
    assert repr(elements) == (f"[F_pElement(p={p}, value={value % p}), "
                              f"F_pElement(p={p}, value=0)]")


# ===========================================================================
# 7. __eq__ / __ne__
# ===========================================================================

@pytest.mark.parametrize("p, v1, v2", [
    (5, 0, 0),
    (5, 3, 3),
    (5, 3, 8),
    (5, 4, -1),
    (7, 6, -1),
    (2, 1, 3),
    (11, 10, -1),
    (13, 1, 14),
    (101, 100, -1),
])
def test_eq_true_for_equal_elements(make_element, p, v1, v2):
    assert make_element(p, v1) == make_element(p, v2)


@pytest.mark.parametrize("p, v1, v2", [
    (5, 0, 1),
    (5, 3, 4),
    (7, 1, 2),
    (2, 0, 1),
    (11, 5, 6),
    (13, 12, 1),
    (101, 0, 100),
])
def test_eq_false_for_different_values(make_element, p, v1, v2):
    assert not (make_element(p, v1) == make_element(p, v2))


@pytest.mark.parametrize("p1, p2, value", [
    (5, 7, 1),
    (2, 3, 1),
    (5, 101, 0),
    (7, 11, 3),
    (13, 101, 5),
    (3, 5, 2),
])
def test_eq_false_for_elements_of_different_fields(make_element, p1, p2, value):
    assert not (make_element(p1, value) == make_element(p2, value))
    assert make_element(p1, value) != make_element(p2, value)


@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 0), (7, 3), (11, 10), (13, 12), (101, 1),
])
def test_eq_is_reflexive(make_element, p, value):
    el = make_element(p, value)
    assert el == el


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 4), (7, 2, 5), (11, 3, 8), (13, 4, 9), (101, 10, 20),
])
def test_eq_is_symmetric(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    assert (a == b) == (b == a)
    assert (a != b) == (b != a)


@pytest.mark.parametrize("p, v1, v2, v3", [
    (5, 1, 1, 2), (7, 3, 3, 4), (11, 5, 5, 6), (13, 7, 7, 8), (101, 9, 9, 10),
])
def test_eq_is_transitive(make_element, p, v1, v2, v3):
    a, b, c = make_element(p, v1), make_element(p, v2), make_element(p, v3)
    assert a == b
    assert a != c
    assert b != c


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES)
def test_eq_with_non_element_is_false(make_element, other):
    # новая реализация: проверка type(self) is not type(other) -> False
    assert (make_element(5, 2) == other) is False


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES)
def test_ne_with_non_element_is_true(make_element, other):
    assert (make_element(5, 2) != other) is True


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES[:6])
def test_eq_reflected_with_non_element_is_false(make_element, other):
    assert (other == make_element(5, 2)) is False
    assert (other != make_element(5, 2)) is True


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_eq_with_subclass_is_false(make_element, p, value):
    # нюанс: используется type(self) is not type(other), а не isinstance
    sub = SubElement(p, value)
    assert (make_element(p, value) == sub) is False
    assert (sub == make_element(p, value)) is False
    assert make_element(p, value) != sub


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_eq_between_subclass_instances_works(p, value):
    assert SubElement(p, value) == SubElement(p, value)
    assert SubElement(p, value) != SubElement(p, value + 1)


@pytest.mark.parametrize("p, v1, v2", [
    (5, 0, 1),
    (5, 2, 3),
    (7, 1, 6),
    (2, 0, 1),
    (11, 4, 5),
    (13, 0, 12),
])
def test_ne_true_for_different_elements(make_element, p, v1, v2):
    assert make_element(p, v1) != make_element(p, v2)


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 1),
    (5, 1, 6),
    (7, 6, -1),
    (11, 10, -1),
    (13, 2, 15),
])
def test_ne_false_for_equal_elements(make_element, p, v1, v2):
    assert not (make_element(p, v1) != make_element(p, v2))


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_eq_result_is_bool(make_element, p, value):
    a = make_element(p, value)
    b = make_element(p, value)
    assert (a == b) is True
    assert (a != b) is False


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_operations_preserve_equality(make_element, p, value):
    a = make_element(p, value)
    b = make_element(p, value)
    assert a + a == b + b
    assert a * a == b * b
    assert -a == -b
    assert a ** 3 == b ** 3


# ===========================================================================
# 8. __hash__
# ===========================================================================

@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 0), (7, 3), (11, 10), (13, 12), (101, 1),
])
def test_element_is_hashable(make_element, p, value):
    el = make_element(p, value)
    assert isinstance(hash(el), int)
    assert hash(el) == hash(el)


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_hash_equals_hash_of_pair(make_element, p, value):
    assert hash(make_element(p, value)) == hash((p, value % p))


@pytest.mark.parametrize("p, value", [
    (5, 1), (5, 4), (7, 6), (11, 10), (13, 12),
])
def test_equal_elements_have_equal_hashes(make_element, p, value):
    assert hash(make_element(p, value)) == hash(make_element(p, value + p))


@pytest.mark.parametrize("p", [5, 7, 11, 13, 101])
def test_all_residues_have_distinct_hashes_in_set(p):
    residues = {F_pElement(p, v) for v in range(p)}
    assert len(residues) == p


@pytest.mark.parametrize("p", [5, 7, 11, 13])
def test_element_can_be_dict_key(p):
    mapping = {F_pElement(p, v): v for v in range(p)}
    assert len(mapping) == p
    for v in range(p):
        assert mapping[F_pElement(p, v)] == v


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_membership_in_set(make_element, p, value):
    collection = {make_element(p, value)}
    assert make_element(p, value) in collection
    assert make_element(p, value + 1) not in collection
    assert 5 not in collection


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 3),
])
def test_elements_of_different_fields_coexist_in_set(make_element, p1, p2):
    collection = {make_element(p1, 1), make_element(p2, 1)}
    assert len(collection) == 2


# ===========================================================================
# 9. __pos__ / __neg__
# ===========================================================================

@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 0), (5, 4), (7, 3), (11, 10), (13, 12), (101, 99),
])
def test_pos_returns_same_object(make_element, p, value):
    el = make_element(p, value)
    assert (+el) is el


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_pos_preserves_value_and_p(make_element, p, value):
    el = make_element(p, value)
    assert (+el) == el
    assert (+el).p == el.p
    assert (+el).value == el.value


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_pos_is_idempotent(make_element, p, value):
    el = make_element(p, value)
    assert +(+el) is el


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 0), (11, 6), (13, 12), (2, 1),
])
def test_pos_keeps_equality(make_element, p, value):
    el = make_element(p, value)
    assert +el == el
    assert not (+el != el)


@pytest.mark.parametrize("p, value, expected", [
    (2, 0, 0),
    (2, 1, 1),
    (5, 0, 0),
    (5, 1, 4),
    (5, 2, 3),
    (5, 4, 1),
    (7, 1, 6),
    (7, 3, 4),
    (7, 6, 1),
    (11, 5, 6),
    (11, 10, 1),
    (13, 7, 6),
    (13, 12, 1),
    (101, 1, 100),
    (101, 100, 1),
])
def test_neg_value(make_element, p, value, expected):
    assert (-make_element(p, value)).value == expected


@pytest.mark.parametrize("p", PRIMES)
def test_neg_of_zero_is_zero(make_element, p):
    assert -F_pElement.zero(p) == F_pElement.zero(p)


@pytest.mark.parametrize("p, value", [
    (5, 1), (5, 4), (7, 3), (11, 7), (13, 5), (101, 42),
])
def test_double_neg_returns_equal_element(make_element, p, value):
    el = make_element(p, value)
    assert -(-el) == el


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 5), (11, 3), (13, 11), (101, 17),
])
def test_element_plus_its_negation_is_zero(make_element, p, value):
    el = make_element(p, value)
    assert el + (-el) == F_pElement.zero(p)
    assert (-el) + el == F_pElement.zero(p)


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 2), (11, 9), (13, 4), (101, 1),
])
def test_neg_returns_new_element_of_same_field(make_element, p, value):
    el = make_element(p, value)
    negated = -el
    assert isinstance(negated, F_pElement)
    assert negated is not el
    assert negated.p == p


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_neg_of_sum_equals_sum_of_negs(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    assert -(a + b) == (-a) + (-b)


# ===========================================================================
# 10. __add__
# ===========================================================================

@pytest.mark.parametrize("p, v1, v2, expected", [
    (2, 0, 0, 0),
    (2, 1, 1, 0),
    (2, 1, 0, 1),
    (5, 2, 3, 0),
    (5, 4, 4, 3),
    (5, 4, 1, 0),
    (5, 0, 0, 0),
    (7, 3, 5, 1),
    (7, 6, 6, 5),
    (7, 6, 1, 0),
    (11, 5, 6, 0),
    (11, 10, 10, 9),
    (13, 6, 7, 0),
    (13, 12, 12, 11),
    (101, 100, 100, 99),
])
def test_add_reduces_sum_modulo_p(make_element, p, v1, v2, expected):
    assert (make_element(p, v1) + make_element(p, v2)).value == expected


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 2), (5, 4, 3), (7, 2, 6), (11, 5, 9), (13, 7, 3), (101, 50, 60),
])
def test_add_is_commutative(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    assert a + b == b + a


@pytest.mark.parametrize("p, v1, v2, v3", [
    (5, 1, 2, 3), (7, 2, 3, 4), (11, 5, 6, 7), (13, 1, 12, 6), (101, 30, 40, 50),
])
def test_add_is_associative(make_element, p, v1, v2, v3):
    a, b, c = make_element(p, v1), make_element(p, v2), make_element(p, v3)
    assert (a + b) + c == a + (b + c)


@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 4), (7, 6), (11, 10), (13, 0), (101, 99),
])
def test_add_zero_is_identity(make_element, p, value):
    a = make_element(p, value)
    assert a + F_pElement.zero(p) == a


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 17), (3, 5),
])
def test_add_different_fields_raises_type_error(make_element, p1, p2):
    with pytest.raises(TypeError, match="cannot add two elements in different fields"):
        make_element(p1, 1) + make_element(p2, 1)


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES)
def test_add_non_element_raises_type_error(make_element, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_element(5, 1) + other


@pytest.mark.parametrize("other", [1, 2, 5, 100, 0])
def test_int_plus_element_raises_type_error(make_element, other):
    # __radd__ не определён
    with pytest.raises(TypeError):
        other + make_element(5, 1)


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_add_with_subclass_raises_type_error(p, value):
    with pytest.raises(TypeError, match="unsupported operand type"):
        F_pElement(p, value) + SubElement(p, value)


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_add_returns_new_element_of_same_field(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    result = a + b
    assert isinstance(result, F_pElement)
    assert result is not a and result is not b
    assert result.p == p


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_add_does_not_mutate_operands(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    _ = a + b
    assert a == make_element(p, v1)
    assert b == make_element(p, v2)


# ===========================================================================
# 11. __sub__
# ===========================================================================

@pytest.mark.parametrize("p, v1, v2, expected", [
    (2, 1, 1, 0),
    (2, 0, 1, 1),
    (5, 3, 1, 2),
    (5, 1, 3, 3),
    (5, 0, 4, 1),
    (5, 4, 4, 0),
    (7, 3, 5, 5),
    (7, 0, 1, 6),
    (7, 6, 6, 0),
    (11, 3, 7, 7),
    (11, 0, 10, 1),
    (13, 2, 5, 10),
    (101, 1, 100, 2),
])
def test_sub_reduces_difference_modulo_p(make_element, p, v1, v2, expected):
    assert (make_element(p, v1) - make_element(p, v2)).value == expected


@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 3), (7, 6), (11, 0), (13, 12), (101, 77),
])
def test_sub_self_is_zero(make_element, p, value):
    a = make_element(p, value)
    assert a - a == F_pElement.zero(p)


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 4), (11, 9), (13, 3), (101, 55),
])
def test_sub_zero_is_identity(make_element, p, value):
    a = make_element(p, value)
    assert a - F_pElement.zero(p) == a
    assert F_pElement.zero(p) - a == -a


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 3), (7, 2, 5), (11, 4, 9), (13, 6, 1), (101, 20, 80),
])
def test_sub_equals_add_of_negation(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    assert (a - b) == (a + (-b))


@pytest.mark.parametrize("p, v1, v2", [
    (5, 4, 1), (7, 1, 6), (11, 2, 9), (13, 3, 8), (101, 10, 90),
])
def test_sub_is_anti_commutative(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    assert (a - b) == -(b - a)


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 17), (3, 5),
])
def test_sub_different_fields_raises_type_error(make_element, p1, p2):
    # нюанс: __sub__ делегирует __add__, поэтому сообщение — про сложение
    with pytest.raises(TypeError, match="cannot add two elements in different fields"):
        make_element(p1, 1) - make_element(p2, 1)


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES)
def test_sub_non_element_raises_type_error(make_element, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_element(5, 1) - other


@pytest.mark.parametrize("other", [1, 2, 5, 0])
def test_int_minus_element_raises_type_error(make_element, other):
    with pytest.raises(TypeError):
        other - make_element(5, 1)


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_sub_returns_new_element_of_same_field(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    result = a - b
    assert isinstance(result, F_pElement)
    assert result is not a and result is not b
    assert result.p == p
    assert a == make_element(p, v1) and b == make_element(p, v2)


# ===========================================================================
# 12. __mul__
# ===========================================================================

@pytest.mark.parametrize("p, v1, v2, expected", [
    (2, 1, 1, 1),
    (2, 1, 0, 0),
    (5, 2, 3, 1),
    (5, 4, 4, 1),
    (5, 2, 4, 3),
    (5, 0, 4, 0),
    (7, 3, 5, 1),
    (7, 6, 6, 1),
    (7, 2, 4, 1),
    (11, 3, 4, 1),
    (11, 10, 10, 1),
    (11, 5, 9, 1),
    (13, 2, 7, 1),
    (13, 5, 8, 1),
    (13, 12, 12, 1),
    (101, 2, 51, 1),
])
def test_mul_reduces_product_modulo_p(make_element, p, v1, v2, expected):
    assert (make_element(p, v1) * make_element(p, v2)).value == expected


@pytest.mark.parametrize("p, v1, v2", [
    (5, 2, 3), (7, 4, 5), (11, 3, 9), (13, 6, 7), (101, 12, 34),
])
def test_mul_is_commutative(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    assert a * b == b * a


@pytest.mark.parametrize("p, v1, v2, v3", [
    (5, 1, 2, 3), (7, 2, 3, 5), (11, 4, 5, 6), (13, 3, 7, 11), (101, 8, 9, 10),
])
def test_mul_is_associative(make_element, p, v1, v2, v3):
    a, b, c = make_element(p, v1), make_element(p, v2), make_element(p, v3)
    assert (a * b) * c == a * (b * c)


@pytest.mark.parametrize("p, v1, v2, v3", [
    (5, 1, 2, 3), (7, 2, 3, 5), (11, 4, 5, 6), (13, 3, 7, 11), (101, 8, 9, 10),
])
def test_mul_distributes_over_add(make_element, p, v1, v2, v3):
    a, b, c = make_element(p, v1), make_element(p, v2), make_element(p, v3)
    assert a * (b + c) == (a * b) + (a * c)
    assert (b + c) * a == (b * a) + (c * a)


@pytest.mark.parametrize("p, value", [
    (2, 1), (5, 0), (7, 3), (11, 10), (13, 6), (101, 100),
])
def test_mul_by_zero_gives_zero(make_element, p, value):
    assert make_element(p, value) * F_pElement.zero(p) == F_pElement.zero(p)


@pytest.mark.parametrize("p, value", [
    (5, 4), (7, 6), (11, 0), (13, 12), (101, 55),
])
def test_mul_by_unit_is_identity(make_element, p, value):
    assert make_element(p, value) * F_pElement.unit(p) == make_element(p, value)


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 17), (3, 5),
])
def test_mul_different_fields_raises_type_error(make_element, p1, p2):
    with pytest.raises(TypeError,
                       match="cannot multiply two elements in different fields"):
        make_element(p1, 1) * make_element(p2, 1)


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES)
def test_mul_non_element_raises_type_error(make_element, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_element(5, 2) * other


@pytest.mark.parametrize("other", [2, 3, 5, 0, -1])
def test_int_times_element_raises_type_error(make_element, other):
    with pytest.raises(TypeError):
        other * make_element(5, 2)


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_mul_with_subclass_raises_type_error(p, value):
    with pytest.raises(TypeError, match="unsupported operand type"):
        F_pElement(p, value) * SubElement(p, value)


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_mul_returns_new_element_and_does_not_mutate(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    result = a * b
    assert isinstance(result, F_pElement)
    assert result is not a and result is not b
    assert result.p == p
    assert a == make_element(p, v1) and b == make_element(p, v2)


# ===========================================================================
# 13. __pow__
# ===========================================================================

@pytest.mark.parametrize("p, value, exp", [
    (5, 2, 0), (5, 0, 0), (7, 3, 1), (11, 5, 2), (13, 0, 5), (101, 17, 3),
])
def test_pow_returns_f_p_element(make_element, p, value, exp):
    result = make_element(p, value) ** exp
    assert isinstance(result, F_pElement)
    assert result.p == p


@pytest.mark.parametrize("p, value, exp, expected", [
    (5, 2, 2, 4),
    (5, 3, 3, 2),
    (5, 4, 2, 1),
    (7, 3, 2, 2),
    (7, 3, 3, 6),
    (7, 2, 5, 4),
    (7, 6, 2, 1),
    (11, 7, 2, 5),
    (11, 2, 3, 8),
    (13, 5, 2, 12),
    (13, 5, 3, 8),
    (101, 2, 10, 14),
    (101, 2, 20, 95),
    (2, 1, 100, 1),
])
def test_pow_positive_exponent(make_element, p, value, exp, expected):
    assert make_element(p, value) ** exp == make_element(p, expected)


@pytest.mark.parametrize("p, value", [
    (5, 0), (5, 3), (7, 6), (11, 0), (13, 12), (101, 1),
])
def test_pow_zero_exponent_is_unit(make_element, p, value):
    assert make_element(p, value) ** 0 == F_pElement.unit(p)


@pytest.mark.parametrize("p, value, expected", [
    (5, 2, 3),
    (5, 3, 2),
    (5, 4, 4),
    (7, 2, 4),
    (7, 3, 5),
    (7, 6, 6),
    (11, 2, 6),
    (11, 7, 8),
    (13, 2, 7),
    (13, 5, 8),
    (101, 2, 51),
    (101, 10, 91),
])
def test_pow_minus_one_is_modular_inverse(make_element, p, value, expected):
    assert make_element(p, value) ** -1 == make_element(p, expected)


@pytest.mark.parametrize("p, value, exp", [
    (5, 2, -2), (7, 3, -3), (11, 4, -5), (13, 5, -7), (101, 7, -11),
])
def test_pow_negative_exponent_matches_builtin_pow(make_element, p, value, exp):
    assert (make_element(p, value) ** exp).value == pow(value % p, exp, p)


@pytest.mark.parametrize("p", [5, 7, 11, 13, 101])
def test_pow_zero_base_negative_exponent_raises_value_error(make_element, p):
    with pytest.raises(ValueError):
        make_element(p, 0) ** -1


@pytest.mark.parametrize("exp", [2.0, 1.5, "2", None, [2], SubElement(5, 2)])
def test_pow_non_int_exponent_raises_type_error(make_element, exp):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_element(5, 2) ** exp


@pytest.mark.parametrize("exp", [True, False])
def test_pow_bool_exponent_raises_type_error(make_element, exp):
    # нюанс: type(exp) is not int -> bool не принимается
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_element(5, 2) ** exp


@pytest.mark.parametrize("p", PRIMES)
def test_fermat_little_theorem(make_element, p):
    for value in range(1, min(p, 10)):
        assert make_element(p, value) ** (p - 1) == F_pElement.unit(p)


@pytest.mark.parametrize("p", PRIMES)
def test_pow_p_returns_original_value(make_element, p):
    for value in range(0, min(p, 10)):
        assert make_element(p, value) ** p == make_element(p, value)


@pytest.mark.parametrize("p, value, exp", [
    (5, 2, 3), (7, 5, 4), (11, 3, 9), (13, 7, 11), (101, 2, 50),
])
def test_builtin_pow_uses_dunder_pow(make_element, p, value, exp):
    assert pow(make_element(p, value), exp) == make_element(p, value) ** exp


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 7),
])
def test_builtin_pow_with_mod_argument_raises_type_error(make_element, p, value):
    with pytest.raises(TypeError):
        pow(make_element(p, value), 2, p)


@pytest.mark.parametrize("p, value, e1, e2", [
    (5, 2, 1, 2), (7, 3, 2, 3), (11, 4, 3, 4), (13, 5, 2, 6), (101, 7, 5, 5),
])
def test_pow_adds_exponents(make_element, p, value, e1, e2):
    el = make_element(p, value)
    assert (el ** e1) * (el ** e2) == el ** (e1 + e2)


@pytest.mark.parametrize("p, value, exp", [
    (5, 2, 3), (7, 3, 4), (11, 5, 2), (13, 7, 3), (101, 9, 6),
])
def test_pow_of_product(make_element, p, value, exp):
    a = make_element(p, value)
    b = make_element(p, value + 1)
    assert (a * b) ** exp == (a ** exp) * (b ** exp)


@pytest.mark.parametrize("p, value, exp", [
    (5, 2, 0), (7, 3, 1), (11, 5, 2), (13, 7, 3), (101, 9, 4),
])
def test_pow_does_not_mutate_base(make_element, p, value, exp):
    el = make_element(p, value)
    _ = el ** exp
    assert el == make_element(p, value)


# ===========================================================================
# 14. __invert__ — мультипликативное обращение
# ===========================================================================

@pytest.mark.parametrize("p, value", [
    (5, 1), (5, 4), (7, 3), (11, 6), (13, 9), (101, 50),
])
def test_invert_returns_f_p_element(make_element, p, value):
    result = ~make_element(p, value)
    assert isinstance(result, F_pElement)
    assert result.p == p


@pytest.mark.parametrize("p, value, expected", [
    (5, 1, 1),
    (5, 2, 3),
    (5, 3, 2),
    (5, 4, 4),
    (7, 2, 4),
    (7, 3, 5),
    (7, 6, 6),
    (11, 2, 6),
    (11, 5, 9),
    (11, 10, 10),
    (13, 2, 7),
    (13, 6, 11),
    (13, 12, 12),
    (101, 2, 51),
    (101, 100, 100),
    (2, 1, 1),
])
def test_invert_value(make_element, p, value, expected):
    assert ~make_element(p, value) == make_element(p, expected)


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 5), (11, 3), (13, 8), (101, 33), (2, 1),
])
def test_invert_matches_builtin_pow(make_element, p, value):
    assert (~make_element(p, value)).value == pow(value % p, -1, p)


@pytest.mark.parametrize("p", PRIMES)
def test_invert_zero_raises_zero_division_error(make_element, p):
    with pytest.raises(
            ZeroDivisionError,
            match=f"cannot divide by zero in the field of order {p}"):
        ~make_element(p, 0)


@pytest.mark.parametrize("p", PRIMES)
def test_invert_zero_object_raises_zero_division_error(p):
    with pytest.raises(ZeroDivisionError, match="cannot divide by zero"):
        ~F_pElement.zero(p)


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 7), (13, 5), (101, 17),
])
def test_element_times_its_inverse_is_unit(make_element, p, value):
    el = make_element(p, value)
    assert el * ~el == F_pElement.unit(p)
    assert ~el * el == F_pElement.unit(p)


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 4), (11, 8), (13, 9), (101, 21),
])
def test_double_invert_returns_original_element(make_element, p, value):
    el = make_element(p, value)
    assert ~(~el) == el


@pytest.mark.parametrize("p", PRIMES)
def test_invert_of_unit_is_unit(p):
    assert ~F_pElement.unit(p) == F_pElement.unit(p)


@pytest.mark.parametrize("p, v1, v2", [
    (5, 2, 3), (7, 3, 5), (11, 4, 7), (13, 5, 6), (101, 11, 13),
])
def test_inverse_of_product_is_product_of_inverses(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    assert ~(a * b) == ~a * ~b


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_invert_does_not_mutate_operand(make_element, p, value):
    el = make_element(p, value)
    _ = ~el
    assert el == make_element(p, value)


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_invert_equals_pow_minus_one(make_element, p, value):
    el = make_element(p, value)
    assert ~el == el ** -1


# ===========================================================================
# 15. __truediv__
# ===========================================================================

@pytest.mark.parametrize("p, v1, v2, expected", [
    (5, 1, 2, 3),
    (5, 3, 4, 2),
    (5, 4, 2, 2),
    (5, 2, 3, 4),
    (7, 3, 2, 5),
    (7, 1, 3, 5),
    (7, 2, 4, 4),
    (7, 5, 3, 4),
    (11, 7, 2, 9),
    (11, 5, 9, 3),
    (13, 5, 6, 3),
    (13, 2, 7, 4),
])
def test_truediv_value(make_element, p, v1, v2, expected):
    assert (make_element(p, v1) / make_element(p, v2)).value == expected


@pytest.mark.parametrize("p, value", [
    (5, 1), (5, 4), (7, 3), (11, 6), (13, 9), (101, 42),
])
def test_truediv_by_self_is_unit(make_element, p, value):
    el = make_element(p, value)
    assert el / el == F_pElement.unit(p)


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 5), (11, 3), (13, 11), (101, 17),
])
def test_truediv_zero_numerator_is_zero(make_element, p, value):
    assert F_pElement.zero(p) / make_element(p, value) == F_pElement.zero(p)


@pytest.mark.parametrize("p, v1, v2", [
    (5, 3, 2), (7, 4, 5), (11, 6, 7), (13, 8, 3), (101, 20, 30),
])
def test_truediv_equals_mul_by_inverse(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    assert a / b == a * ~b


@pytest.mark.parametrize("p, v1, v2", [
    (5, 2, 3), (7, 5, 2), (11, 3, 4), (13, 9, 5), (101, 12, 13),
])
def test_division_then_multiplication_restores_value(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    assert (a / b) * b == a
    assert (a * b) / b == a


@pytest.mark.parametrize("p1, p2", [
    (5, 7), (2, 3), (7, 11), (5, 101), (13, 17), (3, 5),
])
def test_truediv_different_fields_raises_type_error(make_element, p1, p2):
    with pytest.raises(TypeError,
                       match="cannot divide two elements in different fields"):
        make_element(p1, 1) / make_element(p2, 1)


@pytest.mark.parametrize("p, value", [
    (5, 1), (5, 4), (7, 6), (11, 3), (13, 12), (101, 1),
])
def test_truediv_by_zero_raises_zero_division_error(make_element, p, value):
    with pytest.raises(
            ZeroDivisionError,
            match=f"cannot divide by zero in the field of order {p}"):
        make_element(p, value) / F_pElement.zero(p)


@pytest.mark.parametrize("p", PRIMES)
def test_truediv_zero_by_zero_raises_zero_division_error(p):
    with pytest.raises(ZeroDivisionError, match="cannot divide by zero"):
        F_pElement.zero(p) / F_pElement.zero(p)


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES)
def test_truediv_non_element_raises_type_error(make_element, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_element(5, 1) / other


@pytest.mark.parametrize("other", [2, 3, 5, 1])
def test_int_divided_by_element_raises_type_error(make_element, other):
    with pytest.raises(TypeError):
        other / make_element(5, 1)


@pytest.mark.parametrize("p, value", [
    (5, 1), (7, 2), (11, 3), (13, 4), (101, 5),
])
def test_truediv_with_subclass_raises_type_error(p, value):
    with pytest.raises(TypeError, match="unsupported operand type"):
        F_pElement(p, value) / SubElement(p, value)


@pytest.mark.parametrize("p, v1, v2", [
    (5, 3, 2), (7, 4, 3), (11, 6, 5), (13, 8, 7), (101, 20, 30),
])
def test_truediv_returns_new_element_and_does_not_mutate(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    result = a / b
    assert isinstance(result, F_pElement)
    assert result is not a and result is not b
    assert result.p == p
    assert a == make_element(p, v1) and b == make_element(p, v2)


# ===========================================================================
# 16. Служебные нюансы и законы поля
# ===========================================================================

@pytest.mark.parametrize("p, value", [
    (5, 0), (7, 0), (2, 0), (11, 0), (13, 0),
])
def test_zero_element_is_truthy(make_element, p, value):
    # __bool__ не определён -> даже ноль поля истинен
    assert bool(make_element(p, value)) is True


@pytest.mark.parametrize("p, value", [
    (5, 0), (7, 1), (11, 5), (13, 12), (101, 100),
])
def test_nonzero_element_is_truthy(make_element, p, value):
    assert bool(make_element(p, value)) is True


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 1), (2, 1), (11, 5), (13, 12),
])
def test_element_cannot_be_converted_to_int(make_element, p, value):
    with pytest.raises(TypeError):
        int(make_element(p, value))


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 2), (11, 9), (13, 4), (101, 8),
])
def test_element_cannot_be_used_as_index(make_element, p, value):
    with pytest.raises(TypeError):
        [1, 2, 3][make_element(p, value)]


@pytest.mark.parametrize("p, value", [
    (5, 3), (7, 2), (11, 9), (13, 4), (101, 8),
])
def test_element_has_no_length(make_element, p, value):
    with pytest.raises(TypeError):
        len(make_element(p, value))


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_element_supports_membership_in_list(make_element, p, value):
    collection = [make_element(p, value), F_pElement.zero(p)]
    assert make_element(p, value) in collection
    assert make_element(p, value + 1) not in collection


@pytest.mark.parametrize("p, value", [
    (5, 2), (7, 3), (11, 4), (13, 5), (101, 6),
])
def test_ordering_operators_are_not_supported(make_element, p, value):
    # __lt__/__le__/__gt__/__ge__ не определены -> сортировка и сравнения невозможны
    first = make_element(p, value)
    second = make_element(p, value + 1)
    with pytest.raises(TypeError):
        sorted([first, second])
    with pytest.raises(TypeError):
        first < second


@pytest.mark.parametrize("p, v1, v2", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_all_operations_do_not_mutate_operands(make_element, p, v1, v2):
    a = make_element(p, v1)
    b = make_element(p, v2)
    for result in (a + b, a - b, a * b, a / b, -a, -b, +a, +b, a ** 3, ~a, ~b):
        assert isinstance(result, F_pElement)
    assert a == make_element(p, v1)
    assert b == make_element(p, v2)
    assert a.p == p and b.p == p


@pytest.mark.parametrize("prime", PRIMES)
def test_operations_are_closed_in_field(prime):
    p = prime
    a = F_pElement(p, 3)
    b = F_pElement(p, 5)
    results = [a + b, a - b, a * b, -a, +a, a ** 7, b ** 2]
    if b.value != 0:
        results.extend([a / b, ~b, b / a if a.value else b])
    for result in results:
        assert isinstance(result, F_pElement)
        assert result.p == p
        assert 0 <= result.value < p


@pytest.mark.parametrize("prime", PRIMES)
def test_additive_group_laws(prime):
    p = prime
    limit = min(p, 12)
    for v1 in range(limit):
        for v2 in range(limit):
            a = F_pElement(p, v1)
            b = F_pElement(p, v2)
            assert (a + b) - b == a
            assert a + (-a) == F_pElement.zero(p)
            assert a + F_pElement.zero(p) == a


@pytest.mark.parametrize("prime", PRIMES)
def test_multiplicative_group_laws(prime):
    p = prime
    limit = min(p, 12)
    for value in range(1, limit):
        a = F_pElement(p, value)
        assert a * ~a == F_pElement.unit(p)
        assert ~a * a == F_pElement.unit(p)
        assert a / a == F_pElement.unit(p)
        assert ~(~a) == a


@pytest.mark.parametrize("prime", SMALL_PRIMES)
def test_field_has_exactly_p_distinct_elements(prime):
    p = prime
    residues = [F_pElement(p, v) for v in range(p)]
    assert len(set(residues)) == p
    for i, first in enumerate(residues):
        for j, second in enumerate(residues):
            assert (first == second) is (i == j)
    # значения, отличающиеся на p, задают один и тот же элемент
    assert F_pElement(p, p) == residues[0]
    assert F_pElement(p, -1) == residues[p - 1]


@pytest.mark.parametrize("prime", PRIMES)
def test_no_zero_divisors(prime):
    p = prime
    limit = min(p, 12)
    for v1 in range(1, limit):
        for v2 in range(1, limit):
            assert F_pElement(p, v1) * F_pElement(p, v2) != F_pElement.zero(p)
