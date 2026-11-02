"""Тесты модуля src/core/secret_share.py — класс Share (доля секрета).

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.
Проверки выполняются через публичный API (свойства x / value / p, ==, !=,
hash(), str(), repr(), арифметические операторы) — без обращения к приватным полям.

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__          — пара (x, value) из одного поля; проверка типов
                          аргументов через isinstance и проверка поля;
    * x, value, p       — свойства только для чтения;
    * __eq__, __ne__    — сравнение по точке и значению (через isinstance);
    * __hash__          — хешируемость, согласованность с __eq__;
    * __str__, __repr__ — "f(x)=value" и "Share(x, value)";
    * __pos__, __neg__  — унарные операции;
    * __add__, __sub__, __mul__, __truediv__ — арифметика долей в одной точке,
                          TypeError для разных x и для чужих типов;
    * __pow__           — возведение значения доли в степень;
    * интеграция        — согласованность с F_pPolynomial.share_at и
                          LagrangeInterpolation;
    * служебные нюансы — отсутствие __bool__/__len__/__iter__/__getitem__/
                          __radd__/abs()/~/////%/@, неизменяемость операндов.

Особенности реализации, зафиксированные тестами:
    * __eq__ сравнивает точку и значение доли (проверка типа через isinstance);
    * арифметика определена только для долей с ОДИНАКОВЫМ x: точка результата
      наследуется от левого операнда, значения складываются/вычитаются/
      умножаются/делятся как элементы поля;
    * __pow__ принимает показатель любого типа, который принимает
      F_pElement.__pow__ (то есть int, включая bool), и не проверяет x;
    * __bool__ отсутствует -> любая доля (даже со значением 0) истинна;
    * x и value хранятся как есть (элементы поля неизменяемы).

Конструктор проверяет аргументы через isinstance: x и value обязаны быть
F_pElement, иначе TypeError с именем переданного типа —
"x must be an instance of F_pElement, not 'int'" или
"value must be an instance of F_pElement, not 'str'". Эта проверка выполняется
раньше, чем сравнение полей ("x and value must be in the same field").
"""

import pytest

from src import F_pElement, F_pPolynomial, Share

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

PRIMES = [2, 3, 5, 7, 11, 13, 101]

SMALL_PRIMES = [2, 3, 5, 7, 11, 13]

FIELD_PAIRS = [(5, 7), (7, 5), (2, 3), (11, 13), (13, 101), (3, 5)]

NON_SHARE_VALUES = [0, 5, "x", None, 3.5, (1, 2), [1], object()]


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
def share5(make_share):
    """Доля f(1) = 2 над F_5."""
    return make_share(5, 1, 2)


@pytest.fixture
def share5_other(make_share):
    """Доля f(1) = 3 над F_5 (та же точка, другое значение)."""
    return make_share(5, 1, 3)


@pytest.fixture
def share5_other_x(make_share):
    """Доля f(2) = 3 над F_5 (другая точка)."""
    return make_share(5, 2, 3)


# ===========================================================================
# 1. __init__ — валидные комбинации
# ===========================================================================

@pytest.mark.parametrize("p, x, value", [
    (5, 0, 0),
    (5, 1, 2),
    (7, 3, 6),
    (11, 10, 0),
    (13, 12, 12),
    (2, 1, 1),
    (101, 100, 99),
])
def test_init_stores_point_and_value(make_share, p, x, value):
    share = make_share(p, x, value)
    assert isinstance(share, Share)
    assert type(share) is Share
    assert share.x == F_pElement(p, x)
    assert share.value == F_pElement(p, value)


@pytest.mark.parametrize("p", PRIMES)
def test_init_with_zero_point_and_zero_value(make_element, p):
    share = Share(F_pElement.zero(p), F_pElement.zero(p))
    assert share.x == F_pElement.zero(p)
    assert share.value == F_pElement.zero(p)
    assert share.p == p


@pytest.mark.parametrize("p, x, value", [
    (5, 7, 12), (7, 9, 16), (11, 13, 25), (13, 15, 28), (101, 105, 210),
])
def test_init_accepts_unreduced_values(make_share, p, x, value):
    # значения приводятся по модулю внутри F_pElement
    share = make_share(p, x, value)
    assert share.x == F_pElement(p, x % p)
    assert share.value == F_pElement(p, value % p)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_init_accepts_element_objects_directly(p, x, value):
    share = Share(F_pElement(p, x), F_pElement(p, value))
    assert share.x.value == x % p
    assert share.value.value == value % p
    assert share.p == p


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_init_does_not_copy_elements(make_element, p, x, value):
    point = make_element(p, x)
    polynomial_value = make_element(p, value)
    share = Share(point, polynomial_value)
    assert share.x is point
    assert share.value is polynomial_value


# ===========================================================================
# 2. __init__ — исключения
# ===========================================================================

@pytest.mark.parametrize("p1, p2", FIELD_PAIRS)
def test_init_different_fields_raises_type_error(make_element, p1, p2):
    with pytest.raises(TypeError, match="x and value must be in the same field"):
        Share(make_element(p1, 1), make_element(p2, 1))


@pytest.mark.parametrize("p1, p2", FIELD_PAIRS)
def test_init_different_fields_message_is_type_error_not_value_error(
        make_element, p1, p2):
    # нюанс: сообщение про поле, но тип исключения — TypeError
    with pytest.raises(TypeError):
        Share(make_element(p1, 0), make_element(p2, 0))


@pytest.mark.parametrize("x, value", [
    (1, 2), ("1", "2"), (None, None), (1.5, 2.5), ([1], [2]),
])
def test_init_non_element_arguments_raise_type_error(x, value):
    with pytest.raises(TypeError, match="x must be an instance of F_pElement"):
        Share(x, value)


@pytest.mark.parametrize("x, value", [
    (1, F_pElement(5, 2)),
    (None, F_pElement(5, 1)),
    ("x", F_pElement(5, 1)),
    (1.5, F_pElement(5, 1)),
    ([1], F_pElement(5, 1)),
])
def test_init_invalid_x_is_reported_before_value(x, value):
    # нюанс: сначала проверяется x, поэтому сообщение всегда про x
    with pytest.raises(TypeError, match="x must be an instance of F_pElement"):
        Share(x, value)


@pytest.mark.parametrize("p, x", [
    (5, 1), (7, 3), (11, 5), (13, 7), (101, 9),
])
def test_init_rejects_polynomials(make_poly, p, x):
    # многочлены имеют свойство p, но не являются F_pElement
    with pytest.raises(
            TypeError,
            match="x must be an instance of F_pElement, not 'F_pPolynomial'"):
        Share(make_poly(p, [x]), make_poly(p, [x + 1]))


@pytest.mark.parametrize("x, value, message", [
    (1, F_pElement(5, 2), "x must be an instance of F_pElement, not 'int'"),
    ("1", F_pElement(5, 2), "x must be an instance of F_pElement, not 'str'"),
    (None, F_pElement(5, 1),
     "x must be an instance of F_pElement, not 'NoneType'"),
    (F_pElement(5, 1), 2, "value must be an instance of F_pElement, not 'int'"),
    (F_pElement(5, 1), None,
     "value must be an instance of F_pElement, not 'NoneType'"),
    (F_pElement(5, 1), "2", "value must be an instance of F_pElement, not 'str'"),
])
def test_init_type_error_messages_mention_class_name(x, value, message):
    with pytest.raises(TypeError) as exc_info:
        Share(x, value)
    assert str(exc_info.value) == message


@pytest.mark.parametrize("p, other_p", FIELD_PAIRS)
def test_init_type_check_precedes_field_check(p, other_p):
    with pytest.raises(TypeError, match="x must be an instance of F_pElement"):
        Share(1, F_pElement(other_p, 1))


@pytest.mark.parametrize("p", PRIMES)
def test_init_requires_both_arguments(p):
    with pytest.raises(TypeError):
        Share(F_pElement(p, 1))
    with pytest.raises(TypeError):
        Share()


# ===========================================================================
# 3. Свойства x, value, p
# ===========================================================================

@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_x_property_returns_point(make_share, p, x, value):
    assert make_share(p, x, value).x == F_pElement(p, x)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_value_property_returns_polynomial_value(make_share, p, x, value):
    assert make_share(p, x, value).value == F_pElement(p, value)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (2, 1, 1), (101, 9, 10),
])
def test_p_property_returns_field_of_point(make_share, p, x, value):
    share = make_share(p, x, value)
    assert share.p == p
    assert share.p == share.x.p
    assert share.p == share.value.p


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_properties_are_read_only(make_share, p, x, value):
    share = make_share(p, x, value)
    for attribute in ("x", "value", "p"):
        with pytest.raises(AttributeError):
            setattr(share, attribute, 1)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_properties_are_stable(make_share, p, x, value):
    share = make_share(p, x, value)
    assert share.x == share.x
    assert share.value == share.value
    assert share.p == share.p


@pytest.mark.parametrize("p", PRIMES)
def test_public_api_of_secret_share(make_share, p):
    share = make_share(p, 1 % p, 2 % p)
    public_names = {name for name in dir(share) if not name.startswith("_")}
    assert public_names == {"x", "value", "p"}


# ===========================================================================
# 4. __eq__ / __ne__
# ===========================================================================

@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 0, 0), (2, 1, 1), (101, 9, 10),
])
def test_eq_is_reflexive(make_share, p, x, value):
    share = make_share(p, x, value)
    assert share == share
    assert not (share != share)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_eq_true_for_equal_point_and_value(make_share, p, x, value):
    assert make_share(p, x, value) == make_share(p, x, value)
    assert not (make_share(p, x, value) != make_share(p, x, value))


@pytest.mark.parametrize("p, x, v1, v2", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 1, 2),
])
def test_eq_false_for_different_values(make_share, p, x, v1, v2):
    assert make_share(p, x, v1) != make_share(p, x, v2)
    assert not (make_share(p, x, v1) == make_share(p, x, v2))


@pytest.mark.parametrize("p, x1, x2, value", [
    (5, 1, 2, 3), (7, 0, 3, 4), (11, 5, 6, 7), (13, 7, 8, 9), (101, 1, 2, 3),
])
def test_eq_false_for_different_points(make_share, p, x1, x2, value):
    assert make_share(p, x1, value) != make_share(p, x2, value)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 7), (7, 3, 11), (11, 5, 16), (13, 7, 20), (101, 9, 110),
])
def test_eq_ignores_unreduced_values(make_share, p, x, value):
    assert make_share(p, x, value) == make_share(p, x, value % p)


@pytest.mark.parametrize("other", NON_SHARE_VALUES + [F_pElement(5, 1)])
def test_eq_with_non_share_is_false(share5, other):
    assert (share5 == other) is False
    assert (share5 != other) is True


@pytest.mark.parametrize("other", NON_SHARE_VALUES[:6])
def test_eq_reflected_with_non_share_is_false(share5, other):
    assert (other == share5) is False
    assert (other != share5) is True


@pytest.mark.parametrize("p, a, b, c", [
    (5, (1, 2), (1, 2), (1, 3)),
    (7, (2, 3), (2, 3), (3, 3)),
    (11, (4, 5), (4, 5), (4, 6)),
    (13, (0, 1), (0, 1), (1, 1)),
    (101, (7, 8), (7, 8), (8, 8)),
])
def test_eq_is_symmetric_and_transitive(make_share, p, a, b, c):
    first = make_share(p, a[0], a[1])
    second = make_share(p, b[0], b[1])
    third = make_share(p, c[0], c[1])
    assert (first == second) == (second == first)
    assert first == second
    assert second != third
    assert first != third


# ===========================================================================
# 5. __hash__
# ===========================================================================

@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_hash_is_consistent_with_eq(make_share, p, x, value):
    first = make_share(p, x, value)
    second = make_share(p, x, value)
    assert first == second
    assert hash(first) == hash(second)
    assert isinstance(hash(first), int)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_hash_equals_hash_of_pair(make_share, p, x, value):
    share = make_share(p, x, value)
    assert hash(share) == hash((share.x, share.value))


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_share_can_be_dict_key(make_share, p, x, value):
    share = make_share(p, x, value)
    mapping = {share: "secret"}
    assert mapping[make_share(p, x, value)] == "secret"
    assert make_share(p, x, value) in mapping


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_distinct_shares_coexist_in_set(make_share, p, x, value):
    first = make_share(p, x, value)
    second = make_share(p, x, value + 1)
    third = make_share(p, x + 1, value)
    assert len({first, second, third, make_share(p, x, value)}) == 3


# ===========================================================================
# 6. __str__ / __repr__
# ===========================================================================

@pytest.mark.parametrize("p, x, value, expected", [
    (5, 1, 2, "f((1 mod 5))=(2 mod 5)"),
    (7, 0, 6, "f((0 mod 7))=(6 mod 7)"),
    (11, 10, 0, "f((10 mod 11))=(0 mod 11)"),
    (13, 3, 12, "f((3 mod 13))=(12 mod 13)"),
    (2, 1, 1, "f((1 mod 2))=(1 mod 2)"),
])
def test_str_format(make_share, p, x, value, expected):
    assert str(make_share(p, x, value)) == expected


@pytest.mark.parametrize("p, x, value, expected", [
    (5, 1, 2, "Share((1 mod 5), (2 mod 5))"),
    (7, 0, 6, "Share((0 mod 7), (6 mod 7))"),
    (11, 10, 0, "Share((10 mod 11), (0 mod 11))"),
    (13, 3, 12, "Share((3 mod 13), (12 mod 13))"),
    (2, 1, 1, "Share((1 mod 2), (1 mod 2))"),
])
def test_repr_format(make_share, p, x, value, expected):
    assert repr(make_share(p, x, value)) == expected


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_str_and_repr_are_consistent_with_properties(make_share, p, x, value):
    share = make_share(p, x, value)
    assert str(share) == f"f({share.x})={share.value}"
    assert repr(share) == f"Share({share.x}, {share.value})"


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_repr_differs_from_str(make_share, p, x, value):
    share = make_share(p, x, value)
    assert repr(share) != str(share)
    assert repr(share).startswith("Share(")


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_str_of_equal_shares_is_equal(make_share, p, x, value):
    assert str(make_share(p, x, value)) == str(make_share(p, x, value + p))
    assert repr(make_share(p, x, value)) == repr(make_share(p, x, value + p))


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_repr_of_list_of_shares(make_share, p, x, value):
    share = make_share(p, x, value)
    assert repr([share]) == f"[{repr(share)}]"
    assert f"{share}" == str(share)


# ===========================================================================
# 7. __pos__ / __neg__
# ===========================================================================

@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_pos_returns_same_object(make_share, p, x, value):
    share = make_share(p, x, value)
    assert (+share) is share


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 0), (11, 5, 6), (13, 7, 8), (2, 1, 1),
])
def test_pos_preserves_share(make_share, p, x, value):
    share = make_share(p, x, value)
    assert +share == share
    assert str(+share) == str(share)


@pytest.mark.parametrize("p, x, value, expected", [
    (5, 1, 2, 3),
    (5, 1, 0, 0),
    (7, 3, 4, 3),
    (11, 5, 10, 1),
    (13, 7, 1, 12),
    (2, 1, 1, 1),
])
def test_neg_negates_value_and_keeps_point(make_share, p, x, value, expected):
    negated = -make_share(p, x, value)
    assert negated.x == F_pElement(p, x)
    assert negated.value == F_pElement(p, expected)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_double_negation_restores_share(make_share, p, x, value):
    share = make_share(p, x, value)
    assert -(-share) == share


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_neg_returns_new_object(make_share, p, x, value):
    share = make_share(p, x, value)
    negated = -share
    assert negated is not share
    assert share == make_share(p, x, value)


@pytest.mark.parametrize("p, x", [
    (5, 1), (7, 3), (11, 5), (13, 7), (101, 9),
])
def test_neg_of_zero_value_share(make_share, p, x):
    share = make_share(p, x, 0)
    assert -share == share
    assert (-share).value == F_pElement.zero(p)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_neg_string_representation(make_share, p, x, value):
    negated = -make_share(p, x, value)
    assert str(negated) == f"f({F_pElement(p, x)})={F_pElement(p, -value)}"


# ===========================================================================
# 8. __add__
# ===========================================================================

@pytest.mark.parametrize("p, x, v1, v2, expected", [
    (5, 1, 2, 3, 0),
    (5, 2, 4, 4, 3),
    (7, 3, 3, 5, 1),
    (11, 4, 4, 9, 2),
    (13, 5, 6, 7, 0),
    (2, 1, 1, 1, 0),
    (101, 50, 30, 40, 70),
])
def test_add_values_at_same_point(make_share, p, x, v1, v2, expected):
    result = make_share(p, x, v1) + make_share(p, x, v2)
    assert result == make_share(p, x, expected)
    assert result.x == F_pElement(p, x)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_add_is_commutative(make_share, p, x, value):
    first = make_share(p, x, value)
    second = make_share(p, x, value + 1)
    assert first + second == second + first


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_add_zero_value_share_is_identity(make_share, p, x, value):
    share = make_share(p, x, value)
    assert share + make_share(p, x, 0) == share
    assert make_share(p, x, 0) + share == share


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_add_of_opposite_shares_is_zero(make_share, p, x, value):
    share = make_share(p, x, value)
    assert share + (-share) == make_share(p, x, 0)


@pytest.mark.parametrize("p, x1, x2, value", [
    (5, 1, 2, 3), (7, 0, 3, 4), (11, 5, 6, 7), (13, 7, 8, 9), (101, 1, 2, 3),
])
def test_add_different_points_raises_type_error(make_share, p, x1, x2, value):
    with pytest.raises(TypeError,
                       match="cannot \\+ with two secret shares with different x"):
        make_share(p, x1, value) + make_share(p, x2, value)


@pytest.mark.parametrize("other", NON_SHARE_VALUES + [F_pElement(5, 1)])
def test_add_non_share_raises_type_error(make_share, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_share(5, 1, 2) + other


@pytest.mark.parametrize("other", [1, 2, 5, 0])
def test_int_plus_share_raises_type_error(make_share, other):
    with pytest.raises(TypeError):
        other + make_share(5, 1, 2)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_add_does_not_mutate_operands(make_share, p, x, value):
    first = make_share(p, x, value)
    second = make_share(p, x, value + 1)
    _ = first + second
    assert first == make_share(p, x, value)
    assert second == make_share(p, x, value + 1)


# ===========================================================================
# 9. __sub__
# ===========================================================================

@pytest.mark.parametrize("p, x, v1, v2, expected", [
    (5, 1, 2, 3, 4),
    (5, 2, 0, 4, 1),
    (7, 3, 3, 5, 5),
    (11, 4, 4, 9, 6),
    (13, 5, 6, 7, 12),
    (2, 1, 1, 1, 0),
    (101, 50, 30, 40, 91),
])
def test_sub_values_at_same_point(make_share, p, x, v1, v2, expected):
    assert make_share(p, x, v1) - make_share(p, x, v2) == make_share(p, x, expected)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_sub_self_is_zero_share(make_share, p, x, value):
    share = make_share(p, x, value)
    assert share - share == make_share(p, x, 0)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_sub_zero_value_share_is_identity(make_share, p, x, value):
    share = make_share(p, x, value)
    assert share - make_share(p, x, 0) == share


@pytest.mark.parametrize("p, x, v1, v2", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 1, 2),
])
def test_sub_is_anti_commutative(make_share, p, x, v1, v2):
    first = make_share(p, x, v1)
    second = make_share(p, x, v2)
    assert first - second == -(second - first)


@pytest.mark.parametrize("p, x1, x2, value", [
    (5, 1, 2, 3), (7, 0, 3, 4), (11, 5, 6, 7), (13, 7, 8, 9), (101, 1, 2, 3),
])
def test_sub_different_points_raises_type_error(make_share, p, x1, x2, value):
    with pytest.raises(TypeError,
                       match="cannot - with two secret shares with different x"):
        make_share(p, x1, value) - make_share(p, x2, value)


@pytest.mark.parametrize("other", NON_SHARE_VALUES + [F_pElement(5, 1)])
def test_sub_non_share_raises_type_error(make_share, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_share(5, 1, 2) - other


# ===========================================================================
# 10. __mul__
# ===========================================================================

@pytest.mark.parametrize("p, x, v1, v2, expected", [
    (5, 1, 2, 3, 1),
    (5, 2, 0, 4, 0),
    (7, 3, 3, 5, 1),
    (11, 4, 4, 9, 3),
    (13, 5, 6, 7, 3),
    (2, 1, 1, 1, 1),
    (101, 50, 30, 40, 89),
])
def test_mul_values_at_same_point(make_share, p, x, v1, v2, expected):
    assert make_share(p, x, v1) * make_share(p, x, v2) == make_share(p, x, expected)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_mul_by_one_value_share_is_identity(make_share, p, x, value):
    share = make_share(p, x, value)
    assert share * make_share(p, x, 1) == share
    assert make_share(p, x, 1) * share == share


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_mul_by_zero_value_share_is_zero(make_share, p, x, value):
    assert make_share(p, x, value) * make_share(p, x, 0) == make_share(p, x, 0)


@pytest.mark.parametrize("p, x, v1, v2", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 11, 12),
])
def test_mul_is_commutative(make_share, p, x, v1, v2):
    assert (make_share(p, x, v1) * make_share(p, x, v2) ==
            make_share(p, x, v2) * make_share(p, x, v1))


@pytest.mark.parametrize("p, x1, x2, value", [
    (5, 1, 2, 3), (7, 0, 3, 4), (11, 5, 6, 7), (13, 7, 8, 9), (101, 1, 2, 3),
])
def test_mul_different_points_raises_type_error(make_share, p, x1, x2, value):
    with pytest.raises(TypeError,
                       match="cannot \\* with two secret shares with different x"):
        make_share(p, x1, value) * make_share(p, x2, value)


@pytest.mark.parametrize("other", NON_SHARE_VALUES + [F_pElement(5, 1)])
def test_mul_non_share_raises_type_error(make_share, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_share(5, 1, 2) * other


@pytest.mark.parametrize("other", [2, 3, 5])
def test_int_times_share_raises_type_error(make_share, other):
    with pytest.raises(TypeError):
        other * make_share(5, 1, 2)


# ===========================================================================
# 11. __truediv__
# ===========================================================================

@pytest.mark.parametrize("p, x, v1, v2, expected", [
    (5, 1, 2, 3, 4),
    (7, 2, 3, 5, 2),
    (11, 3, 4, 9, 9),
    (13, 4, 6, 7, 12),
    (2, 1, 1, 1, 1),
    (101, 50, 30, 40, 26),
])
def test_div_values_at_same_point(make_share, p, x, v1, v2, expected):
    assert make_share(p, x, v1) / make_share(p, x, v2) == make_share(p, x, expected)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_div_by_self_is_one(make_share, p, x, value):
    share = make_share(p, x, value)
    assert share / share == make_share(p, x, 1)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_div_zero_value_share(make_share, p, x, value):
    assert make_share(p, x, 0) / make_share(p, x, value) == make_share(p, x, 0)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_div_by_zero_value_share_raises_zero_division_error(make_share, p, x, value):
    with pytest.raises(ZeroDivisionError, match="cannot divide by zero"):
        make_share(p, x, value) / make_share(p, x, 0)


@pytest.mark.parametrize("p, x1, x2, value", [
    (5, 1, 2, 3), (7, 0, 3, 4), (11, 5, 6, 7), (13, 7, 8, 9), (101, 1, 2, 3),
])
def test_div_different_points_raises_type_error(make_share, p, x1, x2, value):
    with pytest.raises(TypeError,
                       match="cannot / with two secret shares with different x"):
        make_share(p, x1, value) / make_share(p, x2, value)


@pytest.mark.parametrize("other", NON_SHARE_VALUES + [F_pElement(5, 1)])
def test_div_non_share_raises_type_error(make_share, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_share(5, 1, 2) / other


@pytest.mark.parametrize("p, x, v1, v2", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 11, 12),
])
def test_mul_then_div_restores_share(make_share, p, x, v1, v2):
    first = make_share(p, x, v1)
    second = make_share(p, x, v2)
    assert (first * second) / second == first


# ===========================================================================
# 12. __pow__
# ===========================================================================

@pytest.mark.parametrize("p, x, value, exp, expected", [
    (5, 1, 2, 2, 4),
    (5, 1, 2, 3, 3),
    (7, 3, 3, 2, 2),
    (7, 3, 3, 3, 6),
    (11, 5, 4, 2, 5),
    (13, 7, 6, 2, 10),
    (2, 1, 1, 5, 1),
])
def test_pow_raises_value_to_power(make_share, p, x, value, exp, expected):
    result = make_share(p, x, value) ** exp
    assert result == make_share(p, x, expected)
    assert result.x == F_pElement(p, x)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_pow_zero_exponent_gives_one(make_share, p, x, value):
    assert make_share(p, x, value) ** 0 == make_share(p, x, 1)


@pytest.mark.parametrize("p, x, value, expected", [
    (5, 1, 2, 3),
    (7, 3, 3, 5),
    (11, 5, 4, 3),
    (13, 7, 6, 11),
    (101, 9, 10, 91),
])
def test_pow_minus_one_is_inverse(make_share, p, x, value, expected):
    assert make_share(p, x, value) ** -1 == make_share(p, x, expected)


@pytest.mark.parametrize("p, x", [
    (5, 1), (7, 3), (11, 5), (13, 7), (101, 9),
])
def test_pow_zero_value_negative_exponent_raises_value_error(make_share, p, x):
    with pytest.raises(ValueError):
        make_share(p, x, 0) ** -1


@pytest.mark.parametrize("exp", [2.0, "2", None, [2], 3.5])
def test_pow_non_int_exponent_raises_type_error(make_share, exp):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_share(5, 1, 2) ** exp


@pytest.mark.parametrize("exp, expected", [(True, 2), (False, 1)])
def test_pow_bool_exponent_is_accepted_as_int(make_share, exp, expected):
    # нюанс: F_pElement.__pow__ проверяет isinstance(exp, int), bool — подкласс int
    assert make_share(5, 1, 2) ** exp == make_share(5, 1, expected)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_pow_does_not_mutate_share(make_share, p, x, value):
    share = make_share(p, x, value)
    _ = share ** 3
    assert share == make_share(p, x, value)


@pytest.mark.parametrize("p, x, value, e1, e2", [
    (5, 1, 2, 1, 2), (7, 3, 4, 2, 3), (11, 5, 6, 1, 4), (13, 7, 8, 2, 2),
    (101, 9, 10, 3, 1),
])
def test_pow_adds_exponents(make_share, p, x, value, e1, e2):
    share = make_share(p, x, value)
    assert (share ** e1) * (share ** e2) == share ** (e1 + e2)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_pow_returns_secret_share(make_share, p, x, value):
    result = make_share(p, x, value) ** 4
    assert isinstance(result, Share)
    assert result.p == p


# ===========================================================================
# 13. Интеграция с многочленами и интерполяцией
# ===========================================================================

@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2, 3], 2), (7, [6, 5], 4), (11, [10, 1], 9), (13, [1, 0, 2], 5),
    (101, [7, 8], 100),
])
def test_share_from_polynomial_matches_manual_share(make_poly, p, coeffs, x):
    poly = make_poly(p, coeffs)
    point = F_pElement(p, x)
    assert poly.share_at(point) == Share(point, poly(point))


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [1, 2], [0, 1]),
    (7, [3, 4, 5], [1, 2, 3]),
    (11, [1, 2, 3], [0, 1, 2]),
    (13, [9, 8, 7], [2, 3, 4]),
    (101, [7, 6, 5], [10, 20, 30]),
])
def test_shares_of_polynomial_are_accepted_by_interpolation(make_poly, p, coeffs, xs):
    from src.core.algebra.interpolation import LagrangeInterpolation
    poly = make_poly(p, coeffs)
    shares = [poly.share_at(F_pElement(p, x)) for x in xs]
    interpolation = LagrangeInterpolation(shares)
    assert interpolation.lagrange_polynomial == poly
    assert interpolation.shares == shares


@pytest.mark.parametrize("p, coeffs, xs", [
    (5, [3, 1, 2], [1, 2, 3]),
    (7, [6, 5, 4], [1, 2, 3]),
    (11, [10, 9, 8], [1, 2, 3]),
    (13, [1, 2, 3], [4, 5, 6]),
    (101, [42, 7, 8], [1, 2, 3]),
])
def test_interpolation_recovers_secret_from_shares(make_poly, p, coeffs, xs):
    from src.core.algebra.interpolation import LagrangeInterpolation
    poly = make_poly(p, coeffs)
    shares = [poly.share_at(F_pElement(p, x)) for x in xs]
    secret = LagrangeInterpolation(shares)(F_pElement.zero(p))
    assert secret.value == F_pElement(p, coeffs[0])
    assert secret.x == F_pElement.zero(p)


@pytest.mark.parametrize("p, coeffs, x", [
    (5, [1, 2], 3), (7, [3, 4], 5), (11, [9, 8], 2), (13, [5, 6], 7),
    (101, [1, 2], 50),
])
def test_share_str_and_repr_from_polynomial(make_poly, p, coeffs, x):
    poly = make_poly(p, coeffs)
    point = F_pElement(p, x)
    share = poly.share_at(point)
    assert str(share) == f"f({point})={poly(point)}"
    assert repr(share) == f"Share({point}, {poly(point)})"


@pytest.mark.parametrize("p, a, b, x", [
    (5, [1, 2], [3, 4], 2),
    (7, [6, 1], [2, 3], 5),
    (11, [1, 2, 3], [9, 8], 4),
    (13, [0, 1], [1, 0, 2], 7),
    (2, [1, 1], [1], 1),
])
def test_sum_of_shares_equals_share_of_sum(make_poly, p, a, b, x):
    point = F_pElement(p, x)
    left = make_poly(p, a).share_at(point) + make_poly(p, b).share_at(point)
    right = (make_poly(p, a) + make_poly(p, b)).share_at(point)
    assert left == right


# ===========================================================================
# 14. Служебные нюансы
# ===========================================================================

@pytest.mark.parametrize("p, x, value", [
    (5, 1, 0), (7, 3, 0), (11, 5, 0), (13, 7, 0), (2, 1, 0),
])
def test_share_with_zero_value_is_truthy(make_share, p, x, value):
    # __bool__ не определён -> любая доля истинна
    assert bool(make_share(p, x, value)) is True


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_share_is_truthy(make_share, p, x, value):
    assert bool(make_share(p, x, value)) is True
    assert make_share(p, x, value)


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_len_iteration_and_indexing_are_not_supported(make_share, p, x, value):
    share = make_share(p, x, value)
    with pytest.raises(TypeError):
        len(share)
    with pytest.raises(TypeError):
        iter(share)
    with pytest.raises(TypeError):
        share[0]


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_abs_invert_floordiv_mod_matmul_are_not_supported(
        make_share, p, x, value):
    share = make_share(p, x, value)
    with pytest.raises(TypeError):
        abs(share)
    with pytest.raises(TypeError):
        ~share
    with pytest.raises(TypeError):
        share // share
    with pytest.raises(TypeError):
        share % share
    with pytest.raises(TypeError):
        share @ share


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_share_cannot_be_converted_to_int(make_share, p, x, value):
    with pytest.raises(TypeError):
        int(make_share(p, x, value))


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_share_is_not_ordered(make_share, p, x, value):
    first = make_share(p, x, value)
    second = make_share(p, x, value + 1)
    with pytest.raises(TypeError):
        first < second
    with pytest.raises(TypeError):
        sorted([first, second])


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (101, 9, 10),
])
def test_unary_operations_do_not_mutate_share(make_share, p, x, value):
    share = make_share(p, x, value)
    _ = +share
    _ = -share
    _ = share ** 2
    assert share == make_share(p, x, value)
    assert share.x == F_pElement(p, x)
    assert share.value == F_pElement(p, value)


@pytest.mark.parametrize("prime", PRIMES)
def test_shares_of_zero_polynomial(prime):
    p = prime
    poly = F_pPolynomial(p)
    for value in range(min(p, 5)):
        share = poly.share_at(F_pElement(p, value))
        assert share.value == F_pElement.zero(p)
        assert share.x == F_pElement(p, value)
        assert share.p == p
        assert str(share) == (f"f(({value} mod {p}))=(0 mod {p})")


@pytest.mark.parametrize("prime", PRIMES)
def test_negation_of_share_from_polynomial(prime):
    p = prime
    poly = F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    point = F_pElement.one(p)
    share = poly.share_at(point)
    negated = -share
    assert negated.x == share.x
    assert negated.value == -share.value
    assert negated == Share(point, -poly(point))
