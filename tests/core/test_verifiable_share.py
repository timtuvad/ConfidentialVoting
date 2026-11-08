"""Тесты модуля src/core/verifiable_share.py — класс VerifiableShare.

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.
Проверки выполняются через публичный API (свойства x / value / p /
blinding_share, make(), ==, !=, hash(), str(), repr(), операторы + и -,
унарный минус) — без обращения к приватным полям.

VerifiableShare(x, value, blinding_share) — верифицируемая доля секрета
(Pedersen VSS): наследник Share, хранящий дополнительно «долю случайности»
blinding_share. Класс переопределяет make (три аргумента), __eq__, __str__,
__repr__, __pos__, __neg__, __add__, __sub__ и добавляет _new.

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__          — тройка (x, value, blinding_share) из одного поля;
                          порядок проверок: x/value (через Share), затем тип
                          blinding_share, затем поле blinding_share;
    * x, value, p, blinding_share — свойства только для чтения;
    * make              — VerifiableShare.make(f, x, blinding_polynomial) =
                          VerifiableShare(x, f(x), blinding_polynomial(x));
    * __eq__, __ne__    — с VerifiableShare: по всем трём компонентам;
                          с обычным Share: только по x и value (blinding_share
                          игнорируется), симметрично;
    * __hash__          — хешируемость, согласованность с __eq__
                          (хеш как у Share: по x и value);
    * __str__, __repr__ — "f(x)=value, blinding_share=bs" и
                          "VerifiableShare(x, value, blinding_share=bs)";
    * __pos__, __neg__  — унарные операции (минус обращает value И blinding_share);
    * __add__, __sub__  — VS+VS -> VerifiableShare; VS+Share и Share+VS -> Share;
                          ValueError для разных x, TypeError для чужих типов;
    * наследование      — isinstance(vs, Share), подстановка в LagrangeInterpolation;
    * служебные нюансы — неизменяемость операндов.

Особенности реализации, зафиксированные тестами:
    * __add__/__sub__ с обычным Share делегируются в Share.__add__/__sub__,
      которые жёстко создают Share._new — результат ТИПА Share (не
      VerifiableShare), blinding_share теряется; это соответствует docstring
      класса («Если же хотя бы один из них был просто Share, то в результате
      получается Share»);
    * сообщение TypeError для чужих типов содержит имя 'VerifiableShare'
      (Share.__check_types использует self.__class__.__name__);
    * сообщение ValueError для разных x печатает str() обеих долей ВМЕСТЕ с
      blinding_share («cannot + with two shares with different x values:
      f((1 mod 5))=(2 mod 5), blinding_share=(3 mod 5) and ...»);
    * равенство НЕ транзитивно: VS(x, v, 1) == Share(x, v) и
      VS(x, v, 2) == Share(x, v), но VS(x, v, 1) != VS(x, v, 2);
    * make НЕ валидирует аргументы (использует _new): ошибки возникают только
      из F_pPolynomial.__call__ (TypeError для не-F_pElement и для элементов
      чужого поля);
    * LagrangeInterpolation принимает VerifiableShare как Share: ключом dict
      служит share.x (F_pElement).

__hash__ определён как super().__hash__() — то есть Share.__hash__ =
hash((x, value)): blinding_share в хеше не участвует. Это согласовано с
__eq__: VerifiableShare может быть равен обычному Share (по x и value),
а равные объекты обязаны иметь равные хеши.
"""

import pytest

from src.core.algebra.field import F_pElement
from src.core.algebra.interpolation import LagrangeInterpolation
from src.core.algebra.polynomial import F_pPolynomial
from src.core.share import Share
from src.core.verifiable_share import VerifiableShare

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


@pytest.fixture
def make_element():
    """Фабрика элементов поля: make_element(p, value) -> F_pElement."""
    def _make(p, value=None):
        return F_pElement(p, value)
    return _make


@pytest.fixture
def make_share():
    """Фабрика обычных долей: make_share(p, x, value) -> Share."""
    def _make(p, x, value):
        return Share(F_pElement(p, x), F_pElement(p, value))
    return _make


@pytest.fixture
def make_verifiable_share():
    """Фабрика верифицируемых долей: (p, x, value, blinding_share)."""
    def _make(p, x, value, blinding_share):
        return VerifiableShare(F_pElement(p, x), F_pElement(p, value),
                               F_pElement(p, blinding_share))
    return _make


@pytest.fixture
def make_poly():
    """Фабрика многочленов из списка целых коэффициентов (от свободного члена)."""
    def _make(p, coefficients):
        return F_pPolynomial(
            p, coefficients=[F_pElement(p, c) for c in coefficients])
    return _make


@pytest.fixture
def vs5(make_verifiable_share):
    """Верифицируемая доля f(1) = 2, blinding_share = 3 над F_5."""
    return make_verifiable_share(5, 1, 2, 3)


@pytest.fixture
def vs5_other_blinding(make_verifiable_share):
    """Та же точка и значение, другая доля случайности: f(1) = 2, bs = 4."""
    return make_verifiable_share(5, 1, 2, 4)


# ===========================================================================
# 1. __init__ — валидные комбинации
# ===========================================================================

@pytest.mark.parametrize("p, x, value, bs", [
    (5, 0, 0, 0),
    (5, 1, 2, 3),
    (7, 3, 6, 1),
    (11, 10, 0, 5),
    (13, 12, 12, 1),
    (2, 1, 1, 1),
    (101, 100, 99, 98),
])
def test_init_stores_all_three_components(make_verifiable_share,
                                          p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert isinstance(vs, VerifiableShare)
    assert isinstance(vs, Share)
    assert type(vs) is VerifiableShare
    assert vs.x == F_pElement(p, x)
    assert vs.value == F_pElement(p, value)
    assert vs.blinding_share == F_pElement(p, bs)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_init_accepts_element_objects_directly(p, x, value, bs):
    vs = VerifiableShare(F_pElement(p, x), F_pElement(p, value),
                         F_pElement(p, bs))
    assert vs.x.value == x % p
    assert vs.value.value == value % p
    assert vs.blinding_share.value == bs % p
    assert vs.p == p


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 7, 12, 8), (7, 9, 16, 20), (11, 13, 25, 30), (13, 15, 28, 41),
])
def test_init_accepts_unreduced_values(make_verifiable_share, p, x, value, bs):
    # значения приводятся по модулю внутри F_pElement
    vs = make_verifiable_share(p, x, value, bs)
    assert vs.x == F_pElement(p, x % p)
    assert vs.value == F_pElement(p, value % p)
    assert vs.blinding_share == F_pElement(p, bs % p)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_init_does_not_copy_elements(make_element, p, x, value, bs):
    point = make_element(p, x)
    polynomial_value = make_element(p, value)
    blinding = make_element(p, bs)
    vs = VerifiableShare(point, polynomial_value, blinding)
    assert vs.x is point
    assert vs.value is polynomial_value
    assert vs.blinding_share is blinding


@pytest.mark.parametrize("p, x, value", [
    (5, 1, 2), (7, 3, 4), (11, 5, 6), (13, 7, 8), (2, 1, 0),
])
def test_init_with_zero_blinding_share(make_verifiable_share, p, x, value):
    vs = make_verifiable_share(p, x, value, 0)
    assert vs.blinding_share == F_pElement.zero(p)
    assert not vs.blinding_share  # __bool__ элемента поля: 0 ложен


@pytest.mark.parametrize("prime", PRIMES)
def test_init_for_every_prime(prime):
    p = prime
    vs = VerifiableShare(F_pElement(p, 1), F_pElement(p, 2 % p),
                         F_pElement(p, 3 % p))
    assert vs.p == p


# ===========================================================================
# 2. __init__ — исключения (порядок проверок: x, value — через Share,
#    затем тип blinding_share, затем поле blinding_share)
# ===========================================================================

@pytest.mark.parametrize("x, value, bs", [
    (1, 2, 3), ("1", "2", "3"), (None, None, None), (1.5, 2.5, 3.5),
    ([1], [2], [3]),
])
def test_init_non_element_arguments_raise_type_error(x, value, bs):
    with pytest.raises(TypeError, match="x must be an instance of F_pElement"):
        VerifiableShare(x, value, bs)


def test_init_x_check_precedes_value_check():
    with pytest.raises(TypeError, match="x must be an instance of F_pElement"):
        VerifiableShare(1, F_pElement(5, 2), F_pElement(5, 3))


def test_init_value_check_precedes_blinding_check():
    # x корректен, value — нет: сообщение про value (проверка Share),
    # а не про blinding_share
    with pytest.raises(TypeError,
                       match="value must be an instance of F_pElement"):
        VerifiableShare(F_pElement(5, 1), 2, F_pElement(5, 3))


@pytest.mark.parametrize("bs", [0, 5, "3", None, 3.5, [3], (3,), object()])
def test_init_non_element_blinding_share_raises_type_error(bs):
    with pytest.raises(
            TypeError,
            match="blinding_share must be an instance of F_pElement"):
        VerifiableShare(F_pElement(5, 1), F_pElement(5, 2), bs)


@pytest.mark.parametrize("bs, name", [
    (3, "int"), ("3", "str"), (None, "NoneType"), (3.5, "float"),
    ([3], "list"),
])
def test_init_blinding_type_error_mentions_class_name(bs, name):
    with pytest.raises(TypeError) as exc_info:
        VerifiableShare(F_pElement(5, 1), F_pElement(5, 2), bs)
    assert str(exc_info.value) == ("blinding_share must be an instance of "
                                   f"F_pElement, not '{name}'")


@pytest.mark.parametrize("p1, p2", FIELD_PAIRS)
def test_init_x_and_value_different_fields_checked_first(p1, p2):
    # super().__init__ выполняется раньше проверки blinding_share:
    # сообщение про x и value (как у Share)
    with pytest.raises(TypeError,
                       match="x and value must be in the same field"):
        VerifiableShare(F_pElement(p1, 1), F_pElement(p2, 1),
                        F_pElement(p1, 1))


@pytest.mark.parametrize("p1, p2", FIELD_PAIRS)
def test_init_blinding_from_other_field_raises_type_error(p1, p2):
    with pytest.raises(
            TypeError,
            match="x, value and blinding_share must be in the same field"):
        VerifiableShare(F_pElement(p1, 1), F_pElement(p1, 1),
                        F_pElement(p2, 1))


def test_init_blinding_type_check_precedes_field_check():
    # blinding_share вообще не элемент — сообщение про тип, не про поле
    with pytest.raises(TypeError,
                       match="blinding_share must be an instance"):
        VerifiableShare(F_pElement(5, 1), F_pElement(5, 2), 3)


def test_init_requires_three_arguments():
    with pytest.raises(TypeError):
        VerifiableShare(F_pElement(5, 1), F_pElement(5, 2))
    with pytest.raises(TypeError):
        VerifiableShare(F_pElement(5, 1))
    with pytest.raises(TypeError):
        VerifiableShare()


@pytest.mark.parametrize("p, x", [
    (5, 1), (7, 3), (11, 5), (13, 7), (101, 9),
])
def test_init_rejects_polynomials(make_poly, p, x):
    # многочлены имеют свойство p, но не являются F_pElement
    with pytest.raises(
            TypeError,
            match="x must be an instance of F_pElement, not 'F_pPolynomial'"):
        VerifiableShare(make_poly(p, [x]), make_poly(p, [x + 1]),
                        make_poly(p, [x + 2]))


def test_init_rejects_share_as_blinding(make_share):
    # Share — не F_pElement
    with pytest.raises(
            TypeError,
            match="blinding_share must be an instance of F_pElement, "
                  "not 'Share'"):
        VerifiableShare(F_pElement(5, 1), F_pElement(5, 2),
                        make_share(5, 1, 3))


# ===========================================================================
# 3. Свойства x, value, p, blinding_share
# ===========================================================================

@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_blinding_share_property(make_verifiable_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert vs.blinding_share == F_pElement(p, bs)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_p_property_is_field_of_all_components(make_verifiable_share,
                                               p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert vs.p == p
    assert vs.p == vs.x.p
    assert vs.p == vs.value.p
    assert vs.p == vs.blinding_share.p


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_inherited_x_and_value_properties(make_verifiable_share,
                                          p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert vs.x == F_pElement(p, x)
    assert vs.value == F_pElement(p, value)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_properties_are_read_only(make_verifiable_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    for attribute in ("x", "value", "p", "blinding_share"):
        with pytest.raises(AttributeError):
            setattr(vs, attribute, 1)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_properties_are_stable(make_verifiable_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert vs.x == vs.x
    assert vs.value == vs.value
    assert vs.p == vs.p
    assert vs.blinding_share == vs.blinding_share


# ===========================================================================
# 4. make(f, x, blinding_polynomial)
# ===========================================================================

@pytest.mark.parametrize("p, f_coeffs, b_coeffs, x", [
    (5, [1, 2], [3, 4], 0),
    (5, [1, 2], [3], 4),
    (7, [6, 5], [1, 2], 3),
    (11, [10], [9, 8], 7),
    (13, [0, 1], [12], 12),
    (2, [1, 1], [1], 1),
])
def test_make_returns_verifiable_share_with_components(make_poly, p,
                                                       f_coeffs, b_coeffs, x):
    f = make_poly(p, f_coeffs)
    b = make_poly(p, b_coeffs)
    point = F_pElement(p, x)
    vs = VerifiableShare.make(f, point, b)
    assert isinstance(vs, VerifiableShare)
    assert vs.x == point
    assert vs.value == f(point)
    assert vs.blinding_share == b(point)


@pytest.mark.parametrize("p, f_coeffs, b_coeffs, x", [
    (5, [1, 2], [3, 4], 3),
    (7, [3, 4], [1, 1], 5),
    (11, [9, 8], [2, 3], 2),
    (13, [5, 6], [7], 7),
    (101, [1, 2], [3, 4], 50),
])
def test_make_equals_manual_construction(make_poly, make_verifiable_share,
                                         p, f_coeffs, b_coeffs, x):
    f = make_poly(p, f_coeffs)
    b = make_poly(p, b_coeffs)
    point = F_pElement(p, x)
    assert VerifiableShare.make(f, point, b) == VerifiableShare(
        point, f(point), b(point))


@pytest.mark.parametrize("p, f_coeffs, b_coeffs, x, expected_v, expected_b", [
    (5, [1, 2], [3, 4], 3, 2, 0),      # f(3)=1+6=7=2; b(3)=3+12=15=0
    (7, [6, 5], [1], 2, 2, 1),         # f(2)=6+10=16=2; b(2)=1
    (11, [1, 2, 3], [5], 3, 1, 5),     # f(3)=1+6+27=34=1
    (13, [12, 1], [0, 2], 12, 11, 11),  # f(12)=24=11; b(12)=24=11
    (5, [3], [4], 0, 3, 4),
    (2, [1, 1], [1], 1, 0, 1),         # f(1)=2=0
])
def test_make_values_match_expected(make_poly, p, f_coeffs, b_coeffs, x,
                                    expected_v, expected_b):
    vs = VerifiableShare.make(make_poly(p, f_coeffs), F_pElement(p, x),
                              make_poly(p, b_coeffs))
    assert vs.value == F_pElement(p, expected_v)
    assert vs.blinding_share == F_pElement(p, expected_b)


@pytest.mark.parametrize("prime", PRIMES)
def test_make_at_zero_returns_constant_terms(prime):
    p = prime
    f = F_pPolynomial(p, coefficients=[F_pElement(p, 1), F_pElement(p, 2)])
    b = F_pPolynomial(p, coefficients=[F_pElement(p, 3), F_pElement(p, 4)])
    vs = VerifiableShare.make(f, F_pElement.zero(p), b)
    assert vs.x == F_pElement.zero(p)
    assert vs.value == F_pElement(p, 1)
    assert vs.blinding_share == F_pElement(p, 3)


@pytest.mark.parametrize("prime", PRIMES)
def test_make_of_zero_polynomials(prime):
    p = prime
    vs = VerifiableShare.make(F_pPolynomial(p), F_pElement(p, 1),
                              F_pPolynomial(p))
    assert vs.value == F_pElement.zero(p)
    assert vs.blinding_share == F_pElement.zero(p)


@pytest.mark.parametrize("other", NON_SHARE_VALUES)
def test_make_with_non_element_x_raises_type_error(make_poly, other):
    with pytest.raises(TypeError,
                       match="polynomial argument must be a F_pElement"):
        VerifiableShare.make(make_poly(5, [1, 2]), other, make_poly(5, [3]))


@pytest.mark.parametrize("other", [5, None, "poly", 3.5, [1], object()])
def test_make_with_non_polynomial_f_raises_type_error(other):
    with pytest.raises(TypeError):
        VerifiableShare.make(other, F_pElement(5, 1), F_pPolynomial(5))


def test_make_validates_f_before_blinding(make_poly):
    # аргументы _new вычисляются слева направо: ошибка f(x) раньше b(x)
    with pytest.raises(TypeError, match="'int' object is not callable"):
        VerifiableShare.make(5, F_pElement(5, 1), 7)


@pytest.mark.parametrize("p, other_p", FIELD_PAIRS)
def test_make_with_polynomials_of_different_fields_raises_type_error(
        make_poly, p, other_p):
    f = make_poly(p, [1, 2])
    b = make_poly(other_p, [1, 2])
    with pytest.raises(TypeError,
                       match="polynomial argument must be in the same field "
                             "as the polynomial"):
        VerifiableShare.make(f, F_pElement(p, 1), b)


@pytest.mark.parametrize("p, f_coeffs, b_coeffs", [
    (5, [1, 2], [3, 4]), (7, [3], [1, 1]), (11, [9, 8], [2]), (2, [1, 1], [1]),
])
def test_make_does_not_mutate_polynomials(make_poly, p, f_coeffs, b_coeffs):
    f = make_poly(p, f_coeffs)
    b = make_poly(p, b_coeffs)
    for value in range(min(p, 4)):
        VerifiableShare.make(f, F_pElement(p, value), b)
    assert f == make_poly(p, f_coeffs)
    assert b == make_poly(p, b_coeffs)


@pytest.mark.parametrize("p, f_coeffs, b_coeffs, x", [
    (5, [1, 2], [3, 4], 1), (7, [3], [1], 2), (11, [9, 8], [2, 3], 3),
    (13, [5], [6], 4), (101, [1, 2], [3], 5),
])
def test_make_returns_new_object_each_call(make_poly, p, f_coeffs, b_coeffs, x):
    f = make_poly(p, f_coeffs)
    b = make_poly(p, b_coeffs)
    point = F_pElement(p, x)
    first = VerifiableShare.make(f, point, b)
    second = VerifiableShare.make(f, point, b)
    assert first is not second
    assert first == second


@pytest.mark.parametrize("p, f_coeffs, b_coeffs, x", [
    (5, [1, 2], [3, 4], 3), (7, [3, 4], [1], 5), (11, [9, 1], [2, 3], 2),
])
def test_make_string_representations(make_poly, p, f_coeffs, b_coeffs, x):
    f = make_poly(p, f_coeffs)
    b = make_poly(p, b_coeffs)
    point = F_pElement(p, x)
    vs = VerifiableShare.make(f, point, b)
    assert str(vs) == f"f({point})={f(point)}, blinding_share={b(point)}"
    assert repr(vs) == (f"VerifiableShare({point}, {f(point)}, "
                        f"blinding_share={b(point)})")


# ===========================================================================
# 5. __eq__ / __ne__
# ===========================================================================

@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (2, 1, 1, 0),
    (101, 9, 10, 11),
])
def test_eq_is_reflexive(make_verifiable_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert vs == vs
    assert not (vs != vs)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_eq_true_for_equal_triples(make_verifiable_share, p, x, value, bs):
    first = make_verifiable_share(p, x, value, bs)
    second = make_verifiable_share(p, x, value, bs)
    assert first == second
    assert not (first != second)


@pytest.mark.parametrize("p, x, value, bs1, bs2", [
    (5, 1, 2, 3, 4), (7, 3, 4, 0, 1), (11, 5, 6, 7, 8), (13, 7, 8, 9, 10),
    (101, 9, 1, 2, 3),
])
def test_eq_false_for_different_blinding_shares(make_verifiable_share,
                                                p, x, value, bs1, bs2):
    first = make_verifiable_share(p, x, value, bs1)
    second = make_verifiable_share(p, x, value, bs2)
    assert first != second
    assert not (first == second)


@pytest.mark.parametrize("p, x, v1, v2, bs", [
    (5, 1, 2, 3, 4), (7, 3, 4, 5, 1), (11, 5, 6, 7, 8), (13, 7, 8, 9, 0),
    (101, 9, 1, 2, 3),
])
def test_eq_false_for_different_values(make_verifiable_share,
                                       p, x, v1, v2, bs):
    assert (make_verifiable_share(p, x, v1, bs)
            != make_verifiable_share(p, x, v2, bs))


@pytest.mark.parametrize("p, x1, x2, value, bs", [
    (5, 1, 2, 3, 4), (7, 0, 3, 4, 1), (11, 5, 6, 7, 8), (13, 7, 8, 9, 0),
    (101, 1, 2, 3, 4),
])
def test_eq_false_for_different_points(make_verifiable_share,
                                       p, x1, x2, value, bs):
    assert (make_verifiable_share(p, x1, value, bs)
            != make_verifiable_share(p, x2, value, bs))


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 7, 9), (7, 3, 11, 15), (11, 5, 16, 20), (13, 7, 20, 27),
    (101, 9, 110, 200),
])
def test_eq_ignores_unreduced_values(make_verifiable_share, p, x, value, bs):
    assert (make_verifiable_share(p, x, value, bs)
            == make_verifiable_share(p, x, value % p, bs % p))


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_eq_with_plain_share_compares_only_x_and_value(
        make_verifiable_share, make_share, p, x, value, bs):
    # важный нюанс: с обычным Share сравнение идёт через Share.__eq__ —
    # blinding_share игнорируется, равенство симметрично
    vs = make_verifiable_share(p, x, value, bs)
    share = make_share(p, x, value)
    assert vs == share
    assert share == vs
    assert not (vs != share)
    assert not (share != vs)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_eq_with_plain_share_different_value_is_false(
        make_verifiable_share, make_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    share = make_share(p, x, value + 1)
    assert vs != share
    assert share != vs


def test_eq_not_transitive_across_classes(make_verifiable_share, make_share):
    # нюанс: равенство НЕ транзитивно — обе VS равны обычному Share
    # (blinding_share игнорируется), но не равны друг другу
    first = make_verifiable_share(5, 1, 2, 1)
    second = make_verifiable_share(5, 1, 2, 2)
    share = make_share(5, 1, 2)
    assert first == share
    assert second == share
    assert first != second


@pytest.mark.parametrize("other", NON_SHARE_VALUES + [F_pElement(5, 1)])
def test_eq_with_non_share_is_false(vs5, other):
    assert (vs5 == other) is False
    assert (vs5 != other) is True


@pytest.mark.parametrize("other", NON_SHARE_VALUES[:6])
def test_eq_reflected_with_non_share_is_false(vs5, other):
    assert (other == vs5) is False
    assert (other != vs5) is True


@pytest.mark.parametrize("p, a, b, c", [
    (5, (1, 2, 3), (1, 2, 3), (1, 2, 4)),
    (7, (2, 3, 4), (2, 3, 4), (3, 3, 4)),
    (11, (4, 5, 6), (4, 5, 6), (4, 5, 7)),
    (13, (0, 1, 2), (0, 1, 2), (0, 1, 3)),
    (101, (7, 8, 9), (7, 8, 9), (7, 8, 10)),
])
def test_eq_is_symmetric_and_transitive_within_class(make_verifiable_share,
                                                     p, a, b, c):
    first = make_verifiable_share(p, a[0], a[1], a[2])
    second = make_verifiable_share(p, b[0], b[1], b[2])
    third = make_verifiable_share(p, c[0], c[1], c[2])
    assert (first == second) == (second == first)
    assert first == second
    assert second != third
    assert first != third


# ===========================================================================
# 6. __hash__ (как у Share: hash((x, value)), без blinding_share)
# ===========================================================================

@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_hash_is_consistent_with_eq(make_verifiable_share, p, x, value, bs):
    first = make_verifiable_share(p, x, value, bs)
    second = make_verifiable_share(p, x, value, bs)
    assert first == second
    assert hash(first) == hash(second)
    assert isinstance(hash(first), int)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_hash_equals_hash_of_plain_share(make_verifiable_share, make_share,
                                         p, x, value, bs):
    # контракт: равные объекты обязаны иметь равные хеши;
    # vs == Share(x, value), поэтому hash(vs) == hash(share)
    vs = make_verifiable_share(p, x, value, bs)
    share = make_share(p, x, value)
    assert vs == share
    assert hash(vs) == hash(share)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_hash_equals_hash_of_x_value_pair(make_verifiable_share,
                                          p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert hash(vs) == hash((vs.x, vs.value))


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_verifiable_share_can_be_dict_key(make_verifiable_share,
                                          p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    mapping = {vs: "secret"}
    assert mapping[make_verifiable_share(p, x, value, bs)] == "secret"
    assert make_verifiable_share(p, x, value, bs) in mapping


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_distinct_verifiable_shares_coexist_in_set(make_verifiable_share,
                                                   p, x, value, bs):
    first = make_verifiable_share(p, x, value, bs)
    second = make_verifiable_share(p, x, value, bs + 1)
    third = make_verifiable_share(p, x, value + 1, bs)
    assert len({first, second, third,
                make_verifiable_share(p, x, value, bs)}) == 3


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_equal_verifiable_shares_collapse_in_set(make_verifiable_share,
                                                 p, x, value, bs):
    first = make_verifiable_share(p, x, value, bs)
    second = make_verifiable_share(p, x, value, bs)
    assert len({first, second}) == 1


# ===========================================================================
# 7. __str__ / __repr__
# ===========================================================================

@pytest.mark.parametrize("p, x, value, bs, expected", [
    (5, 1, 2, 3, "f((1 mod 5))=(2 mod 5), blinding_share=(3 mod 5)"),
    (7, 0, 6, 1, "f((0 mod 7))=(6 mod 7), blinding_share=(1 mod 7)"),
    (11, 10, 0, 5, "f((10 mod 11))=(0 mod 11), blinding_share=(5 mod 11)"),
    (13, 3, 12, 0, "f((3 mod 13))=(12 mod 13), blinding_share=(0 mod 13)"),
    (2, 1, 1, 1, "f((1 mod 2))=(1 mod 2), blinding_share=(1 mod 2)"),
])
def test_str_format(make_verifiable_share, p, x, value, bs, expected):
    assert str(make_verifiable_share(p, x, value, bs)) == expected


@pytest.mark.parametrize("p, x, value, bs, expected", [
    (5, 1, 2, 3,
     "VerifiableShare((1 mod 5), (2 mod 5), blinding_share=(3 mod 5))"),
    (7, 0, 6, 1,
     "VerifiableShare((0 mod 7), (6 mod 7), blinding_share=(1 mod 7))"),
    (11, 10, 0, 5,
     "VerifiableShare((10 mod 11), (0 mod 11), blinding_share=(5 mod 11))"),
    (2, 1, 1, 1,
     "VerifiableShare((1 mod 2), (1 mod 2), blinding_share=(1 mod 2))"),
])
def test_repr_format(make_verifiable_share, p, x, value, bs, expected):
    assert repr(make_verifiable_share(p, x, value, bs)) == expected


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_str_and_repr_are_consistent_with_properties(make_verifiable_share,
                                                     p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert str(vs) == (f"f({vs.x})={vs.value}, "
                       f"blinding_share={vs.blinding_share}")
    assert repr(vs) == (f"VerifiableShare({vs.x}, {vs.value}, "
                        f"blinding_share={vs.blinding_share})")


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_str_extends_share_str_format(make_verifiable_share, make_share,
                                      p, x, value, bs):
    # нюанс: str(vs) — это str(Share) плюс хвост про blinding_share
    vs = make_verifiable_share(p, x, value, bs)
    share = make_share(p, x, value)
    assert str(vs).startswith(str(share))
    assert str(vs) == str(share) + f", blinding_share={vs.blinding_share}"


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_repr_starts_with_class_name_and_differs_from_str(
        make_verifiable_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert repr(vs).startswith("VerifiableShare(")
    assert repr(vs) != str(vs)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_str_of_equal_shares_is_equal(make_verifiable_share,
                                      p, x, value, bs):
    first = make_verifiable_share(p, x, value, bs)
    second = make_verifiable_share(p, x, value + p, bs + p)
    assert str(first) == str(second)
    assert repr(first) == repr(second)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_repr_of_list_of_verifiable_shares(make_verifiable_share,
                                           p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert repr([vs]) == f"[{repr(vs)}]"
    assert f"{vs}" == str(vs)


# ===========================================================================
# 8. __pos__ / __neg__
# ===========================================================================

@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_pos_returns_same_object(make_verifiable_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert (+vs) is vs


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 0, 0), (11, 5, 6, 7), (2, 1, 1, 1),
])
def test_pos_preserves_verifiable_share(make_verifiable_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert +vs == vs
    assert str(+vs) == str(vs)


@pytest.mark.parametrize("p, x, value, bs, ev, ebs", [
    (5, 1, 2, 3, 3, 2),
    (5, 1, 0, 0, 0, 0),
    (7, 3, 4, 1, 3, 6),
    (11, 5, 10, 7, 1, 4),
    (13, 7, 1, 12, 12, 1),
    (2, 1, 1, 1, 1, 1),
])
def test_neg_negates_value_and_blinding_share(make_verifiable_share,
                                              p, x, value, bs, ev, ebs):
    negated = -make_verifiable_share(p, x, value, bs)
    assert type(negated) is VerifiableShare
    assert negated.x == F_pElement(p, x)
    assert negated.value == F_pElement(p, ev)
    assert negated.blinding_share == F_pElement(p, ebs)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_double_negation_restores(make_verifiable_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert -(-vs) == vs


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_neg_returns_new_object(make_verifiable_share, p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    negated = -vs
    assert negated is not vs
    assert vs == make_verifiable_share(p, x, value, bs)


@pytest.mark.parametrize("p, x", [
    (5, 1), (7, 3), (11, 5), (13, 7), (2, 1),
])
def test_neg_of_zero_components(make_verifiable_share, p, x):
    vs = make_verifiable_share(p, x, 0, 0)
    assert -vs == vs
    assert (-vs).value == F_pElement.zero(p)
    assert (-vs).blinding_share == F_pElement.zero(p)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_neg_string_representation(make_verifiable_share, p, x, value, bs):
    negated = -make_verifiable_share(p, x, value, bs)
    assert str(negated) == (f"f({F_pElement(p, x)})={F_pElement(p, -value)}, "
                            f"blinding_share={F_pElement(p, -bs)}")


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_neg_plus_original_is_zero_verifiable_share(make_verifiable_share,
                                                    p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert vs + (-vs) == make_verifiable_share(p, x, 0, 0)


# ===========================================================================
# 9. __add__
# ===========================================================================

@pytest.mark.parametrize("p, x, v1, b1, v2, b2, ev, eb", [
    (5, 1, 2, 3, 1, 4, 3, 2),
    (5, 2, 4, 0, 4, 1, 3, 1),
    (7, 3, 3, 5, 6, 4, 2, 2),
    (11, 4, 4, 9, 9, 10, 2, 8),
    (13, 5, 6, 7, 7, 6, 0, 0),
    (2, 1, 1, 1, 1, 0, 0, 1),
    (101, 50, 30, 40, 40, 70, 70, 9),
])
def test_add_sums_values_and_blinding_shares(make_verifiable_share,
                                             p, x, v1, b1, v2, b2, ev, eb):
    result = (make_verifiable_share(p, x, v1, b1)
              + make_verifiable_share(p, x, v2, b2))
    assert type(result) is VerifiableShare
    assert result == make_verifiable_share(p, x, ev, eb)
    assert result.x == F_pElement(p, x)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_add_is_commutative(make_verifiable_share, p, x, value, bs):
    first = make_verifiable_share(p, x, value, bs)
    second = make_verifiable_share(p, x, value + 1, bs + 1)
    assert first + second == second + first


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_add_is_associative(make_verifiable_share, p, x, value, bs):
    a = make_verifiable_share(p, x, value, bs)
    b = make_verifiable_share(p, x, value + 1, bs + 2)
    c = make_verifiable_share(p, x, value + 2, bs + 3)
    assert (a + b) + c == a + (b + c)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_add_zero_verifiable_share_is_identity(make_verifiable_share,
                                               p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    zero = make_verifiable_share(p, x, 0, 0)
    assert vs + zero == vs
    assert zero + vs == vs


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_add_blinding_share_is_sum_of_blinding_shares(make_verifiable_share,
                                                      p, x, value, bs):
    first = make_verifiable_share(p, x, value, bs)
    second = make_verifiable_share(p, x, value + 1, bs + 1)
    result = first + second
    assert result.value == first.value + second.value
    assert result.blinding_share == (first.blinding_share
                                     + second.blinding_share)


@pytest.mark.parametrize("p, x1, x2, value, bs", [
    (5, 1, 2, 3, 4), (7, 0, 3, 4, 1), (11, 5, 6, 7, 8), (13, 7, 8, 9, 0),
    (101, 1, 2, 3, 4),
])
def test_add_different_points_raises_value_error(make_verifiable_share,
                                                 p, x1, x2, value, bs):
    with pytest.raises(
            ValueError,
            match="cannot \\+ with two shares with different x values"):
        (make_verifiable_share(p, x1, value, bs)
         + make_verifiable_share(p, x2, value, bs))


def test_add_different_points_message_mentions_both_shares(
        make_verifiable_share):
    # сообщение печатает str() обеих долей вместе с blinding_share
    first = make_verifiable_share(5, 1, 2, 3)
    second = make_verifiable_share(5, 2, 4, 0)
    with pytest.raises(ValueError) as exc_info:
        first + second
    assert str(exc_info.value) == (
        "cannot + with two shares with different x values: "
        "f((1 mod 5))=(2 mod 5), blinding_share=(3 mod 5) and "
        "f((2 mod 5))=(4 mod 5), blinding_share=(0 mod 5)")


@pytest.mark.parametrize("p, x, v1, bs, v2", [
    (5, 1, 2, 3, 4), (7, 3, 4, 1, 5), (11, 5, 6, 7, 8), (13, 7, 8, 9, 10),
    (101, 9, 10, 11, 12),
])
def test_add_plain_share_returns_plain_share(make_verifiable_share, make_share,
                                             p, x, v1, bs, v2):
    # согласно docstring: если один из операндов просто Share — результат Share
    vs = make_verifiable_share(p, x, v1, bs)
    share = make_share(p, x, v2)
    result = vs + share
    assert type(result) is Share
    assert not isinstance(result, VerifiableShare)
    assert result == make_share(p, x, (v1 + v2) % p)


@pytest.mark.parametrize("p, x, v1, bs, v2", [
    (5, 1, 2, 3, 4), (7, 3, 4, 1, 5), (11, 5, 6, 7, 8), (13, 7, 8, 9, 10),
    (101, 9, 10, 11, 12),
])
def test_add_plain_share_and_verifiable_returns_plain_share(
        make_verifiable_share, make_share, p, x, v1, bs, v2):
    # нюанс: Share.__check_types принимает VerifiableShare через isinstance
    # (это подкласс Share), поэтому операция симметрична по результату
    vs = make_verifiable_share(p, x, v1, bs)
    share = make_share(p, x, v2)
    result = share + vs
    assert type(result) is Share
    assert result == make_share(p, x, (v1 + v2) % p)


@pytest.mark.parametrize("p, x1, x2, value, bs", [
    (5, 1, 2, 3, 4), (7, 0, 3, 4, 1), (11, 5, 6, 7, 8),
])
def test_add_plain_share_different_points_raises_value_error(
        make_verifiable_share, make_share, p, x1, x2, value, bs):
    vs = make_verifiable_share(p, x1, value, bs)
    share = make_share(p, x2, value)
    with pytest.raises(ValueError, match="different x values"):
        vs + share


def test_add_verifiable_of_different_fields_raises_value_error_about_x():
    # нюанс: поля разные, но проверка сводится к сравнению x
    # (F_pElement разных полей не равны) — поэтому ValueError, не TypeError
    vs5 = VerifiableShare(F_pElement(5, 1), F_pElement(5, 2), F_pElement(5, 3))
    vs7 = VerifiableShare(F_pElement(7, 1), F_pElement(7, 2), F_pElement(7, 3))
    with pytest.raises(ValueError, match="different x values"):
        vs5 + vs7


@pytest.mark.parametrize("other", NON_SHARE_VALUES + [F_pElement(5, 1)])
def test_add_non_share_raises_type_error_with_verifiable_class_name(
        vs5, other):
    # нюанс: сообщение генерирует Share.__check_types, но имя класса берётся
    # из self.__class__.__name__ — 'VerifiableShare'
    with pytest.raises(
            TypeError,
            match="unsupported operand type\\(s\\) for \\+: "
                  "'VerifiableShare' and "):
        vs5 + other


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_add_does_not_mutate_operands(make_verifiable_share, p, x, value, bs):
    first = make_verifiable_share(p, x, value, bs)
    second = make_verifiable_share(p, x, value + 1, bs + 1)
    _ = first + second
    assert first == make_verifiable_share(p, x, value, bs)
    assert second == make_verifiable_share(p, x, value + 1, bs + 1)


# ===========================================================================
# 10. __sub__
# ===========================================================================

@pytest.mark.parametrize("p, x, v1, b1, v2, b2, ev, eb", [
    (5, 1, 2, 3, 1, 4, 1, 4),
    (5, 2, 0, 0, 4, 1, 1, 4),
    (7, 3, 3, 5, 6, 4, 4, 1),
    (11, 4, 4, 9, 9, 10, 6, 10),
    (13, 5, 6, 7, 6, 7, 0, 0),
    (2, 1, 1, 1, 1, 1, 0, 0),
    (101, 50, 30, 40, 40, 70, 91, 71),
])
def test_sub_subtracts_values_and_blinding_shares(make_verifiable_share,
                                                  p, x, v1, b1, v2, b2, ev, eb):
    result = (make_verifiable_share(p, x, v1, b1)
              - make_verifiable_share(p, x, v2, b2))
    assert type(result) is VerifiableShare
    assert result == make_verifiable_share(p, x, ev, eb)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_sub_self_is_zero_verifiable_share(make_verifiable_share,
                                           p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    result = vs - vs
    assert type(result) is VerifiableShare
    assert result == make_verifiable_share(p, x, 0, 0)


@pytest.mark.parametrize("p, x, v1, b1, v2, b2", [
    (5, 1, 2, 3, 1, 4), (7, 3, 4, 5, 6, 1), (11, 5, 6, 7, 8, 9),
    (13, 7, 8, 9, 10, 11), (101, 9, 1, 2, 3, 4),
])
def test_sub_is_anti_commutative(make_verifiable_share, p, x, v1, b1, v2, b2):
    first = make_verifiable_share(p, x, v1, b1)
    second = make_verifiable_share(p, x, v2, b2)
    assert first - second == -(second - first)


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_sub_zero_verifiable_share_is_identity(make_verifiable_share,
                                               p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    assert vs - make_verifiable_share(p, x, 0, 0) == vs


@pytest.mark.parametrize("p, x1, x2, value, bs", [
    (5, 1, 2, 3, 4), (7, 0, 3, 4, 1), (11, 5, 6, 7, 8), (13, 7, 8, 9, 0),
    (101, 1, 2, 3, 4),
])
def test_sub_different_points_raises_value_error(make_verifiable_share,
                                                 p, x1, x2, value, bs):
    with pytest.raises(
            ValueError,
            match="cannot - with two shares with different x values"):
        (make_verifiable_share(p, x1, value, bs)
         - make_verifiable_share(p, x2, value, bs))


@pytest.mark.parametrize("p, x, v1, bs, v2", [
    (5, 1, 2, 3, 4), (7, 3, 4, 1, 5), (11, 5, 6, 7, 8), (13, 7, 8, 9, 10),
    (101, 9, 10, 11, 12),
])
def test_sub_plain_share_returns_plain_share(make_verifiable_share, make_share,
                                             p, x, v1, bs, v2):
    vs = make_verifiable_share(p, x, v1, bs)
    share = make_share(p, x, v2)
    result = vs - share
    assert type(result) is Share
    assert not isinstance(result, VerifiableShare)
    assert result == make_share(p, x, (v1 - v2) % p)


@pytest.mark.parametrize("p, x, v1, bs, v2", [
    (5, 1, 2, 3, 4), (7, 3, 4, 1, 5), (11, 5, 6, 7, 8),
])
def test_sub_verifiable_from_plain_share_returns_plain_share(
        make_verifiable_share, make_share, p, x, v1, bs, v2):
    vs = make_verifiable_share(p, x, v1, bs)
    share = make_share(p, x, v2)
    result = share - vs
    assert type(result) is Share
    assert result == make_share(p, x, (v2 - v1) % p)


@pytest.mark.parametrize("other", NON_SHARE_VALUES + [F_pElement(5, 1)])
def test_sub_non_share_raises_type_error_with_verifiable_class_name(
        vs5, other):
    with pytest.raises(
            TypeError,
            match="unsupported operand type\\(s\\) for -: "
                  "'VerifiableShare' and "):
        vs5 - other


@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_sub_does_not_mutate_operands(make_verifiable_share, p, x, value, bs):
    first = make_verifiable_share(p, x, value, bs)
    second = make_verifiable_share(p, x, value + 1, bs + 1)
    _ = first - second
    assert first == make_verifiable_share(p, x, value, bs)
    assert second == make_verifiable_share(p, x, value + 1, bs + 1)


# ===========================================================================
# 11. Интеграция с многочленами и интерполяцией (Pedersen VSS)
# ===========================================================================

@pytest.mark.parametrize("p, f_coeffs, b_coeffs, x", [
    (5, [1, 2], [3, 4], 2), (7, [6, 5], [1, 2], 4), (11, [10, 1], [9, 8], 9),
    (13, [1, 0, 2], [5, 6], 5), (101, [7, 8], [1, 2], 100),
])
def test_make_share_matches_manual_construction(make_poly, p, f_coeffs,
                                                b_coeffs, x):
    f = make_poly(p, f_coeffs)
    b = make_poly(p, b_coeffs)
    point = F_pElement(p, x)
    assert (VerifiableShare.make(f, point, b)
            == VerifiableShare(point, f(point), b(point)))


@pytest.mark.parametrize("p, a, ba, b, bb, x", [
    (5, [1, 2], [3, 4], [2, 1], [1, 1], 2),
    (7, [6, 1], [2, 3], [1, 5], [4, 0], 5),
    (11, [1, 2, 3], [9, 8], [4, 5], [1, 1, 1], 4),
    (13, [0, 1], [1, 0, 2], [1, 0], [3, 4], 7),
    (2, [1, 1], [1], [1], [0, 1], 1),
])
def test_sum_of_verifiable_shares_equals_share_of_sums(make_poly, p, a, ba,
                                                       b, bb, x):
    point = F_pElement(p, x)
    left = (VerifiableShare.make(make_poly(p, a), point, make_poly(p, ba))
            + VerifiableShare.make(make_poly(p, b), point, make_poly(p, bb)))
    right = VerifiableShare.make(make_poly(p, a) + make_poly(p, b), point,
                                 make_poly(p, ba) + make_poly(p, bb))
    assert left == right
    assert type(left) is VerifiableShare


@pytest.mark.parametrize("p, coeffs, b_coeffs, xs", [
    (5, [1, 2], [3, 4], [1, 2]),
    (7, [3, 4, 5], [1, 2, 3], [1, 2, 3]),
    (11, [1, 2, 3], [9, 8, 7], [2, 3, 4]),
    (13, [9, 8, 7], [1, 0, 2], [4, 5, 6]),
    (101, [7, 6, 5], [1, 2, 3], [10, 20, 30]),
])
def test_verifiable_shares_are_accepted_by_interpolation(make_poly, p, coeffs,
                                                         b_coeffs, xs):
    # VerifiableShare — подкласс Share, интерполяция принимает его напрямую;
    # нюанс: LagrangeInterpolation ключует dict по share.x (F_pElement)
    f = make_poly(p, coeffs)
    b = make_poly(p, b_coeffs)
    shares = [VerifiableShare.make(f, F_pElement(p, x), b) for x in xs]
    interpolation = LagrangeInterpolation(shares)
    assert interpolation.lagrange_polynomial == f
    assert interpolation.shares == shares


@pytest.mark.parametrize("p, coeffs, b_coeffs, xs", [
    (5, [3, 1, 2], [1, 1, 1], [1, 2, 3]),
    (7, [6, 5, 4], [2, 3, 1], [1, 2, 3]),
    (11, [10, 9, 8], [1, 2, 3], [1, 2, 3]),
    (13, [1, 2, 3], [4, 5, 6], [4, 5, 6]),
    (101, [42, 7, 8], [9, 9, 9], [1, 2, 3]),
])
def test_interpolation_recovers_secret_from_verifiable_shares(
        make_poly, p, coeffs, b_coeffs, xs):
    f = make_poly(p, coeffs)
    b = make_poly(p, b_coeffs)
    shares = [VerifiableShare.make(f, F_pElement(p, x), b) for x in xs]
    secret = LagrangeInterpolation(shares)(F_pElement.zero(p))
    assert type(secret) is Share  # интерполяция возвращает обычный Share
    assert secret.value == F_pElement(p, coeffs[0])
    assert secret.x == F_pElement.zero(p)


@pytest.mark.parametrize("p, coeffs, b_coeffs, xs", [
    (5, [1, 2], [3, 4], [1, 2]),
    (7, [3, 4, 5], [1, 2, 3], [1, 2, 3]),
    (11, [1, 2, 3], [9, 8, 7], [2, 3, 4]),
    (13, [9, 8, 7], [1, 0, 2], [4, 5, 6]),
])
def test_blinding_shares_interpolate_blinding_polynomial(make_poly, p, coeffs,
                                                         b_coeffs, xs):
    # свойство Pedersen VSS: blinding-части — корректные доли b(x),
    # из t штук восстанавливается сам многочлен случайности
    f = make_poly(p, coeffs)
    b = make_poly(p, b_coeffs)
    verifiable = [VerifiableShare.make(f, F_pElement(p, x), b) for x in xs]
    blinding_shares = [Share(vs.x, vs.blinding_share) for vs in verifiable]
    interpolation = LagrangeInterpolation(blinding_shares)
    assert interpolation.lagrange_polynomial == b


@pytest.mark.parametrize("p, coeffs, b_coeffs, xs", [
    (7, [3, 4, 5], [1, 2, 3], [1, 2, 3, 4, 5]),
    (11, [1, 2, 3], [9, 8, 7], [2, 3, 4, 5]),
])
def test_extra_consistent_shares_do_not_change_interpolation(make_poly, p,
                                                             coeffs, b_coeffs,
                                                             xs):
    f = make_poly(p, coeffs)
    b = make_poly(p, b_coeffs)
    shares = [VerifiableShare.make(f, F_pElement(p, x), b) for x in xs]
    assert LagrangeInterpolation(shares).lagrange_polynomial == f
    assert (LagrangeInterpolation(shares[:len(coeffs)]).lagrange_polynomial
            == f)


@pytest.mark.parametrize("p, coeffs, b_coeffs, x", [
    (5, [1, 2], [3, 4], 3), (7, [3, 4], [5, 6], 5), (11, [9, 8], [2, 3], 2),
])
def test_verifiable_share_str_from_polynomials(make_poly, p, coeffs, b_coeffs,
                                               x):
    f = make_poly(p, coeffs)
    b = make_poly(p, b_coeffs)
    point = F_pElement(p, x)
    vs = VerifiableShare.make(f, point, b)
    assert str(vs) == f"f({point})={f(point)}, blinding_share={b(point)}"
    assert repr(vs) == (f"VerifiableShare({point}, {f(point)}, "
                        f"blinding_share={b(point)})")


# ===========================================================================
# 12. Наследование от Share
# ===========================================================================

def test_verifiable_share_is_subclass_of_share():
    assert issubclass(VerifiableShare, Share)


def test_verifiable_share_instance_is_share(vs5):
    assert isinstance(vs5, Share)


def test_plain_share_is_not_verifiable_share(make_share):
    assert not isinstance(make_share(5, 1, 2), VerifiableShare)


def test_make_signature_differs_from_share_make(make_poly):
    # VerifiableShare.make принимает ТРИ аргумента, Share.make — два
    with pytest.raises(TypeError):
        VerifiableShare.make(make_poly(5, [1]), F_pElement(5, 1))
    with pytest.raises(TypeError):
        Share.make(make_poly(5, [1]), F_pElement(5, 1), make_poly(5, [1]))


def test_share_make_returns_plain_share(make_poly):
    share = Share.make(make_poly(5, [1, 2]), F_pElement(5, 3))
    assert type(share) is Share
    assert not isinstance(share, VerifiableShare)


def test_verifiable_share_usable_where_share_expected(make_poly):
    # VerifiableShare.make(f, x, b) + Share.make(f, x) — смешанная арифметика
    f = make_poly(5, [1, 2])
    b = make_poly(5, [3, 4])
    point = F_pElement(5, 3)
    vs = VerifiableShare.make(f, point, b)
    share = Share.make(f, point)
    result = vs + share
    assert type(result) is Share
    assert result.value == F_pElement(5, 2 * 2 % 5)  # f(3)=2; 2+2=4


# ===========================================================================
# 13. Прочие нюансы
# ===========================================================================

@pytest.mark.parametrize("p, x, value, bs", [
    (5, 1, 2, 3), (7, 3, 4, 5), (11, 5, 6, 7), (13, 7, 8, 9), (101, 9, 10, 11),
])
def test_unary_operations_do_not_mutate(make_verifiable_share,
                                        p, x, value, bs):
    vs = make_verifiable_share(p, x, value, bs)
    _ = +vs
    _ = -vs
    assert vs == make_verifiable_share(p, x, value, bs)
    assert vs.x == F_pElement(p, x)
    assert vs.value == F_pElement(p, value)
    assert vs.blinding_share == F_pElement(p, bs)
