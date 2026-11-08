"""Тесты модуля src/core/environment.py — класс CryptographyEnvironment.

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.
Проверки выполняются через публичный API (свойства p / g0 / h0 / threshold,
random(), contains_SchnorrGroupElement(), contains_F_pElement(),
contains_F_pPolynomial(), ==, !=, hash(), str(), repr()) — без обращения
к приватным полям.

CryptographyEnvironment(g0, h0, threshold) — «криптографическое окружение»
для Pedersen VSS: пара порождающих элементов группы Шнорра (g0, h0) и порог
threshold. Окружение определяет группу (порядка order, модуль q = 2*order+1)
и поле показателей F_order (environment.p == g0.order), а также ограничивает
степень допустимых многочленов (degree < threshold).

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__          — валидация g0/h0 (SchnorrGroupElement, одна группа
                          по .order, оба порождающие) и threshold
                          (положительный int);
    * random            — статический конструктор случайного окружения:
                          различные случайные генераторы g0 != h0 группы
                          порядка p и заданный threshold;
    * p, g0, h0, threshold — свойства только для чтения;
    * contains_SchnorrGroupElement — принадлежность элемента группе окружения;
    * contains_F_pElement          — принадлежность элемента полю F_order;
    * contains_F_pPolynomial       — принадлежность многочлена полю F_order
                          И ограничение degree < threshold;
    * __eq__, __ne__, __hash__     — сравнение по (g0, h0, threshold);
    * __str__, __repr__            — "CryptographyEnvironment(g0=..., h0=..., threshold=...)";
    * служебные нюансы — неизменяемость аргументов, идемпотентность contains_*.

Особенности реализации, зафиксированные тестами:
    * порядок проверок в __init__: (1) isinstance(g0, h0) — TypeError;
      (2) g0.order == h0.order — ValueError "g0 and h0 must be in the same
      group"; (3) is_generator у обоих — ValueError "g0 and h0 must be
      generators"; (4) threshold — ValueError "threshold must be a positive
      integer, not ...";
    * environment.p == order группы (НЕ q = 2*order+1): поле показателей
      F_order, поэтому contains_F_pElement(F_pElement(q, ...)) == False;
    * нецелый threshold (float, str, None...) даёт ValueError, а НЕ TypeError;
    * bool — подкласс int: threshold=True принимается (== 1), False — отклоняется;
    * g0 == h0 допускается (проверяется только is_generator каждого);
    * g0 и h0 в __eq__ НЕ взаимозаменяемы: перестановка генераторов даёт
      неравные окружения;
    * contains_* не проверяют is_generator: единичный элемент группы
      «принадлежит» окружению;
    * __repr__ совпадает со __str__;
    * contains_SchnorrGroupElement сравнивает element.order с environment.p;
    * random(p, threshold): p — простое, q = 2p+1 тоже простое и p > 2
      (в группе порядка 2 единственный генератор, и поиск различного h0
      зациклился бы навсегда); проверка p <= 2 выполняется уже ПОСЛЕ
      генерации g0, но до генерации h0; для непростого/«небезопасного» p
      раньше всех срабатывает ValueError из SchnorrGroupElement.random_generator;
      threshold валидируется конструктором (после генерации g0/h0);
      цикл `while h0 == g0` гарантирует различные генераторы.
"""

import pytest

from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial
from src.core.algebra.schnorr_group import SchnorrGroupElement
from src.core.environment import CryptographyEnvironment

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

# order — простые, для которых q = 2*order + 1 тоже простое
ORDERS = [2, 3, 5, 11, 23]

# Фиксированные пары значений порождающих элементов: order -> (g0_value, h0_value).
# Все значения — квадратичные вычеты mod q и не равны 1 (кроме порядка 2,
# где единственный нетривиальный вычет 4 используется для обоих генераторов).
GENERATOR_PAIRS = {2: (4, 4), 3: (2, 4), 5: (3, 4), 11: (2, 3), 23: (2, 3)}

ORDER_PAIRS = [(2, 3), (3, 5), (2, 5), (5, 11), (3, 11), (5, 23), (11, 23)]

THRESHOLDS = [1, 2, 3, 4, 5]

NON_POSITIVE_THRESHOLDS = [0, -1, -2, -5, -100]

NON_INT_THRESHOLDS = [2.5, "2", None, [2], (2,), {}, 1.0]

NON_ELEMENT_VALUES = [0, 5, "5", None, 3.5, (5,), [5], {5}, object()]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _q(order):
    """Модуль группы Шнорра: q = 2*order + 1."""
    return 2 * order + 1


def _poly(p, coeffs):
    """Многочлен над F_p из списка целых коэффициентов (от свободного члена)."""
    return F_pPolynomial(p, coefficients=[F_pElement(p, c) for c in coeffs])


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(params=ORDERS)
def order(request):
    """Параметризованная фикстура: допустимые порядки группы Шнорра."""
    return request.param


@pytest.fixture
def make_element():
    """Фабрика элементов группы: make_element(order, value)."""
    def _make(order, value):
        return SchnorrGroupElement(order, value)
    return _make


@pytest.fixture
def make_generators():
    """Пара порождающих элементов (g0, h0) группы данного порядка."""
    def _make(order):
        g, h = GENERATOR_PAIRS[order]
        return (SchnorrGroupElement(order, g), SchnorrGroupElement(order, h))
    return _make


@pytest.fixture
def make_environment(make_generators):
    """Фабрика окружений: make_environment(order, threshold)."""
    def _make(order=5, threshold=2):
        g0, h0 = make_generators(order)
        return CryptographyEnvironment(g0, h0, threshold)
    return _make


@pytest.fixture
def environment5(make_environment):
    """Окружение над группой порядка 5 (q = 11): g0 = 3, h0 = 4, threshold = 2."""
    return make_environment(5, 2)


@pytest.fixture
def environment5_t3(make_environment):
    """Та же группа порядка 5, но threshold = 3."""
    return make_environment(5, 3)


# ===========================================================================
# 1. __init__ — валидная конструкция
# ===========================================================================

@pytest.mark.parametrize("order", ORDERS)
def test_init_accepts_valid_generators_and_threshold(make_environment, order):
    env = make_environment(order, 2)
    assert isinstance(env, CryptographyEnvironment)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_init_accepts_any_positive_threshold(make_environment, threshold):
    env = make_environment(5, threshold)
    assert env.threshold == threshold


def test_init_accepts_large_threshold(make_environment):
    env = make_environment(5, 10 ** 6 + 3)
    assert env.threshold == 10 ** 6 + 3


def test_init_accepts_equal_generators(make_element):
    # нюанс: конструктор не требует g0 != h0 (криптографически слабый,
    # но допустимый реализацией случай)
    env = CryptographyEnvironment(make_element(3, 2), make_element(3, 2), 1)
    assert env.g0 == env.h0
    assert env.threshold == 1


def test_init_stores_generators_by_reference(make_generators):
    g0, h0 = make_generators(5)
    env = CryptographyEnvironment(g0, h0, 2)
    assert env.g0 is g0
    assert env.h0 is h0


@pytest.mark.parametrize("order", ORDERS)
def test_init_accepts_random_generators(order):
    g0 = SchnorrGroupElement.random_generator(order)
    h0 = SchnorrGroupElement.random_generator(order)
    env = CryptographyEnvironment(g0, h0, 3)
    assert env.g0 == g0
    assert env.h0 == h0


def test_init_two_equal_environments_are_distinct_objects(make_environment):
    first = make_environment(5, 2)
    second = make_environment(5, 2)
    assert first is not second
    assert first == second


# ===========================================================================
# 2. __init__ — TypeError для не-SchnorrGroupElement
#    (проверка isinstance выполняется раньше всех остальных)
# ===========================================================================

@pytest.mark.parametrize("g0, h0", [
    (1, 2),
    ("g", "h"),
    (None, None),
    (3.5, 4.5),
    ([1], [2]),
    ((1,), (2,)),
    ({1}, {2}),
    (object(), object()),
    (F_pElement(11, 3), F_pElement(11, 4)),
    (_poly(5, [1]), _poly(5, [2])),
    (SchnorrGroupElement(5, 3), 4),
    (5, SchnorrGroupElement(5, 3)),
    (SchnorrGroupElement(5, 3), F_pElement(5, 1)),
])
def test_init_non_group_elements_raise_type_error(g0, h0):
    with pytest.raises(
            TypeError,
            match="g0 and h0 must be instances of SchnorrGroupElement"):
        CryptographyEnvironment(g0, h0, 2)


def test_init_type_error_message_is_exact():
    with pytest.raises(TypeError) as exc_info:
        CryptographyEnvironment(1, 2, 2)
    assert str(exc_info.value) == ("g0 and h0 must be instances of "
                                   "SchnorrGroupElement")


def test_init_error_is_type_error_not_value_error():
    with pytest.raises(TypeError):
        CryptographyEnvironment(F_pElement(5, 1), F_pElement(5, 2), 1)


def test_init_type_check_precedes_threshold_check():
    # isinstance проверяется первым: при мусорных g0/h0 и threshold=0
    # бросается TypeError про g0/h0, а не ValueError про threshold
    with pytest.raises(TypeError,
                       match="g0 and h0 must be instances"):
        CryptographyEnvironment(None, None, 0)


def test_init_type_check_precedes_order_check():
    # валидный g0 и чужеродный h0 — TypeError (isinstance),
    # а не ValueError про разные группы и не AttributeError
    with pytest.raises(TypeError,
                       match="g0 and h0 must be instances"):
        CryptographyEnvironment(SchnorrGroupElement(5, 3),
                                F_pElement(5, 1), 2)


@pytest.mark.parametrize("threshold", [1, 2, 3, 100])
def test_init_non_elements_raise_type_error_even_with_valid_threshold(threshold):
    with pytest.raises(TypeError,
                       match="g0 and h0 must be instances"):
        CryptographyEnvironment("g0", "h0", threshold)


def test_init_missing_arguments_raise_type_error():
    with pytest.raises(TypeError):
        CryptographyEnvironment(SchnorrGroupElement(5, 3))
    with pytest.raises(TypeError):
        CryptographyEnvironment(SchnorrGroupElement(5, 3),
                                SchnorrGroupElement(5, 4))
    with pytest.raises(TypeError):
        CryptographyEnvironment()


# ===========================================================================
# 3. __init__ — ValueError для генераторов из разных групп
# ===========================================================================

@pytest.mark.parametrize("order1, order2", ORDER_PAIRS)
def test_init_different_groups_raise_value_error(make_generators,
                                                 order1, order2):
    g0 = make_generators(order1)[0]
    h0 = make_generators(order2)[1]
    with pytest.raises(ValueError,
                       match="g0 and h0 must be in the same group"):
        CryptographyEnvironment(g0, h0, 2)


def test_init_different_groups_message_is_exact(make_generators):
    g0 = make_generators(3)[0]
    h0 = make_generators(5)[1]
    with pytest.raises(ValueError) as exc_info:
        CryptographyEnvironment(g0, h0, 2)
    assert str(exc_info.value) == "g0 and h0 must be in the same group"


def test_init_same_values_in_different_groups(make_element):
    # значение 4 — квадратичный вычет и mod 7 (order 3), и mod 11 (order 5),
    # но группы разные
    with pytest.raises(ValueError, match="same group"):
        CryptographyEnvironment(make_element(3, 4), make_element(5, 4), 2)


def test_init_group_check_precedes_generator_check(make_generators):
    # h0 — единица ДРУГОЙ группы: сначала срабатывает «same group» (строка 13),
    # и лишь потом «must be generators» (строка 15)
    g0 = make_generators(3)[0]
    h0 = SchnorrGroupElement.one(5)
    with pytest.raises(ValueError,
                       match="g0 and h0 must be in the same group"):
        CryptographyEnvironment(g0, h0, 2)


def test_init_group_check_precedes_threshold_check(make_generators):
    g0 = make_generators(3)[0]
    h0 = make_generators(5)[0]
    with pytest.raises(ValueError, match="same group"):
        CryptographyEnvironment(g0, h0, 0)


@pytest.mark.parametrize("order1, order2", ORDER_PAIRS[:5])
def test_init_different_groups_error_is_value_error_not_type_error(
        make_generators, order1, order2):
    g0 = make_generators(order1)[0]
    h0 = make_generators(order2)[0]
    with pytest.raises(ValueError):
        CryptographyEnvironment(g0, h0, 2)
    # и НЕ TypeError
    try:
        CryptographyEnvironment(g0, h0, 2)
    except TypeError:
        pytest.fail("разные группы должны давать ValueError, не TypeError")
    except ValueError:
        pass


# ===========================================================================
# 4. __init__ — ValueError для непорождающих элементов (единицы группы)
# ===========================================================================

@pytest.mark.parametrize("order", ORDERS)
def test_init_unit_as_g0_raises_value_error(order):
    h_value = GENERATOR_PAIRS[order][1]
    with pytest.raises(ValueError, match="g0 and h0 must be generators"):
        CryptographyEnvironment(SchnorrGroupElement.one(order),
                                SchnorrGroupElement(order, h_value), 2)


@pytest.mark.parametrize("order", ORDERS)
def test_init_unit_as_h0_raises_value_error(order):
    g_value = GENERATOR_PAIRS[order][0]
    with pytest.raises(ValueError, match="g0 and h0 must be generators"):
        CryptographyEnvironment(SchnorrGroupElement(order, g_value),
                                SchnorrGroupElement.one(order), 2)


def test_init_both_units_raise_value_error():
    with pytest.raises(ValueError, match="g0 and h0 must be generators"):
        CryptographyEnvironment(SchnorrGroupElement.one(5),
                                SchnorrGroupElement.one(5), 2)


def test_init_generator_error_message_is_exact():
    with pytest.raises(ValueError) as exc_info:
        CryptographyEnvironment(SchnorrGroupElement.one(11),
                                SchnorrGroupElement.one(11), 2)
    assert str(exc_info.value) == "g0 and h0 must be generators"


def test_init_generator_check_precedes_threshold_check():
    # обе единицы и threshold=0: сначала «must be generators» (строка 15),
    # потом «threshold must be a positive integer» (строка 20)
    with pytest.raises(ValueError, match="g0 and h0 must be generators"):
        CryptographyEnvironment(SchnorrGroupElement.one(5),
                                SchnorrGroupElement.one(5), 0)


@pytest.mark.parametrize("order", ORDERS)
def test_init_random_generators_pass_generator_check(order):
    # random_generator всегда возвращает порождающий элемент (value != 1)
    g0 = SchnorrGroupElement.random_generator(order)
    h0 = SchnorrGroupElement.random_generator(order)
    assert g0.is_generator and h0.is_generator
    env = CryptographyEnvironment(g0, h0, 1)
    assert env.threshold == 1


# ===========================================================================
# 5. __init__ — ValueError для некорректного threshold
# ===========================================================================

@pytest.mark.parametrize("threshold", NON_POSITIVE_THRESHOLDS)
def test_init_non_positive_threshold_raises_value_error(make_generators,
                                                        threshold):
    g0, h0 = make_generators(5)
    with pytest.raises(ValueError,
                       match="threshold must be a positive integer"):
        CryptographyEnvironment(g0, h0, threshold)


@pytest.mark.parametrize("threshold", [0, -1, -7])
def test_init_non_positive_threshold_message_contains_value(make_generators,
                                                            threshold):
    g0, h0 = make_generators(5)
    with pytest.raises(ValueError) as exc_info:
        CryptographyEnvironment(g0, h0, threshold)
    assert str(exc_info.value) == ("threshold must be a positive integer, "
                                   f"not {threshold}")


@pytest.mark.parametrize("threshold", NON_INT_THRESHOLDS)
def test_init_non_int_threshold_raises_value_error_not_type_error(
        make_generators, threshold):
    # нюанс: даже для совершенно чуждых типов — ValueError, а не TypeError
    g0, h0 = make_generators(5)
    with pytest.raises(ValueError,
                       match="threshold must be a positive integer"):
        CryptographyEnvironment(g0, h0, threshold)


def test_init_bool_threshold_true_is_accepted(make_generators):
    # нюанс: bool — подкласс int, поэтому True проходит проверку (threshold == 1)
    g0, h0 = make_generators(5)
    env = CryptographyEnvironment(g0, h0, True)
    assert env.threshold == 1


def test_init_bool_threshold_false_is_rejected(make_generators):
    g0, h0 = make_generators(5)
    with pytest.raises(ValueError,
                       match="threshold must be a positive integer, not False"):
        CryptographyEnvironment(g0, h0, False)


def test_init_threshold_one_is_accepted(make_generators):
    g0, h0 = make_generators(11)
    env = CryptographyEnvironment(g0, h0, 1)
    assert env.threshold == 1


# ===========================================================================
# 6. Свойства p, g0, h0, threshold
# ===========================================================================

@pytest.mark.parametrize("order", ORDERS)
def test_p_property_is_group_order(make_environment, order):
    # environment.p == order группы — поле показателей F_order,
    # а не поле значений F_q (q = 2*order+1)
    env = make_environment(order, 2)
    assert env.p == order
    assert env.p == env.g0.order
    assert env.p != _q(order)


def test_g0_property_returns_first_generator(make_generators):
    g0, h0 = make_generators(11)
    env = CryptographyEnvironment(g0, h0, 2)
    assert env.g0 == g0
    assert env.g0 is g0
    assert env.g0.value == g0.value


def test_h0_property_returns_second_generator(make_generators):
    g0, h0 = make_generators(11)
    env = CryptographyEnvironment(g0, h0, 2)
    assert env.h0 == h0
    assert env.h0 is h0
    assert env.h0 != env.g0


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_threshold_property_returns_constructor_argument(make_environment,
                                                         threshold):
    assert make_environment(5, threshold).threshold == threshold


def test_properties_are_read_only(environment5):
    for attribute in ("p", "g0", "h0", "threshold"):
        with pytest.raises(AttributeError):
            setattr(environment5, attribute, 1)


def test_properties_are_stable(environment5):
    assert environment5.p == environment5.p
    assert environment5.g0 == environment5.g0
    assert environment5.h0 == environment5.h0
    assert environment5.threshold == environment5.threshold


# ===========================================================================
# 7. contains_SchnorrGroupElement
# ===========================================================================

def test_contains_own_generators(environment5):
    assert environment5.contains_SchnorrGroupElement(environment5.g0) is True
    assert environment5.contains_SchnorrGroupElement(environment5.h0) is True


@pytest.mark.parametrize("value", [1, 3, 4, 5, 9])
def test_contains_every_element_of_own_group(environment5, value):
    # все квадратичные вычеты mod 11, включая единицу группы:
    # contains_* НЕ проверяет is_generator
    element = SchnorrGroupElement(5, value)
    assert environment5.contains_SchnorrGroupElement(element) is True


@pytest.mark.parametrize("order", [2, 3, 11, 23])
def test_does_not_contain_generator_of_other_group(environment5, order):
    g_value = GENERATOR_PAIRS[order][0]
    element = SchnorrGroupElement(order, g_value)
    assert environment5.contains_SchnorrGroupElement(element) is False


@pytest.mark.parametrize("order", [2, 3, 11, 23])
def test_does_not_contain_unit_of_other_group(environment5, order):
    assert environment5.contains_SchnorrGroupElement(
        SchnorrGroupElement.one(order)) is False


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES + [F_pElement(5, 1)])
def test_non_group_elements_are_not_contained(environment5, other):
    assert environment5.contains_SchnorrGroupElement(other) is False


@pytest.mark.parametrize("order", ORDERS)
def test_contains_random_generator_of_own_group(make_environment, order):
    env = make_environment(order, 2)
    random_generator = SchnorrGroupElement.random_generator(order)
    assert env.contains_SchnorrGroupElement(random_generator) is True


def test_contains_schnorr_group_element_returns_bool(environment5):
    result = environment5.contains_SchnorrGroupElement(environment5.g0)
    assert isinstance(result, bool)
    assert result is True


# ===========================================================================
# 8. contains_F_pElement
# ===========================================================================

@pytest.mark.parametrize("value", [0, 1, 2, 3, 4])
def test_contains_every_element_of_exponent_field(environment5, value):
    # поле окружения — F_order = F_5 (поле показателей степеней)
    assert environment5.contains_F_pElement(F_pElement(5, value)) is True


def test_contains_zero_and_one_of_exponent_field(environment5):
    assert environment5.contains_F_pElement(F_pElement.zero(5)) is True
    assert environment5.contains_F_pElement(F_pElement.one(5)) is True


@pytest.mark.parametrize("p", [2, 3, 7, 13, 23, 101])
def test_does_not_contain_elements_of_other_fields(environment5, p):
    assert environment5.contains_F_pElement(F_pElement(p, 1 % p)) is False


def test_does_not_contain_elements_of_field_q(environment5):
    # важный нюанс: environment.p == order == 5, а НЕ q == 11:
    # элементы поля F_q окружению чужие
    assert environment5.contains_F_pElement(F_pElement(11, 3)) is False


@pytest.mark.parametrize("other",
                         NON_ELEMENT_VALUES + [SchnorrGroupElement(5, 3)])
def test_non_field_elements_are_not_contained(environment5, other):
    assert environment5.contains_F_pElement(other) is False


def test_contains_f_p_element_returns_bool(environment5):
    result = environment5.contains_F_pElement(F_pElement(5, 1))
    assert isinstance(result, bool)
    assert result is True


@pytest.mark.parametrize("order", ORDERS)
def test_contains_field_elements_for_every_order(make_environment, order):
    env = make_environment(order, 2)
    assert env.contains_F_pElement(F_pElement(order, 0)) is True
    assert env.contains_F_pElement(F_pElement(order, order - 1)) is True


# ===========================================================================
# 9. contains_F_pPolynomial
# ===========================================================================

@pytest.mark.parametrize("coeffs", [[1], [1, 2], [0, 1], [4, 0], [0, 4]])
def test_contains_polynomials_of_degree_below_threshold(environment5, coeffs):
    # threshold == 2: допускаются степени 0 и 1
    assert environment5.contains_F_pPolynomial(_poly(5, coeffs)) is True


def test_contains_polynomial_of_degree_threshold_minus_one(environment5_t3):
    # threshold == 3: degree 2 == threshold - 1 — граница включается
    assert environment5_t3.contains_F_pPolynomial(_poly(5, [1, 2, 3])) is True


@pytest.mark.parametrize("coeffs", [[1, 2, 3], [0, 0, 1]])
def test_does_not_contain_polynomial_of_degree_threshold(environment5, coeffs):
    # degree 2 == threshold 2 — строгое неравенство degree < threshold
    assert environment5.contains_F_pPolynomial(_poly(5, coeffs)) is False


def test_does_not_contain_polynomial_of_higher_degree(environment5):
    assert environment5.contains_F_pPolynomial(
        _poly(5, [1, 2, 3, 4, 0, 1])) is False


def test_contains_zero_polynomial(environment5):
    # степень нулевого многочлена -inf < любого порога
    assert environment5.contains_F_pPolynomial(F_pPolynomial(5)) is True


@pytest.mark.parametrize("p", [2, 3, 7, 11, 101])
def test_does_not_contain_polynomial_of_other_field(environment5, p):
    assert environment5.contains_F_pPolynomial(_poly(p, [1])) is False
    assert environment5.contains_F_pPolynomial(F_pPolynomial(p)) is False


@pytest.mark.parametrize("other", [5, None, "poly", 3.5, [1], object(),
                                   F_pElement(5, 1),
                                   SchnorrGroupElement(5, 3)])
def test_non_polynomials_are_not_contained(environment5, other):
    assert environment5.contains_F_pPolynomial(other) is False


def test_contains_with_threshold_one_admits_only_constants(make_environment):
    env = make_environment(5, 1)
    assert env.contains_F_pPolynomial(_poly(5, [3])) is True
    assert env.contains_F_pPolynomial(F_pPolynomial(5)) is True
    assert env.contains_F_pPolynomial(_poly(5, [3, 1])) is False


def test_contains_uses_stripped_degree(environment5):
    # хвостовые нули вырезаются конструктором: [1, 2, 0, 0] имеет степень 1 < 2
    assert environment5.contains_F_pPolynomial(_poly(5, [1, 2, 0, 0])) is True


def test_contains_f_p_polynomial_returns_bool(environment5):
    result = environment5.contains_F_pPolynomial(_poly(5, [1]))
    assert isinstance(result, bool)
    assert result is True


# ===========================================================================
# 10. __eq__ / __ne__ / __hash__
# ===========================================================================

def test_eq_is_reflexive(environment5):
    assert environment5 == environment5
    assert not (environment5 != environment5)


def test_eq_for_equal_environments(make_environment):
    assert make_environment(5, 2) == make_environment(5, 2)
    assert make_environment(11, 3) == make_environment(11, 3)


@pytest.mark.parametrize("t1, t2", [(1, 2), (2, 3), (3, 5), (1, 10)])
def test_eq_false_for_different_thresholds(make_environment, t1, t2):
    assert make_environment(5, t1) != make_environment(5, t2)
    assert not (make_environment(5, t1) == make_environment(5, t2))


def test_eq_false_for_different_g0(make_element):
    first = CryptographyEnvironment(make_element(5, 3), make_element(5, 4), 2)
    second = CryptographyEnvironment(make_element(5, 5), make_element(5, 4), 2)
    assert first != second


def test_eq_false_for_different_h0(make_element):
    first = CryptographyEnvironment(make_element(5, 3), make_element(5, 4), 2)
    second = CryptographyEnvironment(make_element(5, 3), make_element(5, 9), 2)
    assert first != second


def test_eq_false_for_swapped_generators(make_element):
    # g0 и h0 НЕ взаимозаменяемы: для обязательств Педерсена важно,
    # какой генератор скрывает секрет, а какой — случайность
    first = CryptographyEnvironment(make_element(5, 3), make_element(5, 4), 2)
    second = CryptographyEnvironment(make_element(5, 4), make_element(5, 3), 2)
    assert first != second
    assert second != first


def test_eq_false_for_different_groups(make_environment):
    assert make_environment(5, 2) != make_environment(11, 2)


def test_eq_with_equal_but_distinct_elements(make_generators):
    # SchnorrGroupElement сравнивается по значению: разные, но равные
    # объекты дают равные окружения
    g0, h0 = make_generators(5)
    g0_copy, h0_copy = make_generators(5)
    first = CryptographyEnvironment(g0, h0, 2)
    second = CryptographyEnvironment(g0_copy, h0_copy, 2)
    assert first.g0 is not second.g0
    assert first == second


@pytest.mark.parametrize("other",
                         NON_ELEMENT_VALUES + [F_pElement(5, 1),
                                               SchnorrGroupElement(5, 3)])
def test_eq_with_non_environment_is_false(environment5, other):
    assert (environment5 == other) is False
    assert (environment5 != other) is True


def test_ne_is_negation_of_eq(make_environment):
    first = make_environment(5, 2)
    second = make_environment(5, 3)
    assert first != second
    assert not (first == second)


def test_hash_is_consistent_with_eq(make_environment):
    first = make_environment(5, 2)
    second = make_environment(5, 2)
    assert first == second
    assert hash(first) == hash(second)
    assert isinstance(hash(first), int)


def test_hash_equals_hash_of_triple(make_environment):
    env = make_environment(5, 2)
    assert hash(env) == hash((env.g0, env.h0, env.threshold))


def test_environment_can_be_dict_key(make_environment):
    env = make_environment(5, 2)
    mapping = {env: "params"}
    assert mapping[make_environment(5, 2)] == "params"
    assert make_environment(5, 2) in mapping


def test_distinct_environments_coexist_in_set(make_environment):
    envs = {make_environment(5, 1), make_environment(5, 2),
            make_environment(5, 2), make_environment(11, 2)}
    assert len(envs) == 3


# ===========================================================================
# 11. __str__ / __repr__
# ===========================================================================

def test_str_format(make_environment):
    env = make_environment(5, 2)
    assert str(env) == ("CryptographyEnvironment("
                        "g0=SchnorrGroupElement(5, value=3), "
                        "h0=SchnorrGroupElement(5, value=4), threshold=2)")


def test_repr_equals_str(environment5):
    assert repr(environment5) == str(environment5)


def test_str_is_consistent_with_properties(environment5):
    assert str(environment5) == (
        f"CryptographyEnvironment(g0={repr(environment5.g0)}, "
        f"h0={repr(environment5.h0)}, threshold={environment5.threshold})")


def test_str_of_equal_environments_is_equal(make_environment):
    assert str(make_environment(5, 2)) == str(make_environment(5, 2))


def test_str_differs_for_different_thresholds(make_environment):
    assert str(make_environment(5, 1)) != str(make_environment(5, 2))


def test_str_via_f_string(environment5):
    assert f"{environment5}" == str(environment5)


def test_str_mentions_generators_and_threshold(make_environment):
    env = make_environment(11, 3)
    assert repr(env.g0) in str(env)
    assert repr(env.h0) in str(env)
    assert "threshold=3" in str(env)
    assert str(env).startswith("CryptographyEnvironment(")


# ===========================================================================
# 12. Прочие нюансы
# ===========================================================================

def test_init_does_not_mutate_generators(make_generators):
    g0, h0 = make_generators(5)
    CryptographyEnvironment(g0, h0, 2)
    fresh_g0, fresh_h0 = make_generators(5)
    assert g0 == fresh_g0
    assert h0 == fresh_h0
    assert g0.is_generator and h0.is_generator


def test_contains_checks_are_idempotent(environment5):
    element = environment5.g0
    first = environment5.contains_SchnorrGroupElement(element)
    second = environment5.contains_SchnorrGroupElement(element)
    assert first is True
    assert second is True


def test_contains_methods_do_not_mutate_environment(environment5):
    environment5.contains_SchnorrGroupElement(SchnorrGroupElement.one(5))
    environment5.contains_F_pElement(F_pElement(7, 1))
    environment5.contains_F_pPolynomial(_poly(7, [1, 2, 3]))
    assert environment5.p == 5
    assert environment5.threshold == 2
    assert environment5.g0 == SchnorrGroupElement(5, 3)
    assert environment5.h0 == SchnorrGroupElement(5, 4)


def test_contains_f_p_checks_do_not_mutate_environment(environment5):
    environment5.contains_F_pElement(F_pElement(7, 1))
    environment5.contains_F_pElement(F_pElement(5, 1))
    environment5.contains_F_pPolynomial(_poly(7, [1, 2, 3]))
    environment5.contains_F_pPolynomial(_poly(5, [1]))
    assert environment5.p == 5
    assert environment5.threshold == 2
    assert environment5.g0 == SchnorrGroupElement(5, 3)
    assert environment5.h0 == SchnorrGroupElement(5, 4)


# ===========================================================================
# 13. random(p, threshold) — случайная генерация окружения
# ===========================================================================

# p — простые, для которых q = 2p + 1 тоже простое, и p > 2
SAFE_PS = [3, 5, 11, 23, 29, 41, 53, 83]

# p простое, но q = 2p + 1 составное
PRIME_NOT_SAFE_PS = [7, 13, 17, 19, 31, 37]

NON_PRIME_PS = [-5, -1, 0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 21, 25, 100]


@pytest.mark.parametrize("p", SAFE_PS)
def test_random_returns_valid_environment(p):
    env = CryptographyEnvironment.random(p, 2)
    assert isinstance(env, CryptographyEnvironment)
    assert env.p == p


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_random_preserves_threshold(threshold):
    env = CryptographyEnvironment.random(11, threshold)
    assert env.threshold == threshold


def test_random_accepts_large_threshold():
    env = CryptographyEnvironment.random(11, 10 ** 6)
    assert env.threshold == 10 ** 6


@pytest.mark.parametrize("p", SAFE_PS)
def test_random_generators_are_elements_of_group_p(p):
    env = CryptographyEnvironment.random(p, 3)
    assert isinstance(env.g0, SchnorrGroupElement)
    assert isinstance(env.h0, SchnorrGroupElement)
    assert env.g0.order == p
    assert env.h0.order == p


@pytest.mark.parametrize("p", SAFE_PS)
def test_random_generators_are_generators(p):
    env = CryptographyEnvironment.random(p, 2)
    assert env.g0.is_generator is True
    assert env.h0.is_generator is True


@pytest.mark.parametrize("p", SAFE_PS)
def test_random_generators_are_distinct(p):
    # цикл `while h0 == g0` гарантирует различные генераторы
    env = CryptographyEnvironment.random(p, 2)
    assert env.g0 != env.h0


def test_random_for_smallest_allowed_p():
    # p = 3 (q = 7): генераторов ровно два — {2, 4}, поэтому g0 != h0
    # означает пару (2, 4) или (4, 2)
    env = CryptographyEnvironment.random(3, 1)
    assert env.p == 3
    assert env.g0 != env.h0
    assert {env.g0.value, env.h0.value} == {2, 4}


def test_random_p_two_raises_value_error():
    # p = 2 (q = 5): единственный генератор 4, различить g0 и h0 невозможно
    with pytest.raises(ValueError, match="p must be greater than 2"):
        CryptographyEnvironment.random(2, 1)


def test_random_p_two_message_is_exact():
    with pytest.raises(ValueError) as exc_info:
        CryptographyEnvironment.random(2, 2)
    assert str(exc_info.value) == "p must be greater than 2"


@pytest.mark.parametrize("p", PRIME_NOT_SAFE_PS)
def test_random_prime_but_not_safe_p_raises_value_error(p):
    with pytest.raises(
            ValueError,
            match="order must be prime such that 2\\*order\\+1 is also a "
                  f"prime number, cannot be {p}"):
        CryptographyEnvironment.random(p, 2)


@pytest.mark.parametrize("p", NON_PRIME_PS)
def test_random_non_prime_p_raises_value_error(p):
    with pytest.raises(ValueError,
                       match=f"order must be prime, cannot be {p}"):
        CryptographyEnvironment.random(p, 2)


@pytest.mark.parametrize("threshold", [0, -1, -5])
def test_random_non_positive_threshold_raises_value_error(threshold):
    with pytest.raises(ValueError,
                       match="threshold must be a positive integer"):
        CryptographyEnvironment.random(11, threshold)


@pytest.mark.parametrize("threshold", [2.5, "2", None, [2], (2,)])
def test_random_non_int_threshold_raises_value_error(threshold):
    # нюанс: как и в __init__ — ValueError, а не TypeError
    with pytest.raises(ValueError,
                       match="threshold must be a positive integer"):
        CryptographyEnvironment.random(11, threshold)


def test_random_bool_threshold_is_accepted():
    # нюанс: bool — подкласс int, проверка threshold делегирована конструктору
    env = CryptographyEnvironment.random(11, True)
    assert env.threshold == 1


def test_random_is_randomized():
    # 10 вызовов дают разные окружения (порядок 23: 22 генератора,
    # 22*21 упорядоченных пар — совпадение всех 10 вызовов исключено
    # практически наверняка)
    envs = {CryptographyEnvironment.random(23, 2) for _ in range(10)}
    assert len(envs) > 1


def test_random_generators_vary_across_calls():
    generators = {CryptographyEnvironment.random(89, 3).g0
                  for _ in range(10)}
    assert len(generators) > 1


def test_random_returns_new_object_each_call():
    first = CryptographyEnvironment.random(11, 2)
    second = CryptographyEnvironment.random(11, 2)
    assert first is not second


def test_random_environment_equals_manual_construction():
    env = CryptographyEnvironment.random(11, 3)
    manual = CryptographyEnvironment(env.g0, env.h0, 3)
    assert env == manual
    assert hash(env) == hash(manual)


def test_random_environment_contains_own_generators():
    env = CryptographyEnvironment.random(23, 2)
    assert env.contains_SchnorrGroupElement(env.g0) is True
    assert env.contains_SchnorrGroupElement(env.h0) is True


def test_random_environment_contains_own_field_objects():
    env = CryptographyEnvironment.random(23, 3)
    assert env.contains_F_pElement(F_pElement(23, 5)) is True
    assert env.contains_F_pElement(F_pElement(47, 5)) is False
    assert env.contains_F_pPolynomial(_poly(23, [1, 2, 3])) is True
    assert env.contains_F_pPolynomial(_poly(23, [1, 2, 3, 4])) is False


def test_random_str_and_repr():
    env = CryptographyEnvironment.random(11, 2)
    assert str(env) == (f"CryptographyEnvironment(g0={repr(env.g0)}, "
                        f"h0={repr(env.h0)}, threshold=2)")
    assert repr(env) == str(env)
