"""Тесты модуля src/core/algebra/schnorr_group.py — класс SchnorrGroupElement.

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.
Проверки выполняются через публичный API (свойства order / value, one(),
random_generator(), ==, !=, hash(), str(), repr(), операторы *, /, **, ~).

Группа Шнорра: пусть order — простое и q = 2*order + 1 — тоже простое
(«безопасное простое»). Тогда ненулевые квадратичные вычеты по модулю q
образуют циклическую мультипликативную группу порядка order. Элемент класса
хранит order и приведённое по модулю q значение value, причём value обязан быть
квадратичным вычетом: pow(value, order, q) == 1.

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__          — валидация order (простое и 2*order+1 простое),
                          валидация value (ненулевой квадратичный вычет mod q),
                          приведение value по модулю q;
    * order, value      — свойства только для чтения, отсутствие публичного q;
    * is_generator      — признак порождающего элемента (value != 1);
    * one               — единица группы;
    * random_generator   — случайный ПОРОЖДАЮЩИЙ элемент группы
                          (его порядок ровно order);
    * __str__, __repr__ — "(value mod q)" и "SchnorrGroupElement(order, value=v)";
    * __eq__, __ne__, __hash__ — сравнение и хешируемость;
    * __mul__, __truediv__, __pow__, __invert__ — групповые операции;
    * законы группы     — замкнутость, коммутативность, ассоциативность,
                          a**order == one, существование первообразного корня;
    * служебные нюансы — стабильность состояния после операций,
      операции возвращают новые объекты.

Особенности реализации, зафиксированные тестами:
    * порядок группы order и модуль q связаны соотношением q = 2*order + 1,
      публичного свойства q нет — тесты вычисляют его сами;
    * значение приводится по модулю q, поэтому допустимы отрицательные и большие
      значения (например, -1 mod 5 = 4 при order = 2);
    * is_generator == (value != 1): порядок группы прост, поэтому любой
      неединичный элемент порождает всю группу, а единица (порядок 1) — нет;
    * 0 и квадратичные невычеты отклоняются с ValueError, причём в тексте
      сообщения печатается ИСХОДНОЕ (неприведённое) значение;
    * проверка типов выполняется через isinstance: чужой тип даёт TypeError
      "unsupported operand type(s) for * ...", а другая группа —
      "cannot * with two elements in different groups with orders 2 and 3";
    * __pow__ принимает int (включая bool — это подкласс int) и возводит
      value в степень по модулю q; показатель F_pElement тоже поддержан:
      при exp.p != order бросается TypeError "cannot raise (value mod q)
      to a power from field F_p", а при exp.p == order возводит value
      в exp.value;
    * random_generator выбирает корень из диапазона [2, q-2], поэтому значение
      никогда не равно 0 и 1; при order = 2 (q = 5) возможен единственный
      результат — 4; результат ВСЕГДА является порождающим элементом группы
      (его порядок ровно order) — это зафиксировано тестами.

Возведение в степень:
    * показатель int: pow(value, exp, q), отрицательные показатели дают
      обратный элемент;
    * показатель F_pElement: exp.p обязан совпадать с порядком группы
      (exp.p == order), иначе TypeError "cannot raise (value mod q) to a
      power from field F_p"; при совпадении возводит value в exp.value.
"""

import pytest

from src import F_pElement
from src.core.algebra.schnorr_group import SchnorrGroupElement

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

# order — простые, для которых q = 2*order + 1 тоже простое
ORDERS = [2, 3, 5, 11, 23, 29, 41, 53, 83, 89, 113]

SMALL_ORDERS = [2, 3, 5, 11, 23]

# order простое, но q = 2*order + 1 составное
PRIME_BUT_NOT_SAFE_ORDERS = [7, 13, 17, 19, 31, 37, 43, 47, 61, 67, 71, 73, 79, 97]

NON_PRIMES = [-5, -1, 0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 21, 25, 100]

NON_ELEMENT_VALUES = [0, 5, "5", None, 3.5, (5,), [5], {5}, object()]

ORDER_PAIRS = [(2, 3), (3, 5), (2, 5), (5, 11), (11, 23), (3, 11)]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _q(order):
    """Модуль группы: q = 2*order + 1."""
    return 2 * order + 1


def _residues(order):
    """Множество ненулевых квадратных вычетов по модулю q (все элементы группы)."""
    q = _q(order)
    return {pow(a, 2, q) for a in range(1, q)}


def _non_residues(order):
    """Значения mod q, которые НЕ являются элементами группы."""
    q = _q(order)
    return sorted(set(range(q)) - _residues(order))


def _is_valid(order, value):
    """Проверка, что значение — квадратичный вычет по модулю q."""
    return pow(value % _q(order), order, _q(order)) == 1


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(params=ORDERS)
def order(request):
    """Параметризованная фикстура: допустимые порядки группы."""
    return request.param


@pytest.fixture(params=SMALL_ORDERS)
def small_order(request):
    """Параметризованная фикстура: небольшие порядки группы."""
    return request.param


@pytest.fixture
def make_element():
    """Фабрика элементов: make_element(order, value) -> SchnorrGroupElement."""
    def _make(order, value):
        return SchnorrGroupElement(order, value)
    return _make


@pytest.fixture
def one_element():
    """Единица группы порядка 3 (q = 7)."""
    return SchnorrGroupElement.one(3)


@pytest.fixture
def generator3():
    """Первообразный корень группы порядка 3 (q = 7) — элемент 2."""
    return SchnorrGroupElement(3, 2)


# ===========================================================================
# 1. __init__ — валидные значения
# ===========================================================================

@pytest.mark.parametrize("order, value", [
    (2, 1), (2, 4),
    (3, 1), (3, 2), (3, 4),
    (5, 1), (5, 3), (5, 4), (5, 5), (5, 9),
    (11, 1), (11, 2), (11, 9), (11, 18),
    (23, 1), (23, 4), (23, 12),
])
def test_init_accepts_quadratic_residues(make_element, order, value):
    element = make_element(order, value)
    assert isinstance(element, SchnorrGroupElement)
    assert element.order == order
    assert element.value == value


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_init_accepts_every_residue(order):
    for value in sorted(_residues(order)):
        element = SchnorrGroupElement(order, value)
        assert element.value == value
        assert element.order == order


@pytest.mark.parametrize("order, value, expected", [
    (2, 6, 1),      # 6 mod 5 = 1
    (2, 9, 4),      # 9 mod 5 = 4
    (2, -1, 4),     # -1 mod 5 = 4
    (3, 9, 2),      # 9 mod 7 = 2
    (3, -3, 4),     # -3 mod 7 = 4
    (5, 14, 3),     # 14 mod 11 = 3
    (5, -7, 4),     # -7 mod 11 = 4
    (11, 25, 2),    # 25 mod 23 = 2
    (23, 50, 3),    # 50 mod 47 = 3
])
def test_init_reduces_value_modulo_q(make_element, order, value, expected):
    element = make_element(order, value)
    assert element.value == expected
    assert element == make_element(order, expected)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_init_value_always_in_range(order):
    q = _q(order)
    for value in _residues(order):
        for shift in (-2 * q, -q, 0, q, 2 * q):
            element = SchnorrGroupElement(order, value + shift)
            assert 1 <= element.value < q


@pytest.mark.parametrize("order", ORDERS)
def test_init_stores_order_unchanged(order):
    element = SchnorrGroupElement(order, 1)
    assert element.order == order
    assert element.value == 1


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 3), (11, 2), (23, 4),
])
def test_init_value_satisfies_group_property(make_element, order, value):
    # значение — квадратичный вычет: value^order == 1 (mod q)
    element = make_element(order, value)
    assert pow(element.value, order, _q(order)) == 1


@pytest.mark.parametrize("order, value", [
    (2, 1), (3, 4), (5, 9), (11, 18), (23, 12),
])
def test_init_accepts_keyword_value(make_element, order, value):
    assert SchnorrGroupElement(order, value=value) == make_element(order, value)


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_init_is_deterministic(make_element, order, value):
    first = make_element(order, value)
    second = make_element(order, value)
    assert first == second
    assert first.value == second.value
    assert first.order == second.order
    assert _q(order) == 2 * order + 1


# ===========================================================================
# 2. __init__ — исключения
# ===========================================================================

@pytest.mark.parametrize("order", NON_PRIMES)
def test_init_non_prime_order_raises_value_error(order):
    with pytest.raises(ValueError, match=f"order must be prime, cannot be {order}"):
        SchnorrGroupElement(order, 1)


@pytest.mark.parametrize("order", PRIME_BUT_NOT_SAFE_ORDERS)
def test_init_prime_but_not_safe_order_raises_value_error(order):
    with pytest.raises(
            ValueError,
            match="order must be prime such that 2\\*order\\+1 is also a prime"):
        SchnorrGroupElement(order, 1)


@pytest.mark.parametrize("order", PRIME_BUT_NOT_SAFE_ORDERS[:6])
def test_init_not_safe_order_message_mentions_order(order):
    with pytest.raises(ValueError) as exc_info:
        SchnorrGroupElement(order, 1)
    assert str(order) in str(exc_info.value)


@pytest.mark.parametrize("order, value", [
    (2, 0), (2, 2), (2, 3),
    (3, 0), (3, 3), (3, 5), (3, 6),
    (5, 0), (5, 2), (5, 6), (5, 7), (5, 8), (5, 10),
    (11, 5), (11, 7), (11, 10), (11, 22),
    (23, 10), (23, 13), (23, 46),
])
def test_init_non_residue_value_raises_value_error(make_element, order, value):
    q = _q(order)
    with pytest.raises(ValueError,
                       match=f"value must be a non-zero square mod {q}"):
        make_element(order, value)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_init_rejects_all_non_residues(order):
    q = _q(order)
    for value in _non_residues(order):
        with pytest.raises(ValueError,
                           match=f"value must be a non-zero square mod {q}"):
            SchnorrGroupElement(order, value)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_init_accepts_exactly_residues(order):
    q = _q(order)
    accepted = set()
    for value in range(q):
        if _is_valid(order, value):
            accepted.add(value)
    assert accepted == _residues(order)
    assert len(accepted) == order


@pytest.mark.parametrize("order, value", [
    (2, 5), (2, 10), (3, 7), (3, 14), (5, 11), (11, 23), (23, 47),
])
def test_init_value_equal_to_q_becomes_zero_and_is_rejected(
        make_element, order, value):
    # value mod q == 0 -> ноль не входит в мультипликативную группу
    with pytest.raises(ValueError, match="value must be a non-zero square"):
        make_element(order, value)


@pytest.mark.parametrize("order, value", [
    (2, 8), (3, 10), (5, 13), (11, 28), (23, 52),
])
def test_init_error_message_prints_original_value(make_element, order, value):
    q = _q(order)
    with pytest.raises(ValueError) as exc_info:
        make_element(order, value)
    assert f"cannot be ({value} mod {q})" in str(exc_info.value)


@pytest.mark.parametrize("value", ["1", None, [1], object()])
def test_init_non_numeric_value_raises_type_error(value):
    with pytest.raises(TypeError):
        SchnorrGroupElement(3, value)


@pytest.mark.parametrize("order", [4, 6, 8, 9, 10])
def test_init_order_check_precedes_value_check(order):
    # сначала проверяется order, поэтому недопустимое value не проверяется
    with pytest.raises(ValueError, match="order must be prime"):
        SchnorrGroupElement(order, 0)


# ===========================================================================
# 3. Свойства order и value
# ===========================================================================

@pytest.mark.parametrize("order", ORDERS)
def test_order_property_returns_group_order(order):
    assert SchnorrGroupElement(order, 1).order == order


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_value_property_returns_reduced_value(make_element, order, value):
    assert make_element(order, value).value == value


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_properties_are_read_only(make_element, order, value):
    element = make_element(order, value)
    for attribute in ("order", "value"):
        with pytest.raises(AttributeError):
            setattr(element, attribute, 1)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_value_is_never_zero(order):
    for value in _residues(order):
        assert SchnorrGroupElement(order, value).value != 0


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_group_has_exactly_order_elements(order):
    elements = {SchnorrGroupElement(order, value) for value in _residues(order)}
    assert len(elements) == order
    assert len(_residues(order)) == order


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_properties_are_stable(make_element, order, value):
    element = make_element(order, value)
    assert element.order == element.order == order
    assert element.value == element.value == value


# ===========================================================================
# 4. one(order) — единица группы
# ===========================================================================

@pytest.mark.parametrize("order", ORDERS)
def test_one_has_value_one(order):
    one = SchnorrGroupElement.one(order)
    assert isinstance(one, SchnorrGroupElement)
    assert one.value == 1
    assert one.order == order


@pytest.mark.parametrize("order", ORDERS)
def test_one_equals_constructor(order):
    assert SchnorrGroupElement.one(order) == SchnorrGroupElement(order, 1)
    assert not (SchnorrGroupElement.one(order) != SchnorrGroupElement(order, 1))


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_one_string_and_repr(order):
    one = SchnorrGroupElement.one(order)
    assert str(one) == f"(1 mod {_q(order)})"
    assert repr(one) == f"SchnorrGroupElement({order}, value=1)"


@pytest.mark.parametrize("order", NON_PRIMES[:6])
def test_one_non_prime_order_raises_value_error(order):
    with pytest.raises(ValueError, match="order must be prime"):
        SchnorrGroupElement.one(order)


@pytest.mark.parametrize("order", PRIME_BUT_NOT_SAFE_ORDERS[:6])
def test_one_not_safe_order_raises_value_error(order):
    with pytest.raises(ValueError, match="2\\*order\\+1 is also a prime"):
        SchnorrGroupElement.one(order)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_one_returns_new_object_each_call(order):
    assert SchnorrGroupElement.one(order) is not SchnorrGroupElement.one(order)


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_one_is_multiplicative_identity(make_element, order, value):
    element = make_element(order, value)
    one = SchnorrGroupElement.one(order)
    assert (element * one).value == value
    assert (one * element).value == value


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_one_times_one_is_one(order):
    one = SchnorrGroupElement.one(order)
    assert (one * one).value == 1


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_one_to_any_power_is_one(order):
    one = SchnorrGroupElement.one(order)
    for exponent in (0, 1, 2, 5, order):
        assert (one ** exponent).value == 1


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_inverse_of_one_is_one(order):
    assert (~SchnorrGroupElement.one(order)).value == 1


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_one_hash_and_str_are_stable(order):
    first = SchnorrGroupElement.one(order)
    second = SchnorrGroupElement.one(order)
    assert hash(first) == hash(second)
    assert str(first) == str(second)
    assert len({first, second}) == 1


# ===========================================================================
# 5. random_generator(order) — случайный элемент
# ===========================================================================

@pytest.mark.parametrize("order", ORDERS)
def test_random_generator_returns_element_with_valid_value(order):
    for _ in range(10):
        element = SchnorrGroupElement.random_generator(order)
        assert isinstance(element, SchnorrGroupElement)
        assert _is_valid(order, element.value)


@pytest.mark.parametrize("order", ORDERS)
def test_random_generator_value_in_range(order):
    q = _q(order)
    for _ in range(20):
        value = SchnorrGroupElement.random_generator(order).value
        assert 1 <= value < q


@pytest.mark.parametrize("order", ORDERS)
def test_random_generator_never_returns_zero_or_one(order):
    # корень выбирается из [2, q-2], поэтому значение не может быть 1
    values = {SchnorrGroupElement.random_generator(order).value
              for _ in range(100)}
    assert 0 not in values
    assert 1 not in values


@pytest.mark.parametrize("order", [3, 5, 11, 23, 29])
def test_random_generator_produces_different_values(order):
    values = {SchnorrGroupElement.random_generator(order).value
              for _ in range(200)}
    assert len(values) > 1


@pytest.mark.parametrize("count", [1, 5, 20])
def test_random_generator_for_smallest_group(count):
    # нюанс: при order = 2 (q = 5) корень ∈ {2, 3}, а 4 — единственный
    # нетривиальный вычет, поэтому результат всегда один и тот же
    for _ in range(count):
        assert SchnorrGroupElement.random_generator(2).value == 4


@pytest.mark.parametrize("order", NON_PRIMES)
def test_random_generator_non_prime_order_raises_value_error(order):
    with pytest.raises(ValueError, match=f"order must be prime, cannot be {order}"):
        SchnorrGroupElement.random_generator(order)


@pytest.mark.parametrize("order", PRIME_BUT_NOT_SAFE_ORDERS)
def test_random_generator_not_safe_order_raises_value_error(order):
    with pytest.raises(ValueError, match="2\\*order\\+1 is also a prime"):
        SchnorrGroupElement.random_generator(order)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_random_generator_exposes_order(order):
    # _new присваивает __order, поэтому свойство order теперь доступно
    element = SchnorrGroupElement.random_generator(order)
    assert element.order == order
    assert repr(element) == (f"SchnorrGroupElement({order}, "
                             f"value={element.value})")


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_random_generator_str_and_repr(order):
    element = SchnorrGroupElement.random_generator(order)
    q = _q(order)
    assert str(element) == f"({element.value} mod {q})"
    assert repr(element) == (f"SchnorrGroupElement({order}, "
                             f"value={element.value})")


@pytest.mark.parametrize("order", [3, 5, 11])
def test_random_generator_values_cover_group_except_one(order):
    values = {SchnorrGroupElement.random_generator(order).value
              for _ in range(500)}
    assert values <= _residues(order)
    assert values == _residues(order) - {1}


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_random_generator_result_is_usable_in_arithmetic(order):
    first = SchnorrGroupElement.random_generator(order)
    second = SchnorrGroupElement.random_generator(order)
    product = first * second
    assert isinstance(product, SchnorrGroupElement)
    assert product.order == order


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_random_generator_result_can_be_compared(order):
    first = SchnorrGroupElement.random_generator(order)
    second = SchnorrGroupElement(order, first.value)
    assert first == second
    assert hash(first) == hash(second)


# ===========================================================================
# 6. __str__ / __repr__
# ===========================================================================

@pytest.mark.parametrize("order, value, expected", [
    (2, 1, "(1 mod 5)"),
    (2, 4, "(4 mod 5)"),
    (3, 2, "(2 mod 7)"),
    (3, 4, "(4 mod 7)"),
    (5, 3, "(3 mod 11)"),
    (5, 9, "(9 mod 11)"),
    (11, 18, "(18 mod 23)"),
    (23, 12, "(12 mod 47)"),
])
def test_str_format(make_element, order, value, expected):
    assert str(make_element(order, value)) == expected


@pytest.mark.parametrize("order, value, expected", [
    (2, 6, "(1 mod 5)"),
    (2, -1, "(4 mod 5)"),
    (3, 9, "(2 mod 7)"),
    (5, 14, "(3 mod 11)"),
    (11, 25, "(2 mod 23)"),
    (23, 50, "(3 mod 47)"),
])
def test_str_uses_reduced_value(make_element, order, value, expected):
    assert str(make_element(order, value)) == expected


@pytest.mark.parametrize("order, value, expected", [
    (2, 1, "SchnorrGroupElement(2, value=1)"),
    (2, 4, "SchnorrGroupElement(2, value=4)"),
    (3, 2, "SchnorrGroupElement(3, value=2)"),
    (5, 9, "SchnorrGroupElement(5, value=9)"),
    (11, 18, "SchnorrGroupElement(11, value=18)"),
    (23, 12, "SchnorrGroupElement(23, value=12)"),
])
def test_repr_format(make_element, order, value, expected):
    assert repr(make_element(order, value)) == expected


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_repr_is_consistent_with_properties(make_element, order, value):
    element = make_element(order, value)
    assert repr(element) == (f"SchnorrGroupElement({element.order}, "
                             f"value={element.value})")


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_str_differs_from_repr(make_element, order, value):
    element = make_element(order, value)
    assert str(element) != repr(element)
    assert f"{element}" == str(element)


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_str_of_equal_elements_is_equal(make_element, order, value):
    q = _q(order)
    first = make_element(order, value)
    second = make_element(order, value + q)
    assert first == second
    assert str(first) == str(second)
    assert repr(first) == repr(second)


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_repr_of_list_of_elements(make_element, order, value):
    element = make_element(order, value)
    assert repr([element]) == f"[{repr(element)}]"


# ===========================================================================
# 7. __eq__ / __ne__ / __hash__
# ===========================================================================

@pytest.mark.parametrize("order, value", [
    (2, 1), (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_eq_is_reflexive(make_element, order, value):
    element = make_element(order, value)
    assert element == element
    assert not (element != element)


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_eq_true_for_equal_elements(make_element, order, value):
    assert make_element(order, value) == make_element(order, value)
    assert not (make_element(order, value) != make_element(order, value))


@pytest.mark.parametrize("order, value, shift", [
    (2, 4, 5), (2, 1, 10), (3, 2, 7), (5, 9, 11), (11, 18, 23), (23, 12, 47),
])
def test_eq_true_for_values_differing_by_q(make_element, order, value, shift):
    assert make_element(order, value) == make_element(order, value + shift)
    assert make_element(order, value) == make_element(order, value - shift)


@pytest.mark.parametrize("order, v1, v2", [
    (3, 2, 4), (5, 3, 4), (5, 3, 9), (11, 2, 9), (23, 4, 12), (2, 1, 4),
])
def test_eq_false_for_different_values(make_element, order, v1, v2):
    assert make_element(order, v1) != make_element(order, v2)
    assert not (make_element(order, v1) == make_element(order, v2))


@pytest.mark.parametrize("order1, order2", ORDER_PAIRS)
def test_eq_false_for_different_groups(make_element, order1, order2):
    assert make_element(order1, 1) != make_element(order2, 1)
    assert not (make_element(order1, 1) == make_element(order2, 1))


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES)
def test_eq_with_non_element_is_false(make_element, other):
    assert (make_element(3, 2) == other) is False
    assert (make_element(3, 2) != other) is True


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES[:6])
def test_eq_reflected_with_non_element_is_false(make_element, other):
    assert (other == make_element(3, 2)) is False
    assert (other != make_element(3, 2)) is True


@pytest.mark.parametrize("order, v1, v2, v3", [
    (5, 3, 3, 4), (11, 2, 2, 9), (23, 4, 4, 12), (3, 2, 2, 4), (2, 1, 1, 4),
])
def test_eq_is_symmetric_and_transitive(make_element, order, v1, v2, v3):
    first = make_element(order, v1)
    second = make_element(order, v2)
    third = make_element(order, v3)
    assert (first == second) == (second == first)
    assert first == second
    assert second != third
    assert first != third


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_hash_is_consistent_with_eq(make_element, order, value):
    first = make_element(order, value)
    second = make_element(order, value)
    assert first == second
    assert hash(first) == hash(second)
    assert isinstance(hash(first), int)


@pytest.mark.parametrize("order, value, shift", [
    (2, 4, 5), (3, 2, 7), (5, 9, 11), (11, 18, 23), (23, 12, 47),
])
def test_hash_ignores_reduction(make_element, order, value, shift):
    first = make_element(order, value)
    second = make_element(order, value + shift)
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_all_group_elements_are_distinct_in_set(order):
    elements = {SchnorrGroupElement(order, value) for value in _residues(order)}
    assert len(elements) == order


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_element_can_be_dict_key(make_element, order, value):
    element = make_element(order, value)
    mapping = {element: "secret"}
    assert mapping[make_element(order, value)] == "secret"
    assert make_element(order, value) in mapping


@pytest.mark.parametrize("order1, order2", ORDER_PAIRS)
def test_elements_of_different_groups_coexist_in_set(make_element, order1, order2):
    assert len({make_element(order1, 1), make_element(order2, 1)}) == 2


# ===========================================================================
# 8. __mul__
# ===========================================================================

@pytest.mark.parametrize("order, v1, v2, expected", [
    (2, 1, 1, 1),
    (2, 1, 4, 4),
    (2, 4, 4, 1),
    (3, 2, 2, 4),
    (3, 2, 4, 1),
    (3, 4, 4, 2),
    (5, 3, 3, 9),
    (5, 3, 4, 1),
    (5, 3, 5, 4),
    (5, 3, 9, 5),
    (5, 4, 4, 5),
    (5, 4, 5, 9),
    (5, 4, 9, 3),
    (5, 5, 5, 3),
    (5, 5, 9, 1),
    (5, 9, 9, 4),
    (11, 2, 3, 6),
    (11, 8, 9, 3),
    (11, 12, 13, 18),
    (11, 16, 18, 12),
    (23, 2, 3, 6),
    (23, 8, 9, 25),
    (23, 12, 14, 27),
    (23, 16, 18, 6),
])
def test_mul_values(make_element, order, v1, v2, expected):
    result = make_element(order, v1) * make_element(order, v2)
    assert result.value == expected


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_mul_is_closed_in_group(order):
    residues = sorted(_residues(order))
    for v1 in residues:
        for v2 in residues:
            product = SchnorrGroupElement(order, v1) * SchnorrGroupElement(order, v2)
            assert product.value in _residues(order)
            assert _is_valid(order, product.value)


@pytest.mark.parametrize("order, v1, v2", [
    (3, 2, 4), (5, 3, 9), (11, 2, 8), (23, 4, 12), (2, 1, 4),
])
def test_mul_is_commutative(make_element, order, v1, v2):
    first = make_element(order, v1)
    second = make_element(order, v2)
    assert (first * second).value == (second * first).value


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_mul_by_one_is_identity(make_element, order, value):
    element = make_element(order, value)
    one = SchnorrGroupElement.one(order)
    assert (element * one).value == value
    assert (one * element).value == value


@pytest.mark.parametrize("order, v1, v2, v3", [
    (5, 3, 4, 5), (11, 2, 3, 8), (23, 4, 6, 9), (3, 2, 4, 2), (2, 4, 4, 1),
])
def test_mul_is_associative(make_element, order, v1, v2, v3):
    first = make_element(order, v1)
    second = make_element(order, v2)
    third = make_element(order, v3)
    assert (first * second) * third == first * (second * third)


@pytest.mark.parametrize("order, v1, v2, v3", [
    (5, 3, 4, 5), (11, 2, 3, 8), (23, 4, 6, 9), (3, 2, 4, 2), (2, 4, 4, 1),
])
def test_mul_associativity_by_value(make_element, order, v1, v2, v3):
    q = _q(order)
    left = (v1 * v2 % q) * v3 % q
    right = v1 * (v2 * v3 % q) % q
    first, second, third = (make_element(order, v1), make_element(order, v2),
                            make_element(order, v3))
    assert ((first * second).value * v3) % q == left
    assert (v1 * (second * third).value) % q == right


@pytest.mark.parametrize("order1, order2", ORDER_PAIRS)
def test_mul_different_groups_raises_type_error(make_element, order1, order2):
    with pytest.raises(
            TypeError,
            match="cannot \\* with two elements in different groups with orders "
                  f"{order1} and {order2}"):
        make_element(order1, 1) * make_element(order2, 1)


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES)
def test_mul_non_element_raises_type_error(make_element, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_element(3, 2) * other


@pytest.mark.parametrize("order, v1, v2", [
    (3, 2, 4), (5, 3, 9), (11, 2, 8), (23, 4, 12), (2, 1, 4),
])
def test_mul_result_is_comparable_element(make_element, order, v1, v2):
    product = make_element(order, v1) * make_element(order, v2)
    expected_value = (v1 * v2) % _q(order)
    assert isinstance(product, SchnorrGroupElement)
    assert product.order == order
    assert product == make_element(order, expected_value)
    assert str(product) == f"({expected_value} mod {_q(order)})"


@pytest.mark.parametrize("order, v1, v2", [
    (3, 2, 4), (5, 3, 9), (11, 2, 8), (23, 4, 12), (2, 1, 4),
])
def test_mul_does_not_mutate_operands(make_element, order, v1, v2):
    first = make_element(order, v1)
    second = make_element(order, v2)
    _ = first * second
    assert first == make_element(order, v1)
    assert second == make_element(order, v2)


# ===========================================================================
# 9. __pow__
# ===========================================================================

@pytest.mark.parametrize("order, value, exp, expected", [
    (2, 4, 0, 1),
    (2, 4, 1, 4),
    (2, 4, 2, 1),
    (2, 4, 3, 4),
    (3, 2, 1, 2),
    (3, 2, 2, 4),
    (3, 2, 3, 1),
    (3, 4, 2, 2),
    (5, 3, 2, 9),
    (5, 3, 3, 5),
    (5, 3, 4, 4),
    (5, 3, 5, 1),
    (11, 2, 2, 4),
    (11, 2, 3, 8),
    (11, 2, 11, 1),
    (23, 2, 2, 4),
    (23, 2, 5, 32),
    (23, 2, 11, 27),
    (23, 2, 23, 1),
])
def test_pow_values(make_element, order, value, exp, expected):
    assert (make_element(order, value) ** exp).value == expected


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_pow_matches_builtin_pow(order):
    q = _q(order)
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        for exponent in range(0, order + 2):
            assert (element ** exponent).value == pow(value, exponent, q)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_pow_zero_exponent_is_one(order):
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        assert (element ** 0).value == 1


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_lagrange_theorem_pow_order_is_one(order):
    # порядок группы равен order, поэтому a**order == 1 для всех a
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        assert (element ** order).value == 1


@pytest.mark.parametrize("order, value, expected", [
    (2, 4, 4),
    (3, 2, 4),
    (3, 4, 2),
    (5, 3, 4),
    (5, 5, 9),
    (11, 2, 12),
    (23, 2, 24),
])
def test_pow_minus_one_is_inverse(make_element, order, value, expected):
    assert (make_element(order, value) ** -1).value == expected


@pytest.mark.parametrize("order, value, e1, e2", [
    (3, 2, 1, 2), (5, 3, 2, 3), (11, 2, 3, 4), (23, 4, 2, 5), (2, 4, 1, 1),
])
def test_pow_adds_exponents_by_value(make_element, order, value, e1, e2):
    q = _q(order)
    element = make_element(order, value)
    left = ((element ** e1).value * (element ** e2).value) % q
    assert left == (element ** (e1 + e2)).value


@pytest.mark.parametrize("exp", [2.0, 1.5, "2", None, [2], 3.5])
def test_pow_non_int_exponent_raises_type_error(make_element, exp):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_element(3, 2) ** exp


@pytest.mark.parametrize("exp, expected", [(True, 2), (False, 1)])
def test_pow_bool_exponent_is_accepted_as_int(make_element, exp, expected):
    # нюанс: bool — подкласс int, поэтому isinstance(exp, int) пропускает его
    assert (make_element(3, 2) ** exp).value == expected


@pytest.mark.parametrize("order, value, exponent", [
    (3, 2, 2), (5, 3, 3), (11, 2, 4), (23, 4, 5), (2, 4, 1),
])
def test_pow_with_field_element_exponent(make_element, order, value, exponent):
    # показатель F_pElement из поля F_order: возводит value в exp.value
    element = make_element(order, value)
    result = element ** F_pElement(order, exponent)
    expected = make_element(order, pow(value, exponent, _q(order)))
    assert result == expected
    assert result.order == order
    assert result.value == pow(value, exponent, _q(order))
    assert str(result) == str(expected)


@pytest.mark.parametrize("order, other_p, value", [
    (3, 5, 2), (5, 2, 3), (11, 7, 2), (23, 5, 4), (2, 3, 4),
])
def test_pow_with_field_element_of_other_field_raises_type_error(
        make_element, order, other_p, value):
    element = make_element(order, value)
    with pytest.raises(
            TypeError,
            match=f"cannot raise \\({value} mod {_q(order)}\\) to a power "
                  f"from field F_{other_p}"):
        element ** F_pElement(other_p, value)


@pytest.mark.parametrize("order, value, exp", [
    (3, 2, 2), (5, 3, 3), (11, 2, 4), (23, 4, 5), (2, 4, 2),
])
def test_pow_result_is_comparable_element(make_element, order, value, exp):
    result = make_element(order, value) ** exp
    expected = make_element(order, pow(value, exp, _q(order)))
    assert result == expected
    assert result.order == order
    assert str(result) == str(expected)


@pytest.mark.parametrize("order, value", [
    (3, 2), (5, 3), (11, 2), (23, 4), (2, 4),
])
def test_pow_does_not_mutate_base(make_element, order, value):
    element = make_element(order, value)
    _ = element ** 3
    assert element == make_element(order, value)
    assert element.value == value


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_pow_result_value_is_in_group(order):
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        for exponent in (2, 3, order - 1):
            assert (element ** exponent).value in _residues(order)


# ===========================================================================
# 10. __invert__
# ===========================================================================

@pytest.mark.parametrize("order, value, expected", [
    (2, 1, 1),
    (2, 4, 4),
    (3, 2, 4),
    (3, 4, 2),
    (5, 3, 4),
    (5, 4, 3),
    (5, 5, 9),
    (5, 9, 5),
    (11, 2, 12),
    (11, 3, 8),
    (11, 12, 2),
    (11, 18, 9),
    (23, 2, 24),
    (23, 3, 16),
    (23, 18, 34),
])
def test_invert_values(make_element, order, value, expected):
    assert (~make_element(order, value)).value == expected


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_invert_matches_builtin_pow(order):
    q = _q(order)
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        assert (~element).value == pow(value, -1, q)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_inverse_is_in_group(order):
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        assert (~element).value in _residues(order)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_element_times_inverse_is_one(order):
    # левый операнд валиден, поэтому проверка типов и вычисление проходят
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        product = element * ~element
        assert product.value == 1
        assert product == SchnorrGroupElement.one(order)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_element_times_inverse_is_one_by_value(order):
    q = _q(order)
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        assert (value * (~element).value) % q == 1


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_double_inverse_restores_element(order):
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        assert ~~element == element


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_inverse_of_inverse_by_value(order):
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        inverse = SchnorrGroupElement(order, (~element).value)
        assert (~inverse).value == value


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_inverse_of_one_is_one(order):
    assert (~SchnorrGroupElement.one(order)).value == 1


@pytest.mark.parametrize("order, value", [
    (3, 2), (5, 3), (11, 2), (23, 4), (2, 4),
])
def test_invert_does_not_mutate_element(make_element, order, value):
    element = make_element(order, value)
    _ = ~element
    assert element == make_element(order, value)
    assert element.value == value


# ===========================================================================
# 11. __truediv__
# ===========================================================================

@pytest.mark.parametrize("order, v1, v2, expected", [
    (3, 4, 2, 2),
    (3, 2, 4, 4),
    (5, 3, 4, 9),
    (5, 5, 9, 3),
    (5, 9, 3, 3),
    (11, 6, 3, 2),
    (11, 12, 9, 9),
    (23, 6, 3, 2),
    (23, 12, 9, 17),
    (2, 4, 4, 1),
])
def test_div_values(make_element, order, v1, v2, expected):
    # левый операнд валиден, поэтому вычисление проходит
    assert (make_element(order, v1) / make_element(order, v2)).value == expected


@pytest.mark.parametrize("order, v1, v2", [
    (3, 4, 2), (5, 3, 4), (11, 6, 3), (23, 12, 9), (2, 4, 1),
])
def test_div_matches_manual_inverse_multiplication(make_element, order, v1, v2):
    q = _q(order)
    first = make_element(order, v1)
    second = make_element(order, v2)
    expected = (v1 * pow(v2, -1, q)) % q
    assert (first / second).value == expected


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_div_by_self_is_one(make_element, order, value):
    element = make_element(order, value)
    assert element / element == SchnorrGroupElement.one(order)


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_div_by_one_is_identity(make_element, order, value):
    element = make_element(order, value)
    assert element / SchnorrGroupElement.one(order) == element


@pytest.mark.parametrize("order, v1, v2", [
    (3, 2, 4), (5, 3, 9), (11, 2, 8), (23, 4, 12), (2, 1, 4),
])
def test_div_equals_mul_by_inverse(make_element, order, v1, v2):
    first = make_element(order, v1)
    second = make_element(order, v2)
    assert first / second == first * ~second


@pytest.mark.parametrize("order1, order2", ORDER_PAIRS)
def test_div_different_groups_raises_type_error(make_element, order1, order2):
    with pytest.raises(
            TypeError,
            match="cannot / with two elements in different groups with orders "
                  f"{order1} and {order2}"):
        make_element(order1, 1) / make_element(order2, 1)


@pytest.mark.parametrize("other", NON_ELEMENT_VALUES)
def test_div_non_element_raises_type_error(make_element, other):
    with pytest.raises(TypeError, match="unsupported operand type"):
        make_element(3, 2) / other


# ===========================================================================
# 12. Законы группы
# ===========================================================================

@pytest.mark.parametrize("order, generator", [
    (2, 4), (3, 2), (5, 3), (11, 2), (23, 2),
])
def test_generator_powers_cover_whole_group(make_element, order, generator):
    element = make_element(order, generator)
    powers = {(element ** exponent).value for exponent in range(1, order + 1)}
    assert powers == _residues(order)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_group_is_abelian(order):
    residues = sorted(_residues(order))
    for v1 in residues:
        for v2 in residues:
            first = SchnorrGroupElement(order, v1)
            second = SchnorrGroupElement(order, v2)
            assert (first * second).value == (second * first).value


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_identity_element_is_unique(order):
    one = SchnorrGroupElement.one(order)
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        assert (element * one).value == value
        assert (element == one) is (value == 1)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_every_element_has_inverse(order):
    q = _q(order)
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        inverse_value = (~element).value
        assert (value * inverse_value) % q == 1
        assert SchnorrGroupElement(order, inverse_value) == \
               SchnorrGroupElement(order, pow(value, -1, q))


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_operation_table_is_closed_and_deterministic(order):
    q = _q(order)
    residues = sorted(_residues(order))
    table = {}
    for v1 in residues:
        for v2 in residues:
            first = SchnorrGroupElement(order, v1)
            second = SchnorrGroupElement(order, v2)
            product = (first * second).value
            assert product == (v1 * v2) % q
            table[(v1, v2)] = product
    assert len(table) == len(residues) ** 2


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_square_of_any_element_is_residue(order):
    q = _q(order)
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        square = (element ** 2).value
        assert square == pow(value, 2, q)
        assert _is_valid(order, square)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_element_order_divides_group_order(order):
    # порядок любого элемента делит порядок группы (следствие теоремы Лагранжа)
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        current = 1
        for exponent in range(1, order + 1):
            current = (element ** exponent).value
            if current == 1:
                assert order % exponent == 0
                break
        else:
            pytest.fail(f"element {value} of order {order} has no finite order")


# ===========================================================================
# 13. Прочие нюансы
# ===========================================================================

@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_element_state_is_stable_after_operations(make_element, order, value):
    element = make_element(order, value)
    _ = element * make_element(order, 1)
    _ = element ** 2
    _ = ~element
    assert element.order == order
    assert element.value == value
    assert str(element) == f"({value} mod {_q(order)})"
    assert repr(element) == f"SchnorrGroupElement({order}, value={value})"


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_elements_of_same_group_are_hashable_and_unique(order):
    elements = [SchnorrGroupElement(order, value) for value in _residues(order)]
    assert len(set(elements)) == order
    mapping = {element: index for index, element in enumerate(elements)}
    assert len(mapping) == order


# ===========================================================================
# 14. is_generator — признак порождающего элемента
# ===========================================================================

@pytest.mark.parametrize("order", ORDERS)
def test_one_is_not_a_generator(order):
    # единица имеет порядок 1 и не является порождающим элементом
    one = SchnorrGroupElement.one(order)
    assert one.is_generator is False
    assert not one.is_generator
    assert one.is_generator == (one.value != 1)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_every_non_identity_element_is_generator(order):
    # порядок группы прост -> любой неединичный элемент порождает группу
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        assert element.is_generator is (value != 1)
        assert element.is_generator == (element.value != 1)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_random_generator_result_is_generator(order):
    for _ in range(20):
        element = SchnorrGroupElement.random_generator(order)
        assert element.is_generator is True


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (3, 4), (5, 3), (5, 4), (5, 5), (5, 9),
    (11, 2), (11, 9), (11, 18), (23, 4), (23, 12),
])
def test_is_generator_returns_bool(make_element, order, value):
    result = make_element(order, value).is_generator
    assert isinstance(result, bool)
    assert result is True


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_is_generator_is_read_only(order):
    element = SchnorrGroupElement.one(order)
    with pytest.raises(AttributeError):
        element.is_generator = True


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_is_generator_of_product(order):
    # произведение порождающих элементов — порождающий элемент, КРОМЕ случая,
    # когда второй множитель является обратным к первому: тогда произведение —
    # единица группы (порядка 1), например 2 * 4 = 1 в группе порядка 3
    residues = [value for value in _residues(order) if value != 1]
    for v1 in residues:
        for v2 in residues:
            expected_value = (v1 * v2) % _q(order)
            product = (SchnorrGroupElement(order, v1) *
                       SchnorrGroupElement(order, v2))
            assert product.value == expected_value
            assert product.is_generator is (expected_value != 1)
            assert (product.value != 1) == (v2 != pow(v1, -1, _q(order)))


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_is_generator_consistent_with_group_order(order):
    # порядок неединичного элемента равен order, единицы — 1
    for value in _residues(order):
        element = SchnorrGroupElement(order, value)
        assert element.is_generator == (element.value != 1)
        if element.is_generator:
            assert (element ** order).value == 1
            assert element != SchnorrGroupElement.one(order)


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_is_generator_of_generator_powers(order):
    generator = SchnorrGroupElement.random_generator(order)
    powers = [generator ** exponent for exponent in range(1, order)]
    assert all(element.is_generator for element in powers)
    identity = generator ** order
    assert identity.is_generator is False


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_inverse_of_generator_is_generator(make_element, order, value):
    element = make_element(order, value)
    assert (~element).is_generator == element.is_generator


@pytest.mark.parametrize("order, value", [
    (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_is_generator_survives_arithmetic(make_element, order, value):
    element = make_element(order, value)
    combined = element * element ** 2 / element
    assert combined.is_generator is True
    assert combined.value == pow(value, 2, _q(order))


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 9), (11, 18), (23, 12),
])
def test_operations_return_new_objects(make_element, order, value):
    element = make_element(order, value)
    other = make_element(order, 1)
    assert element * other is not element
    assert element ** 2 is not element
    assert ~element is not element


# ===========================================================================
# 14. __pow__ с показателем F_pElement (exp.p == order)
# ===========================================================================

@pytest.mark.parametrize("order, value, exponent", [
    (2, 4, 1), (3, 2, 2), (5, 3, 3), (11, 2, 5), (23, 2, 10),
])
def test_pow_by_field_element_returns_result(
        make_element, order, value, exponent):
    element = make_element(order, value)
    result = element ** F_pElement(order, exponent)
    assert result == make_element(order, pow(value, exponent, _q(order)))
    assert result.order == order
    assert str(result) == (f"({pow(value, exponent, _q(order))} "
                           f"mod {_q(order)})")


@pytest.mark.parametrize("order, value, exponent", [
    (2, 4, 1), (3, 2, 2), (5, 3, 3), (11, 2, 5), (23, 2, 10),
])
def test_pow_by_field_element_of_matching_field_is_valid_exponent(
        make_element, order, value, exponent):
    # поле показателя должно совпадать с order (экспоненты из F_order)
    element = make_element(order, value)
    result = element ** F_pElement(order, exponent)
    assert result is not None
    assert result.order == order
    assert result.value == pow(value, exponent, _q(order))


@pytest.mark.parametrize("p, other_p, value", [
    (2, 3, 4), (3, 5, 2), (5, 2, 3), (11, 7, 2), (23, 5, 4),
])
def test_pow_by_field_element_of_other_field_raises_type_error(
        make_element, p, other_p, value):
    element = make_element(p, value)
    with pytest.raises(TypeError) as exc_info:
        element ** F_pElement(other_p, 2)
    assert (f"cannot raise ({value} mod {_q(p)}) to a power from field "
            f"F_{other_p}") in str(exc_info.value)


@pytest.mark.parametrize("p, other_p, value, exponent", [
    (2, 3, 4, 1), (3, 5, 2, 2), (5, 2, 3, 4), (11, 7, 2, 9), (23, 5, 4, 10),
])
def test_pow_by_field_element_error_message_mentions_value_and_field(
        make_element, p, other_p, value, exponent):
    element = make_element(p, value)
    with pytest.raises(TypeError) as exc_info:
        element ** F_pElement(other_p, exponent)
    assert (f"cannot raise ({value} mod {_q(p)}) to a power from field "
            f"F_{other_p}") in str(exc_info.value)


@pytest.mark.parametrize("order, value", [
    (2, 4), (3, 2), (5, 3), (11, 2), (23, 2),
])
def test_pow_by_field_element_zero_exponent(make_element, order, value):
    element = make_element(order, value)
    result = element ** F_pElement(order, 0)
    assert result == SchnorrGroupElement.one(order)
    assert result.order == order


@pytest.mark.parametrize("order", SMALL_ORDERS)
def test_random_generator_is_generator_of_group(order):
    # random_generator возвращает именно ПОРОЖДАЮЩИЙ элемент: его порядок
    # ровно order, то есть a**k != 1 при 1 <= k < order
    q = _q(order)
    element = SchnorrGroupElement.random_generator(order)
    for exponent in range(1, order):
        assert pow(element.value, exponent, q) != 1
    assert pow(element.value, order, q) == 1


@pytest.mark.parametrize("order1, order2", ORDER_PAIRS)
def test_operations_between_groups_are_rejected(make_element, order1, order2):
    first = make_element(order1, 1)
    second = make_element(order2, 1)
    for operation in (lambda: first * second, lambda: first / second):
        with pytest.raises(TypeError, match="different groups with orders"):
            operation()
    # сравнение элементов разных групп исключений не бросает — просто False
    assert (first == second) is False
    assert (first != second) is True
