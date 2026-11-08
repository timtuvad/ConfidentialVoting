"""Тесты модуля src/core/utils.py — утилиты работы с простыми числами.

Стиль написания — единый с остальными тестами проекта (tests/core/algebra/*,
tests/core/test_share.py, tests/core/test_environment.py и др.): docstring
модуля с перечнем проверяемого функционала, блок констант, вспомогательные
функции, секции с заголовками и параметризация pytest.

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * is_prime                        — проверка целого числа на простоту;
    * get_prime_greater_than          — первое по возрастанию простое число,
                                        strictly большее n;
    * get_special_prime_greater_than  — первое по возрастанию «безопасное»
                                        простое p > n, для которого 2p+1 тоже
                                        простое (порядки групп Шнорра);
    * интеграция                      — найденные «безопасные» простые годятся
                                        как order для SchnorrGroupElement и
                                        CryptographyEnvironment.random.

Особенности реализации, зафиксированные тестами:
    * is_prime возвращает False БЕЗ исключения для любого не-int (str, None,
      float, list, ...) — проверка через isinstance; в частности
      is_prime(2.0) is False, хотя 2.0 == 2;
    * bool — подкласс int: is_prime(True) == is_prime(1) == False,
      is_prime(False) == is_prime(0) == False;
    * is_prime: n < 2 -> False; из чётных простое только 2; нечётные
      отсеиваются пробными делениями на нечётные d до d*d <= n;
    * get_prime_greater_than: поиск начинается с max(floor(n)+1, 2) —
      наименьшего целого, строго большего n, поэтому для любого n < 2
      (включая отрицательные) результат — 2;
    * get_special_prime_greater_than: особые случаи — при n < 2 возвращается 2
      (2*2+1 = 5 простое), при n == 2 возвращается 3 (2*3+1 = 7 простое);
    * при n >= 3 перебираются только кандидаты вида 6k-1, начиная с
      k = floor((n+1)/6)+1 (наименьшее k, при котором 6k-1 > n):
      безопасное простое p > 3 всегда сравнимо с 5 по модулю 6, так как
      при p = 6k+1 число 2p+1 = 12k+3 делится на 3; поэтому результат
      при n >= 3 всегда даёт остаток 5 от деления на 6;
    * floor-формулы корректны и для дробных n: обе функции поддерживают
      float-аргументы, возвращая первое подходящее простое строго больше n
      (get_prime_greater_than(2.5) == 3, get_special_prime_greater_than(4.5)
      == 5);
    * явной валидации типов аргумента нет: str/None дают TypeError
      (из math.floor либо из сравнения `2 > n`).
"""

import pytest

from src import is_prime, get_prime_greater_than
from src.core.utils import get_special_prime_greater_than

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                59, 61, 67, 71, 73, 79, 83, 89, 97]

LARGE_PRIMES = [999983, 1000003, 10 ** 9 + 7, 10 ** 9 + 9, 2 ** 31 - 1]

SMALL_COMPOSITES = [4, 6, 8, 9, 10, 12, 14, 15, 21, 25, 27, 100, 1000]

LARGE_COMPOSITES = [
    589423,                    # 691 * 853
    954529,                    # 977 ** 2
    10 ** 9 + 8,               # чётное
    100003 * 100007,           # произведение двух больших простых
    3 * (10 ** 9 + 7),         # имеет малый делитель 3
]

NON_INT_VALUES = ["5", 2.0, 3.5, None, [3], (3,), {3}, object(), 2 + 0j]

NON_NUMERIC_VALUES = ["5", None, [1], (1,), object()]

# (p, q = 2p+1) — пары «безопасных» простых
SAFE_PRIME_PAIRS = [
    (2, 5), (3, 7), (5, 11), (11, 23), (23, 47), (29, 59), (41, 83),
    (53, 107), (83, 167), (89, 179), (113, 227), (131, 263), (173, 347),
    (179, 359), (191, 383),
]

SAFE_PRIMES = [p for p, _ in SAFE_PRIME_PAIRS]

# простые, но НЕ безопасные (2p+1 составное)
PRIME_NOT_SAFE = [7, 13, 17, 19, 31, 37, 43, 47, 59, 61, 67, 71, 73, 79, 97,
                  101, 103, 107, 109]

# Таблица get_prime_greater_than (целые n)
PRIME_GREATER_TABLE = [
    (-7, 2), (-1, 2), (0, 2), (1, 2),
    (2, 3), (3, 5), (4, 5), (5, 7), (6, 7),
    (7, 11), (8, 11), (9, 11), (10, 11),
    (11, 13), (13, 17), (17, 19), (19, 23), (23, 29), (24, 29),
    (100, 101), (101, 103), (1000, 1009),
    (10 ** 9, 10 ** 9 + 7),
]

# Таблица get_special_prime_greater_than (целые n)
SPECIAL_PRIME_TABLE = [
    (-10, 2), (-1, 2), (0, 2), (1, 2),
    (2, 3),
    (3, 5), (4, 5),
    (5, 11), (6, 11), (7, 11), (10, 11),
    (11, 23), (12, 23), (22, 23),
    (23, 29), (28, 29),
    (29, 41), (30, 41), (40, 41),
    (41, 53), (52, 53),
    (53, 83), (82, 83),
    (83, 89), (88, 89),
    (89, 113), (112, 113),
    (113, 131), (130, 131),
    (131, 173), (172, 173),
    (173, 179), (178, 179),
    (179, 191), (190, 191),
    (1000, 1013),
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _is_special(p):
    """p — «безопасное» простое: и p, и 2p+1 просты."""
    return is_prime(p) and is_prime(2 * p + 1)


# ===========================================================================
# 1. is_prime — простые числа
# ===========================================================================

@pytest.mark.parametrize("n", SMALL_PRIMES)
def test_is_prime_true_for_small_primes(n):
    assert is_prime(n) is True


@pytest.mark.parametrize("n", LARGE_PRIMES)
def test_is_prime_true_for_large_primes(n):
    assert is_prime(n) is True


def test_is_prime_finds_exactly_primes_below_100():
    assert {n for n in range(2, 100) if is_prime(n)} == set(SMALL_PRIMES)


def test_is_prime_finds_exactly_primes_between_100_and_200():
    expected = {101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199}
    assert {n for n in range(100, 200) if is_prime(n)} == expected


def test_is_prime_safe_primes_are_prime():
    # оба числа каждой «безопасной» пары просты
    for p, q in SAFE_PRIME_PAIRS:
        assert is_prime(p) is True
        assert is_prime(q) is True


# ===========================================================================
# 2. is_prime — составные числа
# ===========================================================================

@pytest.mark.parametrize("n", SMALL_COMPOSITES)
def test_is_prime_false_for_small_composites(n):
    assert is_prime(n) is False


@pytest.mark.parametrize("n", LARGE_COMPOSITES)
def test_is_prime_false_for_large_composites(n):
    assert is_prime(n) is False


@pytest.mark.parametrize("p", [3, 5, 7, 11, 13, 101, 997])
def test_is_prime_false_for_squares_of_primes(p):
    assert is_prime(p * p) is False


@pytest.mark.parametrize("a, b", [
    (3, 5), (7, 11), (13, 17), (101, 103), (31, 37), (691, 853),
])
def test_is_prime_false_for_products_of_two_primes(a, b):
    assert is_prime(a * b) is False


@pytest.mark.parametrize("p", PRIME_NOT_SAFE)
def test_is_prime_false_for_double_plus_one_of_unsafe_primes(p):
    # 2p+1 составное для «небезопасных» простых — но и сами p просты
    assert is_prime(p) is True
    assert is_prime(2 * p + 1) is False


# ===========================================================================
# 3. is_prime — граничные значения и типы аргумента
# ===========================================================================

@pytest.mark.parametrize("n", [-10 ** 6, -100, -7, -3, -2, -1, 0, 1])
def test_is_prime_false_for_less_than_two(n):
    assert is_prime(n) is False


def test_is_prime_two_is_prime():
    # единственное чётное простое: ветка `n % 2 == 0 -> return n == 2`
    assert is_prime(2) is True


def test_is_prime_other_even_numbers_are_composite():
    for even in range(-10, 200, 2):
        if even == 2:
            continue
        assert is_prime(even) is False


@pytest.mark.parametrize("other", NON_INT_VALUES)
def test_is_prime_false_for_non_int_without_exception(other):
    # нюанс: никакого TypeError — проверка isinstance возвращает False
    assert is_prime(other) is False


def test_is_prime_float_two_is_false():
    # нюанс: 2.0 == 2, но не является int
    assert is_prime(2.0) is False
    assert is_prime(3.0) is False
    assert is_prime(10 ** 9 + 7.0) is False


@pytest.mark.parametrize("flag, expected", [(True, False), (False, False)])
def test_is_prime_bool_arguments(flag, expected):
    # нюанс: bool — подкласс int, но True == 1 и False == 0 не просты
    assert is_prime(flag) is expected


# ===========================================================================
# 4. is_prime — общие свойства
# ===========================================================================

@pytest.mark.parametrize("n", [-5, 0, 1, 2, 3, 4, 17, 100, 10 ** 9 + 7])
def test_is_prime_returns_bool(n):
    result = is_prime(n)
    assert isinstance(result, bool)
    assert result is True or result is False


@pytest.mark.parametrize("n", SMALL_PRIMES[:12])
def test_is_prime_consistent_with_trial_division(n):
    # у простого нет делителей, кроме 1 и самого себя
    divisors = [d for d in range(2, n) if n % d == 0]
    assert divisors == []
    assert is_prime(n) is True


@pytest.mark.parametrize("n", [4, 6, 9, 25, 100, 954529])
def test_is_prime_composite_has_proper_divisor(n):
    divisors = [d for d in range(2, n) if n % d == 0]
    assert divisors
    assert is_prime(n) is False


@pytest.mark.parametrize("n", [2, 3, 17, 100, 589423])
def test_is_prime_is_idempotent(n):
    assert is_prime(n) == is_prime(n)


def test_is_prime_does_not_mutate_arguments_and_is_pure():
    values = [2, 3, 4, 17, 100]
    first = [is_prime(n) for n in values]
    second = [is_prime(n) for n in values]
    assert first == second
    assert values == [2, 3, 4, 17, 100]


# ===========================================================================
# 5. get_prime_greater_than — таблица значений
# ===========================================================================

@pytest.mark.parametrize("n, expected", PRIME_GREATER_TABLE)
def test_get_prime_greater_than_table(n, expected):
    assert get_prime_greater_than(n) == expected


@pytest.mark.parametrize("p, next_p", [
    (2, 3), (3, 5), (5, 7), (7, 11), (11, 13), (13, 17), (97, 101),
])
def test_get_prime_greater_than_of_prime_is_next_prime(p, next_p):
    # для простого p результат — СЛЕДУЮЩЕЕ простое (строго больше)
    assert get_prime_greater_than(p) == next_p


@pytest.mark.parametrize("n", [0, 1, 5, 17, 100, 10 ** 6])
def test_get_prime_greater_than_returns_int(n):
    result = get_prime_greater_than(n)
    assert isinstance(result, int)


def test_get_prime_greater_than_is_deterministic():
    for n in (-5, 0, 7, 100):
        assert get_prime_greater_than(n) == get_prime_greater_than(n)


def test_get_prime_greater_than_covers_small_range():
    # последовательные вызовы перечисляют простые подряд
    p = 1
    collected = []
    for _ in range(10):
        p = get_prime_greater_than(p)
        collected.append(p)
    assert collected == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


# ===========================================================================
# 6. get_prime_greater_than — свойства
# ===========================================================================

@pytest.mark.parametrize("n", [-10 ** 6, -100, -1, 0, 1, 2, 5, 13, 90,
                               1000, 10 ** 6 + 1])
def test_get_prime_greater_than_returns_prime_strictly_above_n(n):
    p = get_prime_greater_than(n)
    assert p > n
    assert is_prime(p) is True


@pytest.mark.parametrize("n", [2, 5, 13, 90, 1000, 10 ** 4])
def test_get_prime_greater_than_is_minimal(n):
    # между n и результатом нет других простых
    p = get_prime_greater_than(n)
    for candidate in range(max(n + 1, 2), p):
        assert is_prime(candidate) is False


@pytest.mark.parametrize("n", [-10 ** 6, -100, -1, 0, 1])
def test_get_prime_greater_than_below_two_returns_two(n):
    # нюанс: старт поиска max(floor(n)+1, 2) отсекает всё меньше 2
    assert get_prime_greater_than(n) == 2


@pytest.mark.parametrize("n1, n2", [(1, 5), (5, 6), (10, 100), (23, 24),
                                    (100, 1000)])
def test_get_prime_greater_than_is_monotone(n1, n2):
    assert get_prime_greater_than(n1) <= get_prime_greater_than(n2)


def test_get_prime_greater_than_agrees_with_is_prime():
    # результат всегда проходит is_prime и не имеет простых между n и собой
    for n in range(-5, 60):
        p = get_prime_greater_than(n)
        assert is_prime(p)
        assert all(not is_prime(c) for c in range(max(n + 1, 2), p))


# ===========================================================================
# 7. get_prime_greater_than — нюансы (float, bool, чужие типы)
# ===========================================================================

@pytest.mark.parametrize("n, expected", [
    (-0.5, 2), (0.5, 2), (1.2, 2), (2.0, 3), (2.5, 3),
    (3.0, 5), (4.0, 5), (10.1, 11), (100.9, 101),
])
def test_get_prime_greater_than_accepts_floats(n, expected):
    # floor(n)+1 — наименьшее целое, строго большее n: дробные аргументы
    # поддерживаются, результат — первое простое строго больше n
    assert get_prime_greater_than(n) == expected


@pytest.mark.parametrize("flag", [True, False])
def test_get_prime_greater_than_accepts_bool(flag):
    # bool — подкласс int: True == 1 -> 2, False == 0 -> 2
    assert get_prime_greater_than(flag) == 2


@pytest.mark.parametrize("other", NON_NUMERIC_VALUES)
def test_get_prime_greater_than_non_numeric_raises_type_error(other):
    # явной валидации нет: TypeError приходит из math.floor
    with pytest.raises(TypeError):
        get_prime_greater_than(other)


# ===========================================================================
# 8. get_special_prime_greater_than — таблица значений
# ===========================================================================

@pytest.mark.parametrize("n, expected", SPECIAL_PRIME_TABLE)
def test_get_special_prime_greater_than_table(n, expected):
    assert get_special_prime_greater_than(n) == expected


@pytest.mark.parametrize("p, q", SAFE_PRIME_PAIRS)
def test_get_special_prime_greater_than_of_previous_gives_safe_prime(p, q):
    # f(p-1) == p и 2p+1 == q простое
    assert get_special_prime_greater_than(p - 1) == p
    assert is_prime(q) is True
    assert q == 2 * p + 1


def test_get_special_prime_greater_than_sequence():
    # цепочка вызовов перечисляет «безопасные» простые подряд
    expected = [2, 3, 5, 11, 23, 29, 41, 53, 83, 89, 113, 131, 173, 179]
    actual = []
    p = 1
    for _ in range(len(expected)):
        p = get_special_prime_greater_than(p)
        actual.append(p)
    assert actual == expected


@pytest.mark.parametrize("n", [0, 5, 100, 1000])
def test_get_special_prime_greater_than_returns_int(n):
    assert isinstance(get_special_prime_greater_than(n), int)


def test_get_special_prime_greater_than_is_deterministic():
    for n in (-5, 0, 2, 7, 100):
        assert (get_special_prime_greater_than(n)
                == get_special_prime_greater_than(n))


# ===========================================================================
# 9. get_special_prime_greater_than — свойства
# ===========================================================================

@pytest.mark.parametrize("n", [-10 ** 6, -10, 0, 1, 2, 3, 10, 50, 100, 1000,
                               10 ** 4, 10 ** 6])
def test_special_prime_is_safe_and_strictly_above_n(n):
    p = get_special_prime_greater_than(n)
    assert p > n
    assert is_prime(p) is True
    assert is_prime(2 * p + 1) is True


@pytest.mark.parametrize("n", [3, 10, 50, 100, 1000, 10 ** 4])
def test_special_prime_is_minimal(n):
    # между n и результатом нет других «безопасных» простых
    p = get_special_prime_greater_than(n)
    for candidate in range(max(n + 1, 2), p):
        assert not _is_special(candidate)


@pytest.mark.parametrize("n", [3, 4, 10, 50, 100, 1000, 10 ** 4])
def test_special_prime_congruent_five_mod_six(n):
    # нюанс оптимизации: при n >= 3 ответ всегда вида 6k-1
    p = get_special_prime_greater_than(n)
    assert p % 6 == 5


@pytest.mark.parametrize("n", [5, 20, 100, 1000])
def test_special_prime_not_less_than_ordinary_prime(n):
    # «безопасное» простое — тоже простое, поэтому special >= ordinary
    assert (get_special_prime_greater_than(n)
            >= get_prime_greater_than(n))


@pytest.mark.parametrize("n1, n2", [(1, 5), (5, 6), (10, 100), (23, 24),
                                    (100, 1000)])
def test_special_prime_is_monotone(n1, n2):
    assert (get_special_prime_greater_than(n1)
            <= get_special_prime_greater_than(n2))


def test_all_safe_primes_below_200_are_found():
    expected = {2, 3, 5, 11, 23, 29, 41, 53, 83, 89, 113, 131, 173, 179, 191}
    found = {n for n in range(2, 200) if _is_special(n)}
    assert found == expected


# ===========================================================================
# 10. get_special_prime_greater_than — нюансы
# ===========================================================================

@pytest.mark.parametrize("n", [-10 ** 6, -5, -1, 0, 1])
def test_special_prime_below_two_returns_two(n):
    # особая ветка `if 2 > n`: 2 — «безопасное» простое (2*2+1 = 5)
    assert get_special_prime_greater_than(n) == 2
    assert _is_special(2)


def test_special_prime_of_two_is_three():
    # особая ветка `if 3 > n`: 3 — «безопасное» простое (2*3+1 = 7)
    assert get_special_prime_greater_than(2) == 3
    assert _is_special(3)


@pytest.mark.parametrize("n, expected", [
    (1.5, 2), (2.5, 3), (3.5, 5), (4.5, 5), (10.5, 11), (52.5, 53),
])
def test_special_prime_accepts_floats(n, expected):
    # k = floor((n+1)/6)+1 — наименьшее k с 6k-1 > n: дробные аргументы
    # поддерживаются, результат — первое «безопасное» простое строго больше n
    assert get_special_prime_greater_than(n) == expected


@pytest.mark.parametrize("flag", [True, False])
def test_special_prime_accepts_bool(flag):
    # bool — подкласс int: True == 1 -> 2, False == 0 -> 2
    assert get_special_prime_greater_than(flag) == 2


@pytest.mark.parametrize("other", NON_NUMERIC_VALUES)
def test_special_prime_non_numeric_raises_type_error(other):
    # явной валидации нет: TypeError приходит из сравнения `2 > n`
    with pytest.raises(TypeError):
        get_special_prime_greater_than(other)


# ===========================================================================
# 11. Интеграция: «безопасные» простые как порядки групп Шнорра
# ===========================================================================

@pytest.mark.parametrize("n", [0, 2, 10, 100, 1000])
def test_special_prime_is_valid_schnorr_group_order(n):
    from src.core.algebra.schnorr_group import SchnorrGroupElement
    order = get_special_prime_greater_than(n)
    # 4 = 2^2 — квадратичный вычет mod q и не единица (q >= 5)
    element = SchnorrGroupElement(order, 4)
    assert element.order == order
    assert element.is_generator is True


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_special_prime_accepts_random_generator(n):
    from src.core.algebra.schnorr_group import SchnorrGroupElement
    order = get_special_prime_greater_than(n)
    generator = SchnorrGroupElement.random_generator(order)
    assert generator.order == order
    assert generator.is_generator is True


@pytest.mark.parametrize("n", [10, 100])
def test_special_prime_usable_in_random_environment(n):
    from src.core.environment import CryptographyEnvironment
    order = get_special_prime_greater_than(n)
    env = CryptographyEnvironment.random(order, 3)
    assert env.p == order
    assert env.g0.is_generator and env.h0.is_generator
    assert env.g0 != env.h0


def test_special_prime_of_small_n_gives_smallest_groups():
    from src.core.algebra.schnorr_group import SchnorrGroupElement
    assert SchnorrGroupElement(get_special_prime_greater_than(0), 4).order == 2
    assert SchnorrGroupElement(get_special_prime_greater_than(2), 4).order == 3
    assert SchnorrGroupElement(get_special_prime_greater_than(3), 4).order == 5


@pytest.mark.parametrize("n", PRIME_NOT_SAFE[:8])
def test_unsafe_primes_are_not_valid_group_orders(n):
    # для «небезопасных» простых группа Шнорра порядка n недопустима:
    # q = 2n+1 составное, конструктор элемента бросает ValueError
    from src.core.algebra.schnorr_group import SchnorrGroupElement
    assert _is_special(n) is False
    with pytest.raises(ValueError):
        SchnorrGroupElement(n, 4)
