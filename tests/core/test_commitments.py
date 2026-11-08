"""Тесты модуля src/core/commitments.py — класс PolynomialCommitments.

Стиль написания — как в tests/core/test_utils.py: фикстуры + параметризация pytest.
Проверки выполняются через публичный API (commitment_vector, __len__, make(),
==, !=, hash(), +, -, унарный минус, len()) — без обращения к приватным полям.

PolynomialCommitments — вектор обязательств Педерсена к коэффициентам
многочлена: c_i = g0**f[i] * h0**blinding_polynomial[i] (i = 0..threshold-1).
Сложение обязательств — поэлементное УМНОЖЕНИЕ групповых элементов
(гомоморфизм: обязательства к f+g равны сумме обязательств к f и g),
унарный минус — поэлементное ОБРАЩЕНИЕ (~c).

Проверяемый функционал (на каждый элемент — не менее 5 тестов):
    * __init__          — непустой вектор элементов ОДНОЙ группы Шнорра;
                          ValueError для пустого вектора и для смешанных групп;
                          TypeError для не-SchnorrGroupElement (isinstance-
                          проверка выполняется до сравнения групп);
    * commitment_vector — кортеж обязательств, только для чтения;
    * __len__           — длина вектора;
    * make              — статический конструктор обязательств к многочлену
                          в данном окружении; TypeError для многочленов чужого
                          поля или степени >= threshold; гомоморфность;
    * __eq__, __ne__, __hash__ — сравнение по вектору обязательств;
    * __pos__, __neg__  — унарные операции (обращение каждого элемента);
    * __add__           — поэлементное умножение; ValueError для разной длины;
    * __sub__           — self + (-other); ValueError для разной длины
                          (тот же тип исключения, что и у __add__);
    * прочие нюансы     — неизменяемость операндов.

Особенности реализации, зафиксированные тестами:
    * __init__ принимает ЛЮБОЙ итерируемый объект (список, кортеж, генератор)
      и преобразует его в кортеж;
    * порядок проверок в __init__: (1) непустота — ValueError "commitment
      vector cannot be empty"; (2) isinstance каждого элемента — TypeError
      "all commitments must be instances of SchnorrGroupElement";
      (3) одинаковость групп по .order — ValueError "all commitments must be
      from the same group";
    * длина вектора при создании через make ВСЕГДА равна environment.threshold
      (недостающие коэффициенты берутся как нули через __getitem__);
    * __add__ НЕ проверяет группу элементов: при равных длинах, но разных
      группах TypeError бросает SchnorrGroupElement.__mul__
      ("cannot * with two elements in different groups with orders ...");
    * environment в make НЕ валидируется: не-окружение даёт AttributeError;
    * _new не валидирует вектор (используется внутри make/__neg__/__add__).
"""

import pytest

from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial
from src.core.algebra.schnorr_group import SchnorrGroupElement
from src.core.commitments import PolynomialCommitments
from src.core.environment import CryptographyEnvironment

# ---------------------------------------------------------------------------
# Общие наборы данных
# ---------------------------------------------------------------------------

# order — простые, для которых q = 2*order + 1 тоже простое
ORDERS = [3, 5, 11]

# Пары значений порождающих элементов: order -> (g0_value, h0_value)
GENERATORS = {3: (2, 4), 5: (3, 4), 11: (2, 3)}

# Первый генератор каждой группы (для тестов смешанных групп)
FIRST_GENERATOR_VALUE = {3: 2, 5: 3, 11: 2}

ORDER_PAIRS = [(3, 5), (5, 11), (3, 11)]

NON_COMMITMENT_VALUES = [0, 5, "c", None, 3.5, (5,), [5], object()]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _poly(p, coeffs):
    """Многочлен над F_p из списка целых коэффициентов (от свободного члена)."""
    return F_pPolynomial(p, coefficients=[F_pElement(p, c) for c in coeffs])


def _values(commitments):
    """Список значений (value) элементов вектора обязательств."""
    return tuple(c.value for c in commitments.commitment_vector)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def make_environment():
    """Фабрика окружений: make_environment(order, threshold)."""
    def _make(order=5, threshold=3):
        g, h = GENERATORS[order]
        return CryptographyEnvironment(SchnorrGroupElement(order, g),
                                       SchnorrGroupElement(order, h),
                                       threshold)
    return _make


@pytest.fixture
def environment(make_environment):
    """Окружение порядка 5 (q = 11): g0 = 3, h0 = 4, threshold = 3."""
    return make_environment(5, 3)


@pytest.fixture
def make_poly():
    """Фабрика многочленов из списка целых коэффициентов (от свободного члена)."""
    def _make(p, coefficients):
        return F_pPolynomial(
            p, coefficients=[F_pElement(p, c) for c in coefficients])
    return _make


@pytest.fixture
def make_commitments():
    """Фабрика обязательств: make_commitments(order, values)."""
    def _make(order, values):
        vector = tuple(SchnorrGroupElement(order, v) for v in values)
        return PolynomialCommitments(vector)
    return _make


@pytest.fixture
def commitments5(make_commitments):
    """Вектор обязательств (3, 4, 5) в группе порядка 5 (q = 11)."""
    return make_commitments(5, [3, 4, 5])


# ===========================================================================
# 1. __init__ — валидные векторы элементов группы
# ===========================================================================

@pytest.mark.parametrize("order, values", [
    (3, [2]),
    (3, [2, 4]),
    (5, [3, 4, 5]),
    (5, [1, 1, 1]),
    (5, [9, 3, 4, 5, 1]),
    (11, [2, 3, 4]),
    (11, [1]),
])
def test_init_accepts_vectors_of_group_elements(make_commitments, order, values):
    commitments = make_commitments(order, values)
    assert isinstance(commitments, PolynomialCommitments)
    assert len(commitments) == len(values)


def test_init_accepts_list_of_elements():
    vector = [SchnorrGroupElement(5, 3), SchnorrGroupElement(5, 4)]
    commitments = PolynomialCommitments(vector)
    assert len(commitments) == 2


def test_init_accepts_tuple_of_elements():
    vector = (SchnorrGroupElement(5, 3), SchnorrGroupElement(5, 4))
    commitments = PolynomialCommitments(vector)
    assert commitments.commitment_vector == vector


def test_init_accepts_generator_expression():
    commitments = PolynomialCommitments(SchnorrGroupElement(5, v)
                                        for v in [3, 4, 5])
    assert len(commitments) == 3


def test_init_stores_values_unchanged(make_commitments):
    commitments = make_commitments(5, [3, 4, 5])
    for stored, original in zip(commitments.commitment_vector, [3, 4, 5]):
        assert stored == SchnorrGroupElement(5, original)
        assert stored.value == original


def test_init_single_element_vector(make_commitments):
    commitments = make_commitments(11, [2])
    assert len(commitments) == 1
    assert commitments.commitment_vector[0] == SchnorrGroupElement(11, 2)


def test_init_accepts_unit_elements(make_commitments):
    # единица группы — корректный элемент (is_generator не проверяется)
    commitments = make_commitments(5, [1, 1, 1])
    assert len(commitments) == 3
    for element in commitments.commitment_vector:
        assert element == SchnorrGroupElement.one(5)


# ===========================================================================
# 2. __init__ — исключения
# ===========================================================================

@pytest.mark.parametrize("empty", [[], (), iter([])])
def test_init_empty_vector_raises_value_error(empty):
    with pytest.raises(ValueError, match="commitment vector cannot be empty"):
        PolynomialCommitments(empty)


def test_init_empty_generator_raises_value_error():
    with pytest.raises(ValueError, match="commitment vector cannot be empty"):
        PolynomialCommitments(SchnorrGroupElement(5, v) for v in [])


def test_init_empty_vector_error_is_value_error_not_type_error():
    with pytest.raises(ValueError):
        PolynomialCommitments([])


def test_init_empty_message_is_exact():
    with pytest.raises(ValueError) as exc_info:
        PolynomialCommitments(())
    assert str(exc_info.value) == "commitment vector cannot be empty"


@pytest.mark.parametrize("vector", [
    [1, 2],
    ["a"],
    [None],
    [3.5],
    [object()],
    [F_pElement(5, 1)],
    [F_pElement(5, 1), F_pElement(5, 2)],
    [(1,), [2]],
])
def test_init_non_group_elements_raise_type_error(vector):
    with pytest.raises(
            TypeError,
            match="all commitments must be instances of SchnorrGroupElement"):
        PolynomialCommitments(vector)


def test_init_type_error_message_is_exact():
    with pytest.raises(TypeError) as exc_info:
        PolynomialCommitments([1])
    assert str(exc_info.value) == ("all commitments must be instances of "
                                   "SchnorrGroupElement")


def test_init_mixed_vector_raises_type_error():
    # даже один чужеродный элемент рядом с корректными — TypeError
    with pytest.raises(TypeError,
                       match="all commitments must be instances"):
        PolynomialCommitments([SchnorrGroupElement(5, 3), F_pElement(5, 1)])


def test_init_type_check_precedes_group_check():
    # isinstance проверяется раньше сравнения order:
    # [SGE(order 3), int] даёт TypeError, а не ValueError/AttributeError
    with pytest.raises(TypeError,
                       match="all commitments must be instances"):
        PolynomialCommitments([SchnorrGroupElement(3, 2), 5])


def test_init_empty_check_precedes_type_check():
    # пустой вектор — ValueError про пустоту (проверяется первым)
    with pytest.raises(ValueError, match="cannot be empty"):
        PolynomialCommitments([1, 2][:0])


@pytest.mark.parametrize("order1, order2", ORDER_PAIRS)
def test_init_elements_of_different_groups_raise_value_error(order1, order2):
    vector = [SchnorrGroupElement(order1, FIRST_GENERATOR_VALUE[order1]),
              SchnorrGroupElement(order2, FIRST_GENERATOR_VALUE[order2])]
    with pytest.raises(ValueError,
                       match="all commitments must be from the same group"):
        PolynomialCommitments(vector)


def test_init_different_groups_message_is_exact():
    vector = [SchnorrGroupElement(3, 2), SchnorrGroupElement(5, 3)]
    with pytest.raises(ValueError) as exc_info:
        PolynomialCommitments(vector)
    assert str(exc_info.value) == "all commitments must be from the same group"


# ===========================================================================
# 3. commitment_vector и __len__
# ===========================================================================

@pytest.mark.parametrize("order, values", [
    (3, [2, 4]), (5, [3, 4, 5]), (5, [1]), (11, [2, 3, 4, 6]),
])
def test_commitment_vector_returns_tuple_of_elements(make_commitments,
                                                     order, values):
    commitments = make_commitments(order, values)
    vector = commitments.commitment_vector
    assert isinstance(vector, tuple)
    assert len(vector) == len(values)
    for element in vector:
        assert isinstance(element, SchnorrGroupElement)


def test_commitment_vector_is_read_only(commitments5):
    with pytest.raises(AttributeError):
        commitments5.commitment_vector = ()


def test_commitment_vector_is_stable(commitments5):
    assert commitments5.commitment_vector == commitments5.commitment_vector


def test_commitment_vector_is_immutable_tuple(commitments5):
    with pytest.raises(TypeError):
        commitments5.commitment_vector[0] = SchnorrGroupElement.one(5)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_len_equals_number_of_commitments(make_commitments, n):
    values = [3, 4, 5, 9, 1][:n]
    commitments = make_commitments(5, values)
    assert len(commitments) == n
    assert len(commitments) == len(commitments.commitment_vector)


def test_len_of_single_element_vector(make_commitments):
    assert len(make_commitments(3, [2])) == 1


def test_len_is_int(commitments5):
    assert isinstance(len(commitments5), int)


# ===========================================================================
# 4. make(f, blinding_polynomial, environment)
#    Группа порядка 5 (q = 11), g0 = 3, h0 = 4, threshold = 3.
# ===========================================================================

@pytest.mark.parametrize("f_coeffs, b_coeffs", [
    ([1], [1]),
    ([1, 2], [3, 4]),
    ([0, 0, 1], [1, 1, 1]),
    ([4], [0]),
    ([1, 2, 3], [4, 3, 2]),
])
def test_make_returns_commitments_of_threshold_length(environment, make_poly,
                                                      f_coeffs, b_coeffs):
    commitments = PolynomialCommitments.make(make_poly(5, f_coeffs),
                                             make_poly(5, b_coeffs),
                                             environment)
    assert isinstance(commitments, PolynomialCommitments)
    assert len(commitments) == environment.threshold


def test_make_coefficients_match_formula(environment, make_poly):
    # c_i == g0**f[i] * h0**b[i]
    f = make_poly(5, [1, 2, 3])
    b = make_poly(5, [4, 3, 2])
    commitments = PolynomialCommitments.make(f, b, environment)
    for i in range(environment.threshold):
        expected = environment.g0 ** f[i] * environment.h0 ** b[i]
        assert commitments.commitment_vector[i] == expected


def test_make_full_vector_matches_hand_computed(environment, make_poly):
    # f = 3 + x, b = 2 + 4x над F_5; q = 11, g0 = 3, h0 = 4:
    # c0 = 3^3 * 4^2 = 27*16 = 432 = 3 (mod 11)
    # c1 = 3^1 * 4^4 = 3*256 = 768 = 9 (mod 11)
    # c2 = 3^0 * 4^0 = 1
    commitments = PolynomialCommitments.make(make_poly(5, [3, 1]),
                                             make_poly(5, [2, 4]),
                                             environment)
    assert _values(commitments) == (3, 9, 1)


def test_make_of_zero_polynomials_gives_unit_vector(environment):
    commitments = PolynomialCommitments.make(F_pPolynomial(5),
                                             F_pPolynomial(5),
                                             environment)
    for element in commitments.commitment_vector:
        assert element == SchnorrGroupElement.one(5)
        assert element.is_generator is False


def test_make_with_zero_blinding_commits_only_to_f(environment, make_poly):
    f = make_poly(5, [2, 3])
    commitments = PolynomialCommitments.make(f, F_pPolynomial(5), environment)
    for i in range(environment.threshold):
        assert commitments.commitment_vector[i] == environment.g0 ** f[i]


def test_make_first_commitment_hides_secret(environment, make_poly):
    # c_0 = g0^secret * h0^b_0
    secret = F_pElement(5, 3)
    blinding = F_pElement(5, 2)
    commitments = PolynomialCommitments.make(make_poly(5, [3, 1]),
                                             make_poly(5, [2, 4]),
                                             environment)
    assert commitments.commitment_vector[0] == (environment.g0 ** secret
                                                * environment.h0 ** blinding)


def test_make_is_homomorphic_over_addition(environment, make_poly):
    f1, b1 = make_poly(5, [1, 2]), make_poly(5, [3, 4])
    f2, b2 = make_poly(5, [2, 1]), make_poly(5, [1, 1])
    left = PolynomialCommitments.make(f1 + f2, b1 + b2, environment)
    right = (PolynomialCommitments.make(f1, b1, environment)
             + PolynomialCommitments.make(f2, b2, environment))
    assert left == right


def test_make_is_homomorphic_over_negation(environment, make_poly):
    f, b = make_poly(5, [1, 2]), make_poly(5, [3, 4])
    assert (PolynomialCommitments.make(-f, -b, environment)
            == -PolynomialCommitments.make(f, b, environment))


def test_make_is_homomorphic_over_subtraction(environment, make_poly):
    f1, b1 = make_poly(5, [1, 2, 0]), make_poly(5, [3, 4])
    f2, b2 = make_poly(5, [2, 1]), make_poly(5, [1, 1])
    left = PolynomialCommitments.make(f1 - f2, b1 - b2, environment)
    right = (PolynomialCommitments.make(f1, b1, environment)
             - PolynomialCommitments.make(f2, b2, environment))
    assert left == right


def test_make_with_f_of_other_field_raises_type_error(environment, make_poly):
    with pytest.raises(TypeError,
                       match="f must be a correct polynomial from the "
                             "perspective of the environment"):
        PolynomialCommitments.make(make_poly(7, [1]), make_poly(5, [1]),
                                   environment)


def test_make_with_blinding_of_other_field_raises_type_error(environment,
                                                             make_poly):
    with pytest.raises(TypeError,
                       match="blinding_polynomial must be a correct "
                             "polynomial from the perspective of the "
                             "environment"):
        PolynomialCommitments.make(make_poly(5, [1]), make_poly(7, [1]),
                                   environment)


def test_make_with_too_high_degree_of_f_raises_type_error(environment,
                                                          make_poly):
    # threshold == 3: степень должна быть МЕНЬШЕ 3
    with pytest.raises(TypeError, match="f must be a correct polynomial"):
        PolynomialCommitments.make(make_poly(5, [1, 2, 3, 4]),
                                   make_poly(5, [1]), environment)


def test_make_with_too_high_degree_of_blinding_raises_type_error(
        environment, make_poly):
    with pytest.raises(TypeError, match="blinding_polynomial must be"):
        PolynomialCommitments.make(make_poly(5, [1]),
                                   make_poly(5, [1, 2, 3, 4]), environment)


def test_make_checks_f_before_blinding(environment, make_poly):
    # оба многочлена некорректны — сообщение про f (проверяется первым)
    with pytest.raises(TypeError) as exc_info:
        PolynomialCommitments.make(make_poly(7, [1]), make_poly(7, [1]),
                                   environment)
    assert str(exc_info.value).startswith("f must be")


@pytest.mark.parametrize("other", [5, None, "poly", 3.5, [1],
                                   F_pElement(5, 1),
                                   SchnorrGroupElement(5, 3)])
def test_make_with_non_polynomial_f_raises_type_error(environment, other):
    with pytest.raises(TypeError, match="f must be a correct polynomial"):
        PolynomialCommitments.make(other, F_pPolynomial(5), environment)


@pytest.mark.parametrize("other", [5, None, "poly", 3.5, [1],
                                   F_pElement(5, 1)])
def test_make_with_non_polynomial_blinding_raises_type_error(environment,
                                                             other):
    with pytest.raises(TypeError, match="blinding_polynomial must be"):
        PolynomialCommitments.make(F_pPolynomial(5), other, environment)


def test_make_with_threshold_one_uses_constant_terms_only(make_environment,
                                                          make_poly):
    env1 = make_environment(5, 1)
    commitments = PolynomialCommitments.make(make_poly(5, [2]),
                                             make_poly(5, [3]), env1)
    # c0 = 3^2 * 4^3 = 9 * 64 = 9 * 9 = 81 = 4 (mod 11)
    assert len(commitments) == 1
    assert commitments.commitment_vector[0].value == 4
    # многочлен степени >= 1 уже не проходит contains_F_pPolynomial
    with pytest.raises(TypeError, match="f must be a correct polynomial"):
        PolynomialCommitments.make(make_poly(5, [2, 1]), make_poly(5, [3]),
                                   env1)


def test_make_does_not_mutate_polynomials(environment, make_poly):
    f = make_poly(5, [1, 2])
    b = make_poly(5, [3])
    PolynomialCommitments.make(f, b, environment)
    assert f == make_poly(5, [1, 2])
    assert b == make_poly(5, [3])


def test_make_is_deterministic(environment, make_poly):
    f = make_poly(5, [1, 2])
    b = make_poly(5, [3, 4])
    first = PolynomialCommitments.make(f, b, environment)
    second = PolynomialCommitments.make(f, b, environment)
    assert first == second
    assert first.commitment_vector == second.commitment_vector


def test_make_returns_new_object_each_call(environment, make_poly):
    f = make_poly(5, [1])
    b = make_poly(5, [2])
    first = PolynomialCommitments.make(f, b, environment)
    second = PolynomialCommitments.make(f, b, environment)
    assert first is not second
    assert first == second


def test_make_with_non_environment_raises_attribute_error(make_poly):
    # нюанс: environment НЕ валидируется — обращение к чужому объекту
    # даёт AttributeError
    with pytest.raises(AttributeError):
        PolynomialCommitments.make(make_poly(5, [1]), make_poly(5, [1]), None)


# ===========================================================================
# 5. __eq__ / __ne__ / __hash__
# ===========================================================================

def test_eq_is_reflexive(commitments5):
    assert commitments5 == commitments5
    assert not (commitments5 != commitments5)


def test_eq_for_equal_vectors(make_commitments):
    assert make_commitments(5, [3, 4, 5]) == make_commitments(5, [3, 4, 5])


def test_eq_false_for_different_values(make_commitments):
    assert make_commitments(5, [3, 4, 5]) != make_commitments(5, [3, 4, 9])
    assert make_commitments(5, [1, 4, 5]) != make_commitments(5, [3, 4, 5])


def test_eq_false_for_different_lengths(make_commitments):
    assert make_commitments(5, [3, 4]) != make_commitments(5, [3, 4, 5])


def test_eq_false_for_different_groups(make_commitments):
    # одинаковые значения 1, но разные группы (order 3 и order 5)
    # значения 2 для примера брать нельзя, так как это невычет по модулю 5
    assert make_commitments(3, [1]) != make_commitments(5, [1])


@pytest.mark.parametrize("other", NON_COMMITMENT_VALUES
                         + [F_pElement(5, 1), SchnorrGroupElement(5, 3)])
def test_eq_with_non_commitments_is_false(commitments5, other):
    assert (commitments5 == other) is False
    assert (commitments5 != other) is True


@pytest.mark.parametrize("other", NON_COMMITMENT_VALUES[:6])
def test_eq_reflected_with_non_commitments_is_false(commitments5, other):
    assert (other == commitments5) is False
    assert (other != commitments5) is True


def test_ne_is_negation_of_eq(make_commitments):
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [3, 5])
    assert first != second
    assert not (first == second)


def test_hash_is_consistent_with_eq(make_commitments):
    first = make_commitments(5, [3, 4, 5])
    second = make_commitments(5, [3, 4, 5])
    assert first == second
    assert hash(first) == hash(second)
    assert isinstance(hash(first), int)


def test_hash_equals_hash_of_vector(commitments5):
    assert hash(commitments5) == hash(commitments5.commitment_vector)


def test_commitments_can_be_dict_key(make_commitments):
    commitments = make_commitments(5, [3, 4])
    mapping = {commitments: "secret"}
    assert mapping[make_commitments(5, [3, 4])] == "secret"


def test_distinct_commitments_coexist_in_set(make_commitments):
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [4, 3])
    third = make_commitments(5, [3, 4, 5])
    assert len({first, second, third, make_commitments(5, [3, 4])}) == 3


def test_eq_of_make_results(environment, make_poly):
    f = make_poly(5, [1, 2])
    b = make_poly(5, [3])
    first = PolynomialCommitments.make(f, b, environment)
    second = PolynomialCommitments.make(f, b, environment)
    third = PolynomialCommitments.make(f, make_poly(5, [4]), environment)
    assert first == second
    assert first != third


# ===========================================================================
# 6. __pos__ / __neg__
# ===========================================================================

def test_pos_returns_same_object(commitments5):
    assert (+commitments5) is commitments5


def test_pos_preserves_commitments(commitments5):
    assert +commitments5 == commitments5


def test_neg_inverts_each_commitment(commitments5):
    negated = -commitments5
    for original, inverse in zip(commitments5.commitment_vector,
                                 negated.commitment_vector):
        assert original * inverse == SchnorrGroupElement.one(5)


def test_neg_values_hand_computed(make_commitments):
    # mod 11: ~3 = 4 (3*4=12=1), ~4 = 3, ~5 = 9 (5*9=45=1)
    commitments = make_commitments(5, [3, 4, 5])
    assert _values(-commitments) == (4, 3, 9)


def test_double_negation_restores(commitments5):
    assert -(-commitments5) == commitments5


def test_neg_returns_new_object(make_commitments):
    commitments = make_commitments(5, [3, 4, 5])
    negated = -commitments
    assert negated is not commitments
    assert commitments == make_commitments(5, [3, 4, 5])


def test_neg_of_unit_vector_is_unit_vector(make_commitments):
    units = make_commitments(5, [1, 1, 1])
    assert -units == units


def test_neg_preserves_length_and_group(commitments5):
    negated = -commitments5
    assert len(negated) == len(commitments5)
    for element in negated.commitment_vector:
        assert element.order == 5


# ===========================================================================
# 7. __add__ — поэлементное умножение обязательств
# ===========================================================================

def test_add_multiplies_commitments_elementwise(make_commitments):
    # mod 11: 3*5 = 15 = 4; 4*9 = 36 = 3
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [5, 9])
    assert _values(first + second) == (4, 3)


def test_add_is_commutative(make_commitments):
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [5, 9])
    assert first + second == second + first


def test_add_is_associative(make_commitments):
    # mod 11: (3*5)*5 = 4*5 = 9; 3*(5*5) = 3*3 = 9
    a = make_commitments(5, [3])
    b = make_commitments(5, [5])
    c = make_commitments(5, [5])
    assert (a + b) + c == a + (b + c)
    assert _values((a + b) + c) == (9,)


def test_add_with_unit_vector_is_identity(make_commitments):
    commitments = make_commitments(5, [3, 4])
    units = make_commitments(5, [1, 1])
    assert commitments + units == commitments
    assert units + commitments == commitments


def test_add_with_negation_gives_unit_vector(commitments5):
    result = commitments5 + (-commitments5)
    for element in result.commitment_vector:
        assert element == SchnorrGroupElement.one(5)


def test_add_preserves_length(make_commitments):
    first = make_commitments(5, [3, 4, 5])
    second = make_commitments(5, [9, 1, 4])
    assert len(first + second) == 3


@pytest.mark.parametrize("other", NON_COMMITMENT_VALUES
                         + [F_pElement(5, 1), SchnorrGroupElement(5, 3)])
def test_add_non_commitments_raises_type_error(commitments5, other):
    with pytest.raises(
            TypeError,
            match="unsupported operand type\\(s\\) for \\+: "
                  "'PolynomialCommitments' and "):
        commitments5 + other


def test_add_different_lengths_raises_value_error(make_commitments):
    with pytest.raises(
            ValueError,
            match="cannot add two commitment vectors of different lengths"):
        make_commitments(5, [3, 4]) + make_commitments(5, [3, 4, 5])


def test_add_different_lengths_message_is_exact(make_commitments):
    with pytest.raises(ValueError) as exc_info:
        make_commitments(5, [3]) + make_commitments(5, [3, 4])
    assert str(exc_info.value) == ("cannot add two commitment vectors of "
                                   "different lengths")


def test_add_different_groups_raises_type_error(make_commitments):
    # нюанс: __add__ НЕ проверяет группу — при равных длинах TypeError
    # бросает SchnorrGroupElement.__mul__
    with pytest.raises(
            TypeError,
            match="cannot \\* with two elements in different groups"):
        make_commitments(3, [2]) + make_commitments(5, [3])


def test_add_does_not_mutate_operands(make_commitments):
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [5, 9])
    _ = first + second
    assert first == make_commitments(5, [3, 4])
    assert second == make_commitments(5, [5, 9])


def test_add_returns_new_object(make_commitments):
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [1, 1])
    result = first + second
    assert result is not first
    assert result is not second


# ===========================================================================
# 8. __sub__ — self + (-other)
# ===========================================================================

def test_sub_divides_commitments_elementwise(make_commitments):
    # mod 11: 3 * ~5 = 3*9 = 27 = 5; 4 * ~9 = 4*5 = 20 = 9
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [5, 9])
    assert _values(first - second) == (5, 9)


def test_sub_self_gives_unit_vector(commitments5):
    result = commitments5 - commitments5
    for element in result.commitment_vector:
        assert element == SchnorrGroupElement.one(5)


def test_sub_equals_add_of_negation(make_commitments):
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [5, 9])
    assert first - second == first + (-second)


def test_sub_is_anti_commutative(make_commitments):
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [5, 9])
    assert first - second == -(second - first)


def test_sub_zero_vector_is_identity(make_commitments):
    commitments = make_commitments(5, [3, 4])
    units = make_commitments(5, [1, 1])
    assert commitments - units == commitments


@pytest.mark.parametrize("other", NON_COMMITMENT_VALUES + [F_pElement(5, 1)])
def test_sub_non_commitments_raises_type_error(commitments5, other):
    with pytest.raises(
            TypeError,
            match="unsupported operand type\\(s\\) for -: "
                  "'PolynomialCommitments' and "):
        commitments5 - other


def test_sub_different_lengths_raises_value_error(make_commitments):
    with pytest.raises(
            ValueError,
            match="cannot subtract two commitment vectors of different "
                  "lengths"):
        make_commitments(5, [3, 4]) - make_commitments(5, [3, 4, 5])


def test_add_and_sub_raise_same_error_type_for_different_lengths(
        make_commitments):
    # согласованность: одна и та же ситуация (разные длины) даёт ValueError
    # и для +, и для -
    short = make_commitments(5, [3, 4])
    long = make_commitments(5, [3, 4, 5])
    with pytest.raises(ValueError):
        short + long
    with pytest.raises(ValueError):
        short - long


def test_sub_does_not_mutate_operands(make_commitments):
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [5, 9])
    _ = first - second
    assert first == make_commitments(5, [3, 4])
    assert second == make_commitments(5, [5, 9])


# ===========================================================================
# 9. Интеграция: окружение + обязательства (Pedersen VSS)
# ===========================================================================

def test_make_result_is_accepted_by_constructor(environment, make_poly):
    commitments = PolynomialCommitments.make(make_poly(5, [1, 2]),
                                             make_poly(5, [3, 4]),
                                             environment)
    again = PolynomialCommitments(commitments.commitment_vector)
    assert again == commitments


def test_commitment_elements_belong_to_environment_group(environment,
                                                         make_poly):
    commitments = PolynomialCommitments.make(make_poly(5, [1, 2]),
                                             make_poly(5, [3]), environment)
    for element in commitments.commitment_vector:
        assert isinstance(element, SchnorrGroupElement)
        assert environment.contains_SchnorrGroupElement(element)


def test_different_blinding_gives_different_commitments(environment,
                                                        make_poly):
    # свойство сокрытия: тот же секрет, другая случайность — другие обязательства
    f = make_poly(5, [1, 2])
    first = PolynomialCommitments.make(f, make_poly(5, [1]), environment)
    second = PolynomialCommitments.make(f, make_poly(5, [2]), environment)
    assert first != second


def test_different_f_gives_different_commitments(environment, make_poly):
    b = make_poly(5, [1])
    first = PolynomialCommitments.make(make_poly(5, [1]), b, environment)
    second = PolynomialCommitments.make(make_poly(5, [2]), b, environment)
    assert first != second


def test_sum_of_commitments_commits_to_sum_of_polynomials(environment,
                                                          make_poly):
    # полный цикл Pedersen VSS: участники складывают обязательства локально
    f1, b1 = make_poly(5, [1, 2]), make_poly(5, [3, 4])
    f2, b2 = make_poly(5, [4, 1]), make_poly(5, [2, 2])
    local_sum = (PolynomialCommitments.make(f1, b1, environment)
                 + PolynomialCommitments.make(f2, b2, environment))
    direct = PolynomialCommitments.make(f1 + f2, b1 + b2, environment)
    assert local_sum == direct
    assert len(local_sum) == environment.threshold


def test_negation_roundtrip_via_make(environment, make_poly):
    f, b = make_poly(5, [1, 2]), make_poly(5, [3, 4])
    commitments = PolynomialCommitments.make(f, b, environment)
    roundtrip = commitments + (-commitments)
    for element in roundtrip.commitment_vector:
        assert element == SchnorrGroupElement.one(5)


# ===========================================================================
# 10. Прочие нюансы
# ===========================================================================

def test_unary_operations_do_not_mutate(commitments5):
    _ = +commitments5
    _ = -commitments5
    assert _values(commitments5) == (3, 4, 5)


def test_binary_operations_do_not_mutate(make_commitments):
    first = make_commitments(5, [3, 4])
    second = make_commitments(5, [5, 9])
    _ = first + second
    _ = first - second
    assert _values(first) == (3, 4)
    assert _values(second) == (5, 9)


def test_make_result_len_matches_environment_threshold(make_environment,
                                                       make_poly):
    for threshold in (1, 2, 3, 5):
        env = make_environment(5, threshold)
        commitments = PolynomialCommitments.make(make_poly(5, [1]),
                                                 make_poly(5, [2]), env)
        assert len(commitments) == threshold
        assert len(commitments.commitment_vector) == threshold


def test_commitments_of_equal_polynomials_are_equal(make_environment,
                                                    make_poly):
    env1 = make_environment(5, 3)
    env2 = make_environment(5, 3)
    f = make_poly(5, [1, 2])
    b = make_poly(5, [3])
    assert (PolynomialCommitments.make(f, b, env1)
            == PolynomialCommitments.make(f, b, env2))


def test_commitments_of_different_thresholds_have_different_lengths(
        make_environment, make_poly):
    f = make_poly(5, [1])
    b = make_poly(5, [2])
    short = PolynomialCommitments.make(f, b, make_environment(5, 2))
    long = PolynomialCommitments.make(f, b, make_environment(5, 4))
    assert len(short) == 2
    assert len(long) == 4
    assert short != long
