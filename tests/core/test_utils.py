import pytest
from src import is_prime, get_prime_greater_than

@pytest.mark.parametrize("n, expected", [
    (-3, False),
    (-1, False),
    (0, False),
    (1, False),
    (2, True),
    (3, True),
    (4, False),
    (5, True),
    (6, False),
    (589423, False),
    (954529, False),
    (10**9 + 7, True)
])
def test_is_prime(n, expected):
    assert is_prime(n) == expected

@pytest.mark.parametrize("n, expected", [
    (-7, 3),
    (-1, 3),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 5),
    (4, 5),
    (5, 7),
    (6, 7),
    (7, 11),
    (11, 13),
    (13, 17),
    (17, 19),
    (19, 23),
    (10**9, 10**9 + 7)
])
def test_get_prime_greater_than(n, expected):
    assert get_prime_greater_than(n) == expected