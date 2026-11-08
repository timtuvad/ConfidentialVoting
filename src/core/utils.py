import math

def is_prime(n: int) -> bool:
    """Проверяет, является ли аргумент простым целым числом."""

    if not isinstance(n, int):
        return False
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True

def get_prime_greater_than(n: int) -> int:
    """Возвращает первое по возрастанию простое число, большее чем `n`."""
    p = max(math.floor(n)+1, 2)
    while not is_prime(p):
        p += 1
    return p

def get_special_prime_greater_than(n: int) -> int:
    """Возвращает первое по возрастанию простое число `p`, большее чем `n`,\
    такое, что `2*p+1` также простое."""
    if 2 > n:
        return 2
    if 3 > n:
        return 3
    # (6k-1 > n) <=> (6k > n+1) <=> (k > (n+1)/6)
    k = math.floor((n + 1) / 6) + 1
    p = 6 * k - 1
    while not (is_prime(p) and is_prime(2*p+1)):
        p += 6
    return p

    