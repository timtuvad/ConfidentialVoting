def is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True

# Простое число больше n
def get_prime_greater_than(n: int) -> int:
    p = max(n, 2) + 1
    while not is_prime(p):
        p += 1
    return p