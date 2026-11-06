from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial

# Известное значение многочлена f(x) = value
# Сам многочлен f неизвестен
class Share:

    def __init__(self, x: F_pElement, value: F_pElement):
        if not isinstance(x, F_pElement):
            raise TypeError("x must be an instance of F_pElement, not "
                            f"'{x.__class__.__name__}'")
        if not isinstance(value, F_pElement):
            raise TypeError("value must be an instance of F_pElement, not "
                            f"'{value.__class__.__name__}'")
        if x.p != value.p:
            raise TypeError("x and value must be in the same field")
        self._x = x
        self._value = value

    @property
    def x(self):
        return self._x

    @property
    def value(self):
        return self._value

    @property
    def p(self):
        return self._x.p

    @staticmethod
    def make(f: F_pPolynomial, x: F_pElement):
        return Share(x, f(x))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Share):
            return False
        return (self._x == other._x
                and self._value == other._value)

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash((self._x, self._value))

    def __str__(self):
        return f"f({self._x})={self._value}"

    def __repr__(self):
        return f"Share({self._x}, {self._value})"

    def __pos__(self):
        return self
    
    def __neg__(self):
        return Share(self._x, -self._value)

    def __add__(self, other):
        self.__check_types(other, "+")
        return Share(self._x, self._value + other._value)

    def __sub__(self, other):
        self.__check_types(other, "-")
        return Share(self._x, self._value - other._value)

    def __check_types(self, other, operation: str):
        if not isinstance(other, Share):
            raise TypeError(f"unsupported operand type(s) for {operation}: "
                            f"'{self.__class__.__name__}' and "
                            f"'{other.__class__.__name__}'")
        if self._x != other._x:
            raise TypeError(f"cannot {operation} with two shares with "
                            f"different x values: {self} and {other}")