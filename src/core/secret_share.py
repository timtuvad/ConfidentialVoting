from src.core.algebra.field import F_pElement

# Известное значение многочлена f(x) = value
# Сам многочлен f неизвестен
class SecretShare:

    def __init__(self, x: F_pElement, value: F_pElement):
        if not isinstance(x, F_pElement):
            raise TypeError("x must be an instance of F_pElement, not "
                            f"'{x.__class__.__name__}'")
        if not isinstance(value, F_pElement):
            raise TypeError("value must be an instance of F_pElement, not "
                            f"'{value.__class__.__name__}'")
        if x.p != value.p:
            raise TypeError("x and value must be in the same field")
        self.__x = x
        self.__value = value

    @property
    def x(self):
        return self.__x

    @property
    def value(self):
        return self.__value

    @property
    def p(self):
        return self.__x.p

    def __eq__(self, other) -> bool:
        if not isinstance(other, SecretShare):
            return False
        return (self.__x == other.__x
                and self.__value == other.__value)

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash((self.__x, self.__value))

    def __str__(self):
        return f"f({self.__x})={self.__value}"

    def __repr__(self):
        return f"SecretShare({self.__x}, {self.__value})"

    def __pos__(self):
        return self
    
    def __neg__(self):
        return SecretShare(self.__x, -self.__value)

    def __add__(self, other):
        self.__check_types(other, "+")
        return SecretShare(self.__x, self.__value + other.__value)

    def __sub__(self, other):
        self.__check_types(other, "-")
        return SecretShare(self.__x, self.__value - other.__value)

    def __mul__(self, other):
        self.__check_types(other, "*")
        return SecretShare(self.__x, self.__value * other.__value)

    def __truediv__(self, other):
        self.__check_types(other, "/")
        return SecretShare(self.__x, self.__value / other.__value)

    def __pow__(self, exp: int):
        return SecretShare(self.__x, self.__value ** exp)

    def __check_types(self, other, operation: str):
        if not isinstance(other, SecretShare):
            raise TypeError(f"unsupported operand type(s) for {operation}: "
                            f"'{self.__class__.__name__}' and "
                            f"'{other.__class__.__name__}'")
        if self.__x != other.__x:
            raise TypeError(f"cannot {operation} with two secret shares with "
                            f"different x values: {self} and {other}")