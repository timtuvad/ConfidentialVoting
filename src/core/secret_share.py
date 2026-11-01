from src.core.algebra.field import F_pElement

# Известное значение многочлена f(x) = value
# Сам многочлен f неизвестен
class SecretShare:

    def __init__(self, x: F_pElement, value: F_pElement):
        if x.p != value.p:
            raise ValueError("x and value must be in the same field")
        self.__x = x
        self.__value = value

    @property
    def x(self):
        return self.__x

    @property
    def value(self):
        return self.__value

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return (self.__x == other.__x
                and self.__value == other.__value)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.__x, self.__value))

    def __str__(self):
        return f"f({self.__x})={self.__value}"

    def __repr__(self):
        return f"SecretShare({self.__x}, {self.__value})"

    # TODO : добавить функции алгебраических операций с секретами