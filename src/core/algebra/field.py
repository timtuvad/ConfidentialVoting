from random import randrange
from ..utils import is_prime

# F_pElement(p[, value=k]) — элемент поля F_p (со значением k mod p)
class F_pElement:

    def __init__(self, p: int, value: int|None = None):
        if not is_prime(p):
            raise ValueError(f"p must be prime, cannot be {p}")
        self.__p = p
        self.__value = 0 if value is None else value % p

    @property
    def p(self):
        return self.__p

    @property
    def value(self):
        return self.__value

    @staticmethod
    def zero(p: int):
        return F_pElement(p, value=0)

    @staticmethod
    def unit(p: int):
        return F_pElement(p, value=1)

    @staticmethod
    def random(p: int):
        return F_pElement(p, value=randrange(p))

    @staticmethod
    def randomGenerating(p: int):
        if not is_prime(p):
            raise ValueError(f"p must be prime, cannot be {p}")
        if p == 2:
            return F_pElement.unit(p)
        return F_pElement(p, value=randrange(2, p))

    def __str__(self):
        return f"({self.__value} mod {self.__p})"

    def __repr__(self):
        return f"F_pElement(p={self.__p}, value={self.__value})"

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self.__p == other.__p and self.__value == other.__value
    
    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.__p, self.__value))

    def __pos__(self):
        return self

    def __neg__(self):
        return F_pElement(self.__p, (-self.__value) % self.__p)

    def __add__(self, other):
        if type(self) is not type(other):
            raise TypeError("unsupported operand type(s) for +: "
               f"'{self.__class__.__name__}' and '{other.__class__.__name__}'")
        if self.__p != other.__p:
            raise TypeError("cannot add two elements in different fields")
        return F_pElement(self.__p, (self.__value + other.__value) % self.__p)

    def __sub__(self, other):
        if type(self) is not type(other):
            raise TypeError("unsupported operand type(s) for -: "
               f"'{self.__class__.__name__}' and '{other.__class__.__name__}'")
        return self + (-other)

    def __mul__(self, other):
        if type(self) is not type(other):
            raise TypeError("unsupported operand type(s) for *: "
               f"'{self.__class__.__name__}' and '{other.__class__.__name__}'")
        if self.__p != other.__p:
            raise TypeError("cannot multiply two elements in different fields")
        return F_pElement(self.__p, (self.__value * other.__value) % self.__p)

    def __pow__(self, exp: int):
        if type(exp) is not int:
            raise TypeError("unsupported operand type(s) for ** or pow(): "
               f"'{self.__class__.__name__}' and '{exp.__class__.__name__}'")
        return F_pElement(self.__p, pow(self.__value, exp, self.__p))

    def __invert__(self):
        if self.__value == 0:
            raise ZeroDivisionError(
                f"cannot divide by zero in the field of order {self.__p}")
        return self.__pow__(-1)

    def __truediv__(self, other):
        if type(self) is not type(other):
            raise TypeError("unsupported operand type(s) for /: "
               f"'{self.__class__.__name__}' and '{other.__class__.__name__}'")
        if self.__p != other.__p:
            raise TypeError("cannot divide two elements in different fields")
        return self * (~other)