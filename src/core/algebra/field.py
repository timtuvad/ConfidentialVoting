import secrets
from ..utils import is_prime

# F_pElement(p[, value=k]) — элемент поля F_p (со значением k mod p)
class F_pElement:

    def __init__(self, p: int, value: int|None = None):
        F_pElement.__check_is_prime(p)
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
    def one(p: int):
        return F_pElement(p, value=1)

    @staticmethod
    def random(p: int):
        """Случайный элемент конечного поля F_p, где `p` — простое.
            
            Args:
                p (int): порядок поля F_p
            
            Returns:
                F_pElement: случайный элемент конечного поля F_p
            
            Raises:
                ValueError: если `p` — не простое число
        """

        F_pElement.__check_is_prime(p)
        return F_pElement._new(p, value=secrets.randbelow(p))

    def __str__(self):
        return f"({self.__value} mod {self.__p})"

    def __repr__(self):
        return f"F_pElement(p={self.__p}, value={self.__value})"

    def __eq__(self, other):
        if not isinstance(other, F_pElement):
            return False
        return self.__p == other.__p and self.__value == other.__value
    
    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.__p, self.__value))

    def __pos__(self):
        return self

    def __neg__(self):
        return F_pElement._new(self.__p, (-self.__value) % self.__p)

    def __add__(self, other):
        self.__check_types(other, "+")
        return F_pElement._new(self.__p, (self.__value + other.__value) % self.__p)

    def __sub__(self, other):
        self.__check_types(other, "-")
        return F_pElement._new(self.__p, (self.__value - other.__value) % self.__p)

    def __mul__(self, other):
        if not isinstance(other, F_pElement):
            return other * self
        else:
            self.__check_types(other, "*")
            return F_pElement._new(self.__p, (self.__value * other.__value) % self.__p)

    def __pow__(self, exp: int):
        self.__check_types(exp, "** or pow()", int)
        return F_pElement._new(self.__p, pow(self.__value, exp, self.__p))

    def __invert__(self):
        if self.__value == 0:
            raise ZeroDivisionError(
                f"cannot divide by zero in the field of order {self.__p}")
        return self.__pow__(-1)

    def __truediv__(self, other):
        self.__check_types(other, "/")
        return self * (~other)

    def __bool__(self) -> bool:
        return self.__value != 0

    @classmethod
    def _new(cls, p: int, value: int) -> "F_pElement":
        obj = cls.__new__(cls)
        obj.__p = p
        obj.__value = value
        return obj

    @staticmethod
    def __check_is_prime(p: int):
        if not is_prime(p):
            raise ValueError(f"p must be prime, cannot be {p}")

    def __check_types(self, other, operation: str, needed: type | None = None):
        if not isinstance(other, needed or type(self)):
            raise TypeError(f"unsupported operand type(s) for {operation}: "
                            f"'{self.__class__.__name__}' and "
                            f"'{other.__class__.__name__}'")
        if needed is None:
            if self.__p != other.__p:
                raise TypeError(f"cannot {operation} with two elements in "
                                "different fields: "
                                f"F_{self.__p} and F_{other.__p}")