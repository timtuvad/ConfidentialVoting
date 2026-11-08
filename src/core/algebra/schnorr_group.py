import secrets
from ..utils import is_prime
from .field import F_pElement

# SchnorrGroupElement(order, value=k) — элемент группы Шнорра порядка order со значением k mod (2*order+1)
class SchnorrGroupElement:

    def __init__(self, order: int, value: int):
        SchnorrGroupElement.__check_is_special_prime(order)
        self.__q = 2*order + 1
        self.__order = order
        if pow(value % self.__q, order, self.__q) != 1:
            raise ValueError(f"value must be a non-zero square mod "
                             f"{self.__q}, cannot be "
                             f"({value} mod {self.__q})")
        self.__value = value % self.__q

    @property
    def order(self):
        return self.__order

    @property
    def value(self):
        return self.__value

    @property
    def is_generator(self) -> bool:
        return self.__value != 1

    @staticmethod
    def one(order: int):
        return SchnorrGroupElement(order, value=1)

    @staticmethod
    def random_generator(order: int):
        """Случайный порождающий элемент группы Шнорра порядка `order`
        
        Args:
            order (int): порядок группы
        
        Returns:
            SchnorrGroupElement: случайный порождающий элемент группы Шнорра
                                порядка `order`
        
        Raises:
            ValueError: если `order` не является простым числом, таким что
                        `2*order+1` также простое
        """

        SchnorrGroupElement.__check_is_special_prime(order)
        q = 2*order + 1
        square_root = 2 + secrets.randbelow(q - 3)
        return SchnorrGroupElement._new(order,
                                        value=square_root*square_root%q)

    def __str__(self):
        return f"({self.__value} mod {self.__q})"

    def __repr__(self):
        return f"SchnorrGroupElement({self.__order}, value={self.__value})"

    def __eq__(self, other):
        if not isinstance(other, SchnorrGroupElement):
            return False
        return self.__order == other.__order and self.__value == other.__value
    
    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.__order, self.__value))

    def __mul__(self, other):
        self.__check_types(other, "*")
        return SchnorrGroupElement._new(self.__order,
                                value=(self.__value*other.__value)%self.__q)

    def __pow__(self, exp: int|F_pElement):
        if isinstance(exp, F_pElement):
            if self.__order != exp.p:
                raise TypeError(f"cannot raise ({self.__value} mod "
                                f"{self.__q}) to a power from field "
                                f"F_{exp.p}")
            exp_value = exp.value
        else:
            self.__check_types(exp, "** or pow()", int)
            exp_value = exp
        return SchnorrGroupElement._new(self.__order,
                                value=pow(self.__value, exp_value, self.__q))

    def __invert__(self):
        return SchnorrGroupElement._new(self.__order,
                                    value=pow(self.__value, -1, self.__q))

    def __truediv__(self, other):
        self.__check_types(other, "/")
        return self * (~other)

    @classmethod
    def _new(cls, order: int, value: int) -> "SchnorrGroupElement":
        obj = cls.__new__(cls)
        obj.__q = 2*order + 1
        obj.__order = order
        obj.__value = value
        return obj

    @staticmethod
    def __check_is_special_prime(order: int):
        if not is_prime(order):
            raise ValueError(f"order must be prime, cannot be {order}")
        q = 2*order + 1
        if not is_prime(q):
            raise ValueError("order must be prime such that 2*order+1 is "
                             f"also a prime number, cannot be {order}")

    def __check_types(self, other, operation: str, needed: type|None = None):
        if not isinstance(other, needed or type(self)):
            raise TypeError(f"unsupported operand type(s) for {operation}: "
                            f"'{self.__class__.__name__}' and "
                            f"'{other.__class__.__name__}'")
        if needed is None:
            if self.__order != other.__order:
                raise TypeError(f"cannot {operation} with two elements in "
                                "different groups with orders "
                                f"{self.__order} and {other.__order}")