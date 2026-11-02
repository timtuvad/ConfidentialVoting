from typing import List
from ..utils import is_prime
from .field import F_pElement
from ..secret_share import SecretShare

class F_pPolynomial:

    def __init__(self, p: int | None = None,
                 coefficients: List[F_pElement]|None = None):
        if p is not None and not is_prime(p):
            raise ValueError(f"p must be prime, cannot be {p}")
        if p is None and (coefficients is None or len(coefficients) == 0):
            raise ValueError("p or coefficients must be provided")
        self.__p = coefficients[0].p if p is None else p
        self.__coefficients = tuple([] if coefficients is None
            else F_pPolynomial.__rstrip(coefficients, F_pElement.zero(self.__p)))
        if any(coefficient.p != self.__p
               for coefficient in self.__coefficients):
            raise TypeError(
                f"coefficients must be from the field of order {self.__p}")

    @property
    def p(self) -> int:
        return self.__p

    @property
    def degree(self) -> int|float:
        if len(self.__coefficients) == 0:
            return float('-inf')
        return len(self.__coefficients) - 1

    @staticmethod
    def one(p: int):
        return F_pPolynomial(p, coefficients=[F_pElement.one(p)])

    @staticmethod
    def random(p: int, degree_bound: int,
               least_coefficient: F_pElement | None = None):
        if (least_coefficient is not None and
            least_coefficient.p != p):
            raise ValueError(
                f"least_coefficient must be from the field of order {p}")
        if degree_bound < 0:
            raise ValueError("degree_bound must be non-negative")
        if least_coefficient is None:
            coefficients = (F_pElement.random(p)
                            for _ in range(degree_bound+1))
        else:
            coefficients = ([least_coefficient] + [F_pElement.random(p)
                            for _ in range(degree_bound)])
        return F_pPolynomial(p, coefficients=coefficients)

    def share_at(self, x: F_pElement) -> SecretShare:
        return SecretShare(x, self(x))

    def __getitem__(self, index: int) -> F_pElement:
        if not isinstance(index, int):
            raise TypeError("index must be an integer")
        if index < 0:
            raise IndexError("index cannot be negative")
        if index >= len(self.__coefficients):
            return F_pElement.zero(self.__p)
        return self.__coefficients[index]

    def __eq__(self, other) -> bool:
        if not isinstance(other, F_pPolynomial):
            return False
        return (self.__p == other.__p
                and self.__coefficients == other.__coefficients)
        
    def __ne__(self, other) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash((self.__p, self.__coefficients))

    def str_format(self, degree_bound: int|None = None) -> str:
        degree = (degree_bound
                  if degree_bound is not None
                  else max(self.degree, 0))
        return " + ".join(str(self[i]) + (f"x^{i}" if i != 0 else "")
                          for i in range(degree, -1, -1))

    def __str__(self) -> str:
        return self.str_format()

    def __repr__(self) -> str:
        return (f"F_pPolynomial({self.__p}, "
                f"coefficients={self.__coefficients})")

    def __pos__(self):
        return self

    def __neg__(self):
        return F_pPolynomial(self.__p,
                             coefficients=[-c for c in self.__coefficients])

    def __add__(self, other):
        self.__check_types(other, '+')
        coeff_count = max(len(self.__coefficients), len(other.__coefficients))
        coefficients = (self[i]+other[i] for i in range(coeff_count))
        return F_pPolynomial(self.__p, coefficients=coefficients)

    def __sub__(self, other):
        self.__check_types(other, '-')
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, F_pElement):
            if self.__p != other.p:
                raise TypeError(f"cannot multiply polynomial to element in "
                                "different field")
            return F_pPolynomial(self.__p,
                        coefficients=[c*other for c in self.__coefficients])
        else:
            self.__check_types(other, '*')
            if not self or not other:
                return F_pPolynomial(self.__p)
            degree = self.degree + other.degree
            coefficients = [F_pElement.zero(self.__p)] * (degree + 1)
            for i in range(len(self.__coefficients)):
                for j in range(len(other.__coefficients)):
                    coefficients[i + j] += (self.__coefficients[i] *
                                            other.__coefficients[j])
            return F_pPolynomial(self.__p, coefficients=coefficients)

    def __truediv__(self, other: F_pElement):
        return self * (~other)

    def __bool__(self) -> bool:
        return len(self.__coefficients) > 0

    def __call__(self, x: F_pElement) -> F_pElement:
        if not isinstance(x, F_pElement):
            raise TypeError(f"polynomial argument must be a F_pElement, not "
                            f"'{x.__class__.__name__}'")
        value = F_pElement.zero(self.__p)
        x_to_i = F_pElement.one(self.__p)
        for coefficient in self.__coefficients:
            value += coefficient * x_to_i
            x_to_i *= x
        return value

    @staticmethod
    def __rstrip(array, value):
        result = list(array)
        while len(result) > 0 and result[-1] == value:
            result.pop()
        return result

    def __check_types(self, other, operation: str):
        if not isinstance(other, F_pPolynomial):
            raise TypeError(f"unsupported operand type(s) for {operation}: "
                            f"'{self.__class__.__name__}' and "
                            f"'{other.__class__.__name__}'")
        if self.__p != other.__p:
            raise TypeError(f"cannot {operation} with two polynomials in "
                            "different fields: "
                            f"F_{self.__p} and F_{other.__p}")