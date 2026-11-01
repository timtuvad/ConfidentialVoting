from typing import List
from ..utils import is_prime
from .field import F_pElement
from ..secret_share import SecretShare

class F_pPolynomial:

    def __init__(self, p: int | None = None, degree_bound: int|None = None,
                 coefficients: List[F_pElement]|None = None):
        if p is not None and not is_prime(p):
            raise ValueError(f"p must be prime, cannot be {p}")
        if degree_bound is not None and degree_bound < 0:
            raise ValueError("degree_bound must be non-negative")
        if p is None and (coefficients is None or len(coefficients) == 0):
            raise ValueError("p or coefficients must be provided")
        if degree_bound is not None and coefficients is not None:
            if len(coefficients) - 1 > degree_bound:
                raise ValueError(
                    "degree_bound is less than coefficients degree")
        self.__p = coefficients[0].p if p is None else p
        self.__coefficients = tuple([] if coefficients is None
            else F_pPolynomial.__rstrip__(coefficients, F_pElement.zero(self.__p)))
        for coefficient in self.__coefficients:
            if coefficient.p != self.__p:
                raise ValueError(
                    f"coefficients must be from the field of order {self.__p}")
        self.__degree_bound = degree_bound

    @property
    def p(self):
        return self.__p

    @property
    def degree(self):
        if len(self.__coefficients) == 0:
            return float('-inf')
        return len(self.__coefficients) - 1

    @property
    def degree_bound(self):
        return self.__degree_bound

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
            least_coefficient = F_pElement.random(p)
        return F_pPolynomial(p, degree_bound, coefficients=
                             [least_coefficient] + [F_pElement.random(p)
                                for _ in range(degree_bound)])

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return (self.__p == other.__p
                and self.__degree_bound == other.__degree_bound
                and self.__coefficients == other.__coefficients)
        
    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.__p, self.__degree_bound, self.__coefficients))

    def __str__(self):
        deg = (self.__degree_bound
               if self.__degree_bound is not None
               else max(self.degree, 0))
        return " + ".join(
                (f"{self.__coefficients[i]}"
                    if i < len(self.__coefficients)
                    else f"(0 mod {self.__p})")
                + (f"x^{i}" if i != 0 else "")
            for i in range(deg, -1, -1))

    def __repr__(self):
        return (f"F_pPolynomial({self.__p}, {self.__degree_bound}, "
                f"coefficients={self.__coefficients})")

    def __pos__(self):
        return self

    def __neg__(self):
        return F_pPolynomial(self.__p, self.__degree_bound,
                             coefficients=[-c for c in self.__coefficients])

    def __add__(self, other):
        if type(self) is not type(other):
            raise TypeError("unsupported operand type(s) for +: "
               f"'{self.__class__.__name__}' and '{other.__class__.__name__}'")
        if self.__p != other.__p:
            raise ValueError("Cannot add two polynomials in different fields")
        if self.__degree_bound is None or other.__degree_bound is None:
            degree_bound = None
        else:
            if self.__degree_bound != other.__degree_bound:
                raise ValueError(
                    "Cannot add two polynomials with different degree_bound")
            degree_bound = self.__degree_bound
        result = []
        for i in range(max(len(self.__coefficients), len(other.__coefficients))):
            if i >= len(self.__coefficients):
                result.append(other.__coefficients[i])
            elif i >= len(other.__coefficients):
                result.append(self.__coefficients[i])
            else:
                result.append(self.__coefficients[i] + other.__coefficients[i])
        return F_pPolynomial(self.__p, degree_bound, coefficients=result)

    def __sub__(self, other):
        if type(self) is not type(other):
            raise TypeError("unsupported operand type(s) for -: "
               f"'{self.__class__.__name__}' and '{other.__class__.__name__}'")
        return self + (-other)

    def __call__(self, x: F_pElement) -> F_pElement:
        value = F_pElement.zero(self.__p)
        x_to_i = F_pElement.unit(self.__p)
        for coefficient in self.__coefficients:
            value = value + coefficient * x_to_i
            x_to_i = x_to_i * x
        return value

    def share_at(self, x: F_pElement) -> SecretShare:
        return SecretShare(x, self(x))

    @staticmethod
    def __rstrip__(array, value):
        result = list(array)
        while len(result) > 0 and result[-1] == value:
            result.pop()
        return result