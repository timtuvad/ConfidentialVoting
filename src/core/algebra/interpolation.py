from typing import List, Iterable
from .field import F_pElement
from .polynomial import F_pPolynomial
from ..share import Share

class LagrangeInterpolation:

    def __init__(self, shares: Iterable[Share]):
        shares = list(shares)
        if len(shares) < 1:
            raise ValueError("at least one share is required")
        self.__p = shares[0].p
        for share in shares:
            if share.p != self.__p:
                raise TypeError("all shares must be from the same field")
        self.__shares = LagrangeInterpolation.__filter_unique(shares)
        self.__largange_polynomial = self.__compute_lagrange_polynomial()

    @property
    def lagrange_polynomial(self) -> F_pPolynomial:
        return self.__largange_polynomial

    @property
    def shares(self) -> List[Share]:
        return self.__shares

    def __call__(self, x: F_pElement) -> Share:
        return Share.make(self.__largange_polynomial, x)

    @staticmethod
    def __filter_unique(shares: Iterable[Share]) -> List[Share]:
        unique_shares = {}
        for share in shares:
            if share.x in unique_shares:
                other_share = unique_shares[share.x]
                if share.value != other_share.value:
                    raise ValueError("cannot interpolate two different "
                                     "values at the same x: "
                                     f"{share} and {other_share}")
            else:
                unique_shares[share.x] = share
        return list(unique_shares.values())

    def __compute_largange_term(self, i: int) -> F_pPolynomial:
        term = F_pPolynomial.one(self.__p)
        one = F_pElement.one(self.__p)
        for j in range(len(self.__shares)):
            if i != j:
                term *= F_pPolynomial(self.__p, [-self.__shares[j].x, one])
        denominator = term(self.__shares[i].x)
        return self.__shares[i].value * term / denominator

    def __compute_lagrange_polynomial(self) -> F_pPolynomial:
        zero = F_pPolynomial(self.__p)
        return sum((self.__compute_largange_term(i)
                    for i in range(len(self.__shares))), start=zero)