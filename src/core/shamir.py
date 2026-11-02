from typing import List
from src.core.share import Share
from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial
from src.core.algebra.interpolation import LagrangeInterpolation

class ShamirCryptography:

    # TODO : добавить подробные комментарии
    def __init__(self, threshold: int, person_id: F_pElement,
                 secret: F_pElement):
        if not isinstance(threshold, int):
            raise TypeError("threshold must be an integer, not "
                            f"'{threshold.__class__.__name__}'")
        if not isinstance(person_id, F_pElement):
            raise TypeError("person_id must be an instance of F_pElement, "
                            f"not '{person_id.__class__.__name__}'")
        if not isinstance(secret, F_pElement):
            raise TypeError("secret must be an instance of F_pElement, not "
                            f"'{secret.__class__.__name__}'")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if person_id.p != secret.p:
            raise ValueError("person_id and secret must be from "
                             "the same field")
        
        self.__threshold = threshold
        self.__person_id = person_id
        self.__secret = secret
        self.__personal_polynomial = F_pPolynomial.random(self.__person_id.p,
                                            degree_bound=self.__threshold-1,
                                            least_coefficient=self.__secret)

    
    @property
    def person_id(self) -> F_pElement:
        return self.__person_id

    @property
    def secret(self) -> F_pElement:
        return self.__secret

    @property
    def personal_polynomial(self) -> F_pPolynomial:
        return self.__personal_polynomial

    def share_part_with(self, other_id: F_pElement) -> Share:
        return self.__personal_polynomial.share_at(other_id)

    # TODO : написать подробный комментарий
    @staticmethod
    def total_sum_at_participant(shares: List[Share]) -> Share:
        x = shares[0].x
        return sum(shares, start=Share(x, F_pElement.zero(x.p)))

    # TODO : написать подробный комментарий
    @staticmethod
    def sum_of_secrets(shares_of_sum: List[Share],
                       threshold: int) -> F_pElement:
        interpolation = LagrangeInterpolation(shares_of_sum)
        if len(interpolation.shares) < threshold:
            raise ValueError("not enough shares to sum secrets")
        p = shares_of_sum[0].x.p
        return interpolation(F_pElement.zero(p))