from typing import List
from src.core.secret_share import SecretShare
from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial

class ShamirCryptography:

    # TODO : добавить подробные комментарии
    # TODO : добавить необходимые проверки
    def __init__(self, t: int, person_id: F_pElement, vote: F_pElement):
        self.__t = t
        self.__person_id = person_id
        self.__vote = vote
        self.__personal_polynomial = F_pPolynomial.random(self.__person_id.p,
                        degree_bound=self.__t-1, least_coefficient=self.__vote)

    @property
    def person_id(self) -> F_pElement:
        return self.__person_id

    @property
    def vote(self) -> F_pElement:
        return self.__vote

    @property
    def personal_polynomial(self) -> F_pPolynomial:
        return self.__personal_polynomial

    def share_part_with(self, other_id: F_pElement) -> SecretShare:
        return self.__personal_polynomial.share_at(other_id)

    # TODO : написать подробный комментарий
    # TODO : реализовать
    @staticmethod
    def total_sum_at_participant(share_parts: List[SecretShare]) -> SecretShare:
        raise NotImplementedError

    # TODO : написать подробный комментарий
    # TODO : реализовать
    @staticmethod
    def total_sum_at_zero(participants_shares: List[SecretShare]) -> F_pElement:
        raise NotImplementedError