"""Исправление конструктора и типа результата из feature/PedersenVSS."""
from typing import List
from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial
from src.core.algebra.interpolation import LagrangeInterpolation
from src.core.share import Share
from src.core.verifiable_share import VerifiableShare
from src.core.environment import CryptographyEnvironment
from src.core.commitments import PolynomialCommitments
from src.core.pedersen import PedersenVSS


class ShamirCryptography:
    def __init__(self, environment: CryptographyEnvironment,
                 person_id: F_pElement, secret: F_pElement):
        if not isinstance(environment, CryptographyEnvironment):
            raise TypeError('environment must be CryptographyEnvironment')
        if not environment.contains_F_pElement(person_id):
            raise TypeError('person_id must be in F_p from environment')
        if not environment.contains_F_pElement(secret):
            raise TypeError('secret must be in F_p from environment')
        if person_id.value == 0:
            raise ValueError('person_id must not be zero')
        self.__environment = environment
        self.__person_id = person_id
        self.__secret = secret
        # В feature/PedersenVSS ошибочно использовался self.__threshold.
        self.__personal_polynomial = F_pPolynomial.random(
            self.__person_id.p,
            degree_bound=environment.threshold - 1,
            least_coefficient=self.__secret,
        )
        self.__pedersen_vss = PedersenVSS(environment, self.__personal_polynomial)

    @property
    def environment(self) -> CryptographyEnvironment:
        return self.__environment

    @property
    def person_id(self) -> F_pElement:
        return self.__person_id

    @property
    def secret(self) -> F_pElement:
        return self.__secret

    @property
    def personal_polynomial(self) -> F_pPolynomial:
        return self.__personal_polynomial

    @property
    def pedersen_vss(self) -> PedersenVSS:
        return self.__pedersen_vss

    @property
    def blinding_polynomial(self) -> F_pPolynomial:
        return self.__pedersen_vss.blinding_polynomial

    @property
    def commitments(self) -> PolynomialCommitments:
        """Публичные обязательства (удобный alias для протокола)."""
        return self.__pedersen_vss.public_commitments

    def verifiable_share_to(self, other_id: F_pElement) -> VerifiableShare:
        return self.__pedersen_vss.verifiable_share_to(other_id)

    def verify(self, verifiable_share: VerifiableShare,
               public_commitments: PolynomialCommitments) -> bool:
        return self.__pedersen_vss.verify(verifiable_share, public_commitments)

    @staticmethod
    def commitments_of_total_sum(commitments_of_polynomials: List[PolynomialCommitments]
                                 ) -> PolynomialCommitments:
        if not commitments_of_polynomials:
            raise ValueError('Нет обязательств')
        return PedersenVSS.commitments_of_total_sum(commitments_of_polynomials)

    @staticmethod
    def total_sum_at_participant(verifiable_shares: List[VerifiableShare]
                                 ) -> VerifiableShare:
        if not verifiable_shares:
            raise ValueError('Нет долей')
        result = verifiable_shares[0]
        for share in verifiable_shares[1:]:
            result += share
        return result

    @staticmethod
    def sum_of_secrets(shares_of_sum: List[Share], threshold: int) -> F_pElement:
        if len(shares_of_sum) < threshold:
            raise ValueError('not enough shares to sum secrets')
        interpolation = LagrangeInterpolation(shares_of_sum)
        if len(interpolation.shares) < threshold:
            raise ValueError('not enough unique shares to sum secrets')
        p = shares_of_sum[0].x.p
        # LagrangeInterpolation.__call__ возвращает Share, а не F_pElement.
        return interpolation(F_pElement.zero(p)).value
