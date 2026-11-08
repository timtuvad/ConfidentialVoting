from typing import List
from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial
from src.core.verifiable_share import VerifiableShare
from src.core.environment import CryptographyEnvironment
from src.core.commitments import PolynomialCommitments

class PedersenVSS:

    def __init__(self, environment: CryptographyEnvironment,
                 personal_polynomial: F_pPolynomial):
        if not isinstance(environment, CryptographyEnvironment):
            raise TypeError("environment must be an instance of "
                            "CryptographyEnvironment")
        if not environment.contains_F_pPolynomial(personal_polynomial):
            raise TypeError("personal_polynomial must be a correct "
                            "polynomial from the perspective of the "
                            "environment")
        self.__environment = environment
        self.__personal_polynomial = personal_polynomial
        self.__blinding_polynomial = F_pPolynomial.random(
            self.__environment.p,
            degree_bound=self.__environment.threshold-1
        )

    @property
    def environment(self) -> CryptographyEnvironment:
        return self.__environment

    @property
    def personal_polynomial(self) -> F_pPolynomial:
        return self.__personal_polynomial

    @property
    def blinding_polynomial(self) -> F_pPolynomial:
        return self.__blinding_polynomial
    
    @property
    def public_commitments(self) -> PolynomialCommitments:
        return PolynomialCommitments.make(
            self.__personal_polynomial,
            self.__blinding_polynomial,
            self.__environment
        )

    def verifiable_share_to(self, other_id: F_pElement) -> VerifiableShare:
        return VerifiableShare.make(self.__personal_polynomial, other_id,
                                    self.__blinding_polynomial)

    def verify(self, verifiable_share: VerifiableShare,
               public_commitments: PolynomialCommitments) -> bool:
        environment = self.__environment
        if not environment.contains_VerifiableShare(verifiable_share):
            raise TypeError("verifiable_share must be a correct "
                            "VerifiableShare from the perspective of the "
                            "environment")
        if not environment.contains_PolynomialCommitments(public_commitments):
            raise TypeError("public_commitments must be correct "
                            "PolynomialCommitments from the perspective of "
                            "the environment")
        recipient_id = verifiable_share.x
        value = verifiable_share.value
        blinding_share = verifiable_share.blinding_share
        g0 = environment.g0
        h0 = environment.h0
        blinding_value = g0**value * h0**blinding_share
        commited_value = public_commitments.blinding_value_at(recipient_id)
        return blinding_value == commited_value

    @staticmethod
    def commitments_of_total_sum(
                    commitments_of_polynomials: List[PolynomialCommitments]
                    ) -> PolynomialCommitments:
        sum_commitments = commitments_of_polynomials[0]
        for commitments in commitments_of_polynomials[1:]:
            sum_commitments += commitments
        return sum_commitments