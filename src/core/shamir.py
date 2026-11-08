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
            raise TypeError("environment must be an instance of "
                            "CryptographyEnvironment")
        if not environment.contains_F_pElement(person_id):
            raise TypeError("person_id must be an element of F_p from the "
                            "environment")
        if not environment.contains_F_pElement(secret):
            raise TypeError("secret must be an element of F_p from the "
                            "environment")
        
        self.__environment = environment
        self.__person_id = person_id
        self.__secret = secret
        self.__personal_polynomial = F_pPolynomial.random(self.__person_id.p,
                                            degree_bound=self.__threshold-1,
                                            least_coefficient=self.__secret)
        self.__pedersen_vss = PedersenVSS(self.__environment,
                                          self.__personal_polynomial)

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

    def verifiable_share_to(self, other_id: F_pElement) -> VerifiableShare:
        return self.__pedersen_vss.verifiable_share_to(other_id)

    def verify(self, verifiable_share: VerifiableShare,
                   public_commitments: PolynomialCommitments) -> bool:
        return self.__pedersen_vss.verify(verifiable_share, public_commitments)

    @staticmethod
    def commitments_of_total_sum(
                    commitments_of_polynomials: List[PolynomialCommitments]
                    ) -> PolynomialCommitments:
        return PedersenVSS.commitments_of_total_sum(commitments_of_polynomials)

    @staticmethod
    def total_sum_at_participant(verifiable_shares: List[VerifiableShare]
                                 ) -> VerifiableShare:
        sum_share = verifiable_shares[0]
        for share in verifiable_shares[1:]:
            sum_share += share
        return sum_share

    @staticmethod
    def sum_of_secrets(shares_of_sum: List[Share],
                       threshold: int) -> F_pElement:
        interpolation = LagrangeInterpolation(shares_of_sum)
        if len(interpolation.shares) < threshold:
            raise ValueError("not enough shares to sum secrets")
        p = shares_of_sum[0].x.p
        return interpolation(F_pElement.zero(p))