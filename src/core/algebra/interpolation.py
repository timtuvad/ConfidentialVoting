from .field import F_pElement
from .polynomial import F_pPolynomial
from ..secret_share import SecretShare

class Interpolation:

    def __init__(self, shares: SecretShare):
        self.__shares = shares

    # TODO : реализовать
    def compute_polynomial(self) -> F_pPolynomial:
        raise NotImplementedError

    # TODO : реализовать
    def compute(self, x) -> SecretShare:
        raise NotImplementedError