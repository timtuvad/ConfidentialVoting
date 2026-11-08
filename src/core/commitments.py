from typing import Tuple, Iterable
from src.core.environment import CryptographyEnvironment
from src.core.algebra.schnorr_group import SchnorrGroupElement
from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial

class PolynomialCommitments:

    def __init__(self, commitment_vector: Iterable[SchnorrGroupElement]):
        self.__commitment_vector = tuple(commitment_vector)
        if len(self.__commitment_vector) < 1:
            raise ValueError("commitment vector cannot be empty")
        if not all(isinstance(c, SchnorrGroupElement)
                   for c in self.__commitment_vector):
            raise TypeError("all commitments must be instances of "
                            "SchnorrGroupElement")
        order = self.__commitment_vector[0].order
        if any(c.order != order for c in self.__commitment_vector):
            raise ValueError("all commitments must be from the same group")

    @property
    def commitment_vector(self) -> Tuple[SchnorrGroupElement, ...]:
        return self.__commitment_vector

    @property
    def p(self) -> int:
        return self.__commitment_vector[0].order

    def blinding_value_at(self, person_id: F_pElement) -> SchnorrGroupElement:
        if not isinstance(person_id, F_pElement):
            raise TypeError("person_id must be an instance of F_pElement")
        if person_id.p != self.p:
            raise ValueError("person_id must be from the same field as "
                             "the commitments")
        product = SchnorrGroupElement.one(self.p)
        for k, c in enumerate(self.__commitment_vector):
            product *= c**(person_id**k)
        return product

    def __len__(self) -> int:
        """Количество обязательств в векторе."""
        return len(self.__commitment_vector)

    @staticmethod
    def make(f: F_pPolynomial, blinding_polynomial: F_pPolynomial,
             environment: CryptographyEnvironment) -> "PolynomialCommitments":
        """
        Создает вектор обязательств к данному многочлену.

        Args:
            f (F_pPolynomial): Многочлен.
            blinding_polynomial (F_pPolynomial): Многочлен со случайными
                коэффициентами в поле F_p, используемый для создания
                обязательств по схеме Pedersen VSS.
            environment (CryptographyEnvironment): Криптографическое кружение,
                в котором будут создаваться обязательства.
        
        Returns:
            PolynomialCommitments: Обязательства к многочлену f.
        
        Raises:
            TypeError: Если `f` или `blinding_polynomial` определены
                не над тем полем F_p или содержат больше `environment.threshold`
                коэффициентов.
        """

        if not environment.contains_F_pPolynomial(f):
            raise TypeError("f must be a correct polynomial from the "
                            "perspective of the environment")
        if not environment.contains_F_pPolynomial(blinding_polynomial):
            raise TypeError("blinding_polynomial must be a correct "
                            "polynomial from the perspective of the "
                            "environment")
        g0 = environment.g0
        h0 = environment.h0
        commitment_vector = tuple(g0**f[i] * h0**blinding_polynomial[i]
                                  for i in range(environment.threshold))
        return PolynomialCommitments._new(commitment_vector)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PolynomialCommitments):
            return False
        return self.__commitment_vector == other.__commitment_vector

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self.__commitment_vector)

    def __pos__(self):
        return self

    def __neg__(self):
        """
        Возвращает вектор обязательств к противоположному по знаку многочлену.

        Args:
            self (PolynomialCommitments): Обязательства к неизвестному многочлену `f`.
        
        Returns:
            PolynomialCommitments: Обязательства к многочлену `-f`.
        """

        commitment_vector = tuple(~c for c in self.__commitment_vector)
        return PolynomialCommitments._new(commitment_vector)

    def __add__(self, other):
        """
        Возвращает вектор обязательств к сумме двух многочленов.

        Args:
            self (PolynomialCommitments): Обязательства к неизвестному
                многочлену `f`.
            other (PolynomialCommitments): Обязательства к неизвестному
                многочлену `g`.
        
        Returns:
            PolynomialCommitments: Обязательства к многочлену `f + g`.

        Raises:
            TypeError: Если `other` не является объектом `PolynomialCommitments`.
            ValueError: Если длины векторов обязательств не равны.
        """

        if not isinstance(other, PolynomialCommitments):
            raise TypeError("unsupported operand type(s) for +: "
                            f"'{self.__class__.__name__}' and "
                            f"'{other.__class__.__name__}'")
        if len(self.__commitment_vector) != len(other.__commitment_vector):
            raise ValueError(f"cannot add two commitment vectors of different "
                             "lengths")
        pairs_iter = zip(self.__commitment_vector, other.__commitment_vector)
        commitment_vector = tuple(c1*c2 for c1, c2 in pairs_iter)
        return PolynomialCommitments._new(commitment_vector)

    def __sub__(self, other):
        """
        Возвращает вектор обязательств к разности двух многочленов.

        Args:
            self (PolynomialCommitments): Обязательства к неизвестному многочлену `f`.
            other (PolynomialCommitments): Обязательства к неизвестному многочлену `g`.
        
        Returns:
            PolynomialCommitments: Обязательства к многочлену `f - g`.

        Raises:
            TypeError: Если `other` не является объектом `PolynomialCommitments`.
            ValueError: Если длины векторов обязательств не равны.
        """

        if not isinstance(other, PolynomialCommitments):
            raise TypeError("unsupported operand type(s) for -: "
                            f"'{self.__class__.__name__}' and "
                            f"'{other.__class__.__name__}'")
        if len(self.__commitment_vector) != len(other.__commitment_vector):
            raise ValueError(f"cannot subtract two commitment vectors of "
                            "different lengths")
        return self + (-other)

    @classmethod
    def _new(cls, commitment_vector: Tuple[SchnorrGroupElement, ...]
            ) -> "PolynomialCommitments":
        obj = cls.__new__(cls)
        obj.__commitment_vector = commitment_vector
        return obj

    