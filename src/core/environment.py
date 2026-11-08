from src.core.algebra.schnorr_group import SchnorrGroupElement
from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial
from src.core.share import Share
from src.core.verifiable_share import VerifiableShare
from src.core.commitments import PolynomialCommitments

class CryptographyEnvironment:

    def __init__(self, g0: SchnorrGroupElement, h0: SchnorrGroupElement,
                 threshold: int):
        if not (isinstance(g0, SchnorrGroupElement)
                and isinstance(h0, SchnorrGroupElement)):
            raise TypeError("g0 and h0 must be instances of "
                            "SchnorrGroupElement")
        if g0.order != h0.order:
            raise ValueError("g0 and h0 must be in the same group")
        if not (g0.is_generator and h0.is_generator):
            raise ValueError("g0 and h0 must be generators")
        self.__p = g0.order
        self.__g0 = g0
        self.__h0 = h0
        if not (isinstance(threshold, int) and threshold > 0):
            raise ValueError("threshold must be a positive integer, not "
                            f"{threshold}")
        self.__threshold = threshold

    @property
    def p(self) -> int:
        return self.__p

    @property
    def g0(self) -> SchnorrGroupElement:
        return self.__g0

    @property
    def h0(self) -> SchnorrGroupElement:
        return self.__h0

    @property
    def threshold(self) -> int:
        return self.__threshold

    @staticmethod
    def random(p: int, threshold: int) -> "CryptographyEnvironment":
        """
        Генерирует случайное криптографическое окружение с заданным простым \
            p > 2 таким, что q=2p+1 тоже простое, и заданным порогом \
            `threshold`.
        
        В группе Шнорра порядка `p` генерируются различные случайные \
            порождающие элементы `g0` и `h0`, для которых дискретный логарифм\
            `h0` по основанию `g0` вычислительно трудно находим.
        
        Args:
            p (int): простое число
            threshold (int): порог
        
        Returns:
            CryptographyEnvironment: случайное криптографическое окружение
        
        Raises:
            ValueError: если p не является простым числом, таким что число \
                2*p+1 тоже простое, или если p=2
            ValueError: если `threshold` не является положительным целым числом
        """

        g0 = SchnorrGroupElement.random_generator(p)
        if p <= 2:
            raise ValueError("p must be greater than 2")
        h0 = SchnorrGroupElement.random_generator(p)
        while h0 == g0:
            h0 = SchnorrGroupElement.random_generator(p)
        return CryptographyEnvironment(g0, h0, threshold)

    def contains_SchnorrGroupElement(self,
                                     element: SchnorrGroupElement) -> bool:
        return (isinstance(element, SchnorrGroupElement)
                and element.order == self.__p)

    def contains_F_pElement(self, element: F_pElement) -> bool:
        return (isinstance(element, F_pElement)
                and element.p == self.__p)
    
    def contains_F_pPolynomial(self, polynomial: F_pPolynomial) -> bool:
        return (isinstance(polynomial, F_pPolynomial)
                and polynomial.p == self.__p
                and polynomial.degree < self.__threshold)
    
    def contains_Share(self, share: Share) -> bool:
        return (isinstance(share, Share)
                and share.p == self.__p)
    
    def contains_VerifiableShare(self,
                                verifiable_share: VerifiableShare) -> bool:
        return (isinstance(verifiable_share, VerifiableShare)
                and verifiable_share.p == self.__p)
    
    def contains_PolynomialCommitments(self,
                                commitments: PolynomialCommitments) -> bool:
        return (isinstance(commitments, PolynomialCommitments)
                and commitments.p == self.__p
                and len(commitments) == self.__threshold)

    def __eq__(self, other) -> bool:
        if not isinstance(other, CryptographyEnvironment):
            return False
        return (self.__g0 == other.__g0 and self.__h0 == other.__h0
                and self.__threshold == other.__threshold)

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash((self.__g0, self.__h0, self.__threshold))

    def __str__(self) -> str:
        return (f"CryptographyEnvironment(g0={repr(self.__g0)}, "
                f"h0={repr(self.__h0)}, threshold={self.__threshold})")

    def __repr__(self) -> str:
        return str(self)