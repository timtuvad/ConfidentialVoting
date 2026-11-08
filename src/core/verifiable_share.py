from src.core.share import Share
from src.core.algebra.field import F_pElement
from src.core.algebra.polynomial import F_pPolynomial

class VerifiableShare(Share):
    """
    Наследуется от класса `Share`, представляя не только значение \
        f(x) = value, но еще и вспомогательную «долю случайности» к нему \
        (согласно Pedersen VSS).

    С объектами `VerifiableShare` (верифицируемыми долями секрета) можно \
        выполнять те же самые арифметические операции (сложение и вычитание),\
        что и с объектами `Share`.

    Если операция производится с двумя `VerifiableShare`, то в \
        результате получается `VerifiableShare`. Если же хотя бы один из них \
        был просто `Share`, то в результате получается `Share`.

    Attributes:
        x (F_pElement): Точка x, в которой известно значение.
        value (F_pElement): Значение f(x) = value.
        blinding_share (F_pElement): Доля случайности к значению f(x) = value.
    """

    def __init__(self, x: F_pElement, value: F_pElement,
                 blinding_share: F_pElement):
        super().__init__(x, value)
        if not isinstance(blinding_share, F_pElement):
            raise TypeError("blinding_share must be an instance of "
                            "F_pElement, not "
                            f"'{blinding_share.__class__.__name__}'")
        if x.p != blinding_share.p:
            raise TypeError("x, value and blinding_share must be in the "
                            "same field")
        self._blinding_share = blinding_share

    @property
    def blinding_share(self) -> F_pElement:
        return self._blinding_share

    @staticmethod
    def make(f: F_pPolynomial, x: F_pElement,
             blinding_polynomial: F_pPolynomial) -> "VerifiableShare":
        return VerifiableShare._new(x, f(x), blinding_polynomial(x))

    def __eq__(self, other) -> bool:
        if isinstance(other, VerifiableShare):
            return (self._x == other._x
                    and self._value == other._value
                    and self._blinding_share == other._blinding_share)
        else:
            return super().__eq__(other)

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return super().__hash__()

    def __str__(self) -> str:
        return (f"f({self._x})={self._value}, "
                f"blinding_share={self._blinding_share}")

    def __repr__(self) -> str:
        return (f"VerifiableShare({self._x}, {self._value}, "
                f"blinding_share={self._blinding_share})")

    def __pos__(self):
        return self
    
    def __neg__(self):
        return VerifiableShare._new(self._x, -self._value,
                                    -self._blinding_share)

    def __add__(self, other):
        if isinstance(other, VerifiableShare):
            if self._x != other._x:
                raise ValueError(f"cannot + with two shares with "
                                 f"different x values: {self} and {other}")
            return VerifiableShare._new(self._x, self._value + other._value,
                                self._blinding_share + other._blinding_share)
        else:
            return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, VerifiableShare):
            if self._x != other._x:
                raise ValueError(f"cannot - with two shares with "
                                 f"different x values: {self} and {other}")
            return VerifiableShare._new(self._x, self._value - other._value,
                                self._blinding_share - other._blinding_share)
        else:
            return super().__sub__(other)

    @classmethod
    def _new(cls, x: F_pElement, value: F_pElement,
             blinding_share: F_pElement) -> "VerifiableShare":
        obj = cls.__new__(cls)
        obj._x = x
        obj._value = value
        obj._blinding_share = blinding_share
        return obj