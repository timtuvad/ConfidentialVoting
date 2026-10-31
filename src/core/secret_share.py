from src.core.algebra.field import F_pElement

# Известное значение многочлена f(x) = polynomial_value
# Сам многочлен f неизвестен
class SecretShare:

    def __init__(self, x: F_pElement, polynomial_value: F_pElement):
        # TODO : добавить проверку на то, что x и polynomial_value принадлежат одному полю
        self.__x = x
        self.__polynomial_value = polynomial_value

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return (self.__x == other.x
                and self.__polynomial_value == other.polynomial_value)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.__x, self.__polynomial_value)

    # TODO : добавить функции алгебраических операций с секретами