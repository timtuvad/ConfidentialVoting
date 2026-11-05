"""Пассивная публичная доска: никаких вычислений и интерполяции."""
from .models import Deal


class BulletinBoard:
    def __init__(self):
        self._log: list[Deal] = []

    def post(self, deal: Deal) -> None:
        if deal.receiver != -1:
            raise ValueError('Доска принимает только публичные сообщения')
        self._log.append(deal)

    def read(self) -> tuple[Deal, ...]:
        return tuple(self._log)
