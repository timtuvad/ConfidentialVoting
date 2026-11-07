"""Единственная точка подключения готового алгоритма определения malicious.

После замены тела detect() названия функций в других модулях не меняются.
Детектор не должен читать God Snapshot, чужие полиномы или attack_info.
"""
from __future__ import annotations
from .consensus import Evidence


def detect(observer: int, suspect: int, evidence: Evidence) -> bool:
    """True, если observer обвиняет suspect; False, если не обвиняет.

    В evidence доступны:
      - round_number, active_participants;
      - my_vote, my_polynomials (только мои);
      - received: полученные лично observer приватные доли;
      - my_share и public_shares (текущие значения доски).

    Замени содержимое функции вызовом своего алгоритма.

    Пример (псевдокод):
        return bool(my_existing_detector(
            observer_id=observer,
            target_id=suspect,
            received=evidence.received,
            published=evidence.public_shares,
        ))
    """
    raise NotImplementedError(
        'Подключите свой детектор в threshold_voting/detector_adapter.py:detect(). '
        'До этого выберите ручные MaliciousReport или учебный oracle в Streamlit.'
    )
