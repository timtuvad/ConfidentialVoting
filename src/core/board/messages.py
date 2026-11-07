"""Пассивные сообщения доски для OHE/Shamir. Шифрование выполняется вне Board.

В MVP `PrivateShare.value` содержит открытый вектор. Это НЕ защита приватности;
в будущем перед публикацией он заменяется ciphertext, а дешифрование
происходит на стороне получателя (тип допускает bytes).
"""
from __future__ import annotations
from dataclasses import dataclass

Vector = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrivateShare:
    round_id: int
    sender: int
    receiver: int
    value: Vector | bytes


@dataclass(frozen=True, slots=True)
class PublicShare:
    round_id: int
    sender: int
    value: Vector


@dataclass(frozen=True, slots=True)
class MaliciousReport:
    round_id: int
    reporter: int
    suspect: int
    is_malicious: bool
    evidence_ref: str | None = None


BoardMessage = PrivateShare | PublicShare | MaliciousReport


@dataclass(frozen=True, slots=True)
class BoardEntry:
    message_id: int
    message: BoardMessage
