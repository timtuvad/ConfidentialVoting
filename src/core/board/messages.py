"""Сообщения общего журнала для OHE / Pedersen VSS.

Обязательства и жалобы — публичные. Приватный f_i(j) предназначен только j.
Суммарный s_j=F(j) доступен участникам конкретного раунда, но не посторонним.
Реального шифрования пока нет; payload хранится как VerifiableShare.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.verifiable_share import VerifiableShare
    from src.core.commitments import PolynomialCommitments

ShareBundle = tuple['VerifiableShare', ...]
CommitmentBundle = tuple['PolynomialCommitments', ...]


class ReportKind(str, Enum):
    PRIVATE_SHARE = 'invalid_private_share'
    TOTAL_SHARE = 'invalid_total_share'


@dataclass(frozen=True, slots=True)
class PrivateShare:
    round_id: int
    sender: int
    receiver: int
    value: ShareBundle


@dataclass(frozen=True, slots=True)
class PublicCommitments:
    round_id: int
    sender: int
    value: CommitmentBundle


@dataclass(frozen=True, slots=True)
class ParticipantTotalShare:
    """s_j = F(j): доступно всем голосующим данного раунда, но не публично."""
    round_id: int
    sender: int
    value: ShareBundle


@dataclass(frozen=True, slots=True)
class MaliciousReport:
    """Публичная жалоба со ссылкой на исходное сообщение.

    Для PRIVATE_SHARE раскрывается VerifiableShare, включая blinding_share.
    Для TOTAL_SHARE источник виден только голосующим, раскрывать его не надо.
    """
    round_id: int
    reporter: int
    suspect: int
    kind: ReportKind
    source_message_id: int
    revealed_share: ShareBundle | None = None


BoardMessage = PrivateShare | PublicCommitments | ParticipantTotalShare | MaliciousReport


@dataclass(frozen=True, slots=True)
class BoardEntry:
    message_id: int
    message: BoardMessage
