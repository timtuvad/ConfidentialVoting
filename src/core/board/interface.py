"""Контракт Board, пригодный для замены InMemoryBoard на BlockchainBoard."""
from __future__ import annotations
from abc import ABC, abstractmethod
from .messages import (BoardEntry, PrivateShare, PublicCommitments,
                       ParticipantTotalShare, MaliciousReport)


class Board(ABC):
    @abstractmethod
    def open_round(self, round_id: int, voter_ids: tuple[int, ...]) -> None:
        """Зафиксировать список голосующих. Board не решает, кого включать."""
        ...

    @abstractmethod
    def post_commitments(self, message: PublicCommitments) -> BoardEntry: ...

    @abstractmethod
    def post_private_share(self, message: PrivateShare) -> BoardEntry: ...

    @abstractmethod
    def post_total_share(self, message: ParticipantTotalShare) -> BoardEntry: ...

    @abstractmethod
    def post_malicious_report(self, message: MaliciousReport) -> BoardEntry: ...

    @abstractmethod
    def get_commitments(self, round_id: int) -> tuple[PublicCommitments, ...]: ...

    @abstractmethod
    def get_private_shares(self, receiver: int, round_id: int) -> tuple[PrivateShare, ...]:
        """Только доли, адресованные receiver (доверие к переданному ID пока модельное)."""
        ...

    @abstractmethod
    def get_total_shares(self, viewer_id: int, round_id: int) -> tuple[ParticipantTotalShare, ...]:
        """Только голосующий данного раунда может запросить s_j=F(j)."""
        ...

    @abstractmethod
    def get_malicious_reports(self, round_id: int) -> tuple[MaliciousReport, ...]: ...

    @abstractmethod
    def get_source_for_report(self, report: MaliciousReport, viewer_id: int) -> (
        PrivateShare | ParticipantTotalShare | None
    ):
        """Для разбора жалобы в пределах раунда; только голосующим."""
        ...

    @abstractmethod
    def read_all(self) -> tuple[BoardEntry, ...]:
        """Привилегированный диагностический обход. Запрещён вне God View."""
        ...
