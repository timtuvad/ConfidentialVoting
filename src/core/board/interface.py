"""Контракт хранения; любая будущая BlockchainBoard реализует этот API."""
from abc import ABC, abstractmethod
from .messages import BoardEntry, PrivateShare, PublicShare, MaliciousReport


class Board(ABC):
    @abstractmethod
    def post_private_share(self, share: PrivateShare) -> BoardEntry: ...

    @abstractmethod
    def post_public_share(self, share: PublicShare) -> BoardEntry: ...

    @abstractmethod
    def post_malicious_report(self, report: MaliciousReport) -> BoardEntry: ...

    @abstractmethod
    def get_private_shares(self, receiver: int, round_id: int) -> tuple[PrivateShare, ...]: ...

    @abstractmethod
    def get_public_shares(self, round_id: int) -> tuple[PublicShare, ...]: ...

    @abstractmethod
    def get_malicious_reports(self, round_id: int) -> tuple[MaliciousReport, ...]: ...

    @abstractmethod
    def read_all(self) -> tuple[BoardEntry, ...]:
        """Только для отладки/God View: в MVP содержит открытые приватные доли."""
        ...
