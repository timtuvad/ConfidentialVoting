"""Абстрактная доска и её локальная реализация; заменяемый storage backend."""
from .interface import Board
from .memory import InMemoryBoard
from .messages import BoardEntry, PrivateShare, PublicShare, MaliciousReport

__all__ = ['Board', 'InMemoryBoard', 'BoardEntry', 'PrivateShare', 'PublicShare', 'MaliciousReport']
