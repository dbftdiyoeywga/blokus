"""Type definitions for Blokus Duo."""

from typing import TypedDict, Literal
import numpy as np


# Basic types
Position = tuple[int, int]
Rotation = int  # 0-7: 回転と反転の組み合わせ
PieceID = int   # 0-20: 21種類のピース
PlayerID = int  # 0-1: 2人のプレイヤー


# Complex types
class GameState(TypedDict):
    """Game state representation."""
    board: np.ndarray
    available_pieces: np.ndarray
    current_player: PlayerID


class ActionInfo(TypedDict):
    """Information about an action."""
    piece_id: PieceID
    position: Position
    rotation: Rotation


class RewardInfo(TypedDict):
    """Information about a reward."""
    piece_size: int
    corner_bonus: float
    blocking_bonus: float
    total: float


# Observation space keys
ObservationType = Literal["board", "available_pieces", "current_player"]
