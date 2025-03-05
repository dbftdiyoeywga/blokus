"""Type definitions for Blokus Duo."""

from typing import Literal, TypedDict, Any, TypeVar, Generic

import numpy as np
from numpy.typing import NDArray

# Type variables for numpy arrays
DType = TypeVar('DType')

# Basic types
Position = tuple[int, int]
Rotation = int  # 0-7: 回転と反転の組み合わせ
PieceID = int   # 0-20: 21種類のピース
PlayerID = int  # 0-1: 2人のプレイヤー

# Numpy array types
BoardArray = NDArray[np.int32]  # Board representation
PieceArray = NDArray[np.int32]  # Piece shape representation


# Complex types
class GameState(TypedDict):
    """Game state representation."""
    board: BoardArray
    available_pieces: NDArray[np.int32]
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
