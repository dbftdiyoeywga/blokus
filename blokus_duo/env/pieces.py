"""Piece implementation for Blokus Duo."""

import numpy as np
from typing import List, Dict
from blokus_duo.utils.types import Position, Rotation, PieceID, PieceArray


class Piece:
    """Piece representation for Blokus Duo.

    A piece is a polyomino shape represented as a 2D numpy array where:
    - 1 represents a cell that is part of the piece
    - 0 represents an empty cell
    """

    def __init__(self, id: PieceID, shape: PieceArray) -> None:
        """Initialize a new piece.

        Args:
            id: The ID of the piece (0-20)
            shape: The shape of the piece as a 2D numpy array
        """
        self.id = id
        self.shape = shape

    def rotate(self, rotation: Rotation) -> PieceArray:
        """Rotate and/or flip the piece.

        Args:
            rotation: The rotation value (0-7)
                0: Original shape
                1: 90 degrees clockwise
                2: 180 degrees
                3: 270 degrees clockwise
                4: Horizontal flip
                5: Horizontal flip + 90 degrees clockwise
                6: Horizontal flip + 180 degrees
                7: Horizontal flip + 270 degrees clockwise

        Returns:
            The rotated/flipped shape
        """
        shape = self.shape.copy()

        # Apply flip if needed (rotations 4-7)
        if rotation >= 4:
            shape = np.fliplr(shape)
            rotation -= 4

        # Apply rotation
        if rotation == 0:
            return shape
        elif rotation == 1:
            return np.rot90(shape, k=3)  # Clockwise rotation
        elif rotation == 2:
            return np.rot90(shape, k=2)
        elif rotation == 3:
            return np.rot90(shape, k=1)

        # Should never reach here
        raise ValueError(f"Invalid rotation value: {rotation}")

    def get_cells(self, position: Position, rotation: Rotation) -> List[Position]:
        """Get the list of cell positions occupied by the piece.

        Args:
            position: The position (row, col) of the top-left corner of the piece
            rotation: The rotation value (0-7)

        Returns:
            A list of (row, col) positions occupied by the piece
        """
        rotated_shape = self.rotate(rotation)
        cells: List[Position] = []

        row_offset, col_offset = position
        for i in range(rotated_shape.shape[0]):
            for j in range(rotated_shape.shape[1]):
                if rotated_shape[i, j] == 1:
                    cells.append((row_offset + i, col_offset + j))

        return cells


def create_standard_pieces() -> Dict[PieceID, Piece]:
    """Create the standard set of 21 Blokus pieces.

    Returns:
        A dictionary mapping piece IDs to Piece objects
    """
    pieces: Dict[PieceID, Piece] = {}

    # 1-cell piece (monomino)
    pieces[0] = Piece(0, np.array([[1]]))  # I1

    # 2-cell piece (domino)
    pieces[1] = Piece(1, np.array([[1, 1]]))  # I2

    # 3-cell pieces (triominoes)
    pieces[2] = Piece(2, np.array([[1, 1, 1]]))  # I3
    pieces[3] = Piece(3, np.array([[1, 0],
                                   [1, 1]]))  # L3

    # 4-cell pieces (tetrominoes)
    pieces[4] = Piece(4, np.array([[1, 1, 1, 1]]))  # I4
    pieces[5] = Piece(5, np.array([[1, 1],
                                   [1, 1]]))  # O4
    pieces[6] = Piece(6, np.array([[1, 0],
                                   [1, 0],
                                   [1, 1]]))  # L4
    pieces[7] = Piece(7, np.array([[1, 0],
                                   [1, 1],
                                   [0, 1]]))  # Z4
    pieces[8] = Piece(8, np.array([[0, 1, 0],
                                   [1, 1, 1]]))  # T4

    # 5-cell pieces (pentominoes)
    pieces[9] = Piece(9, np.array([[1, 1, 1, 1, 1]]))  # I5
    pieces[10] = Piece(10, np.array([[1, 1, 1],
                                     [1, 0, 0],
                                     [1, 0, 0]]))  # L5
    pieces[11] = Piece(11, np.array([[1, 1, 1],
                                     [0, 0, 1],
                                     [0, 0, 1]]))  # Y5
    pieces[12] = Piece(12, np.array([[1, 1, 0],
                                     [0, 1, 0],
                                     [0, 1, 1]]))  # N5
    pieces[13] = Piece(13, np.array([[1, 1, 0],
                                     [0, 1, 1],
                                     [0, 0, 1]]))  # Z5
    pieces[14] = Piece(14, np.array([[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]]))  # X5
    pieces[15] = Piece(15, np.array([[0, 1, 0],
                                     [0, 1, 0],
                                     [1, 1, 1]]))  # T5
    pieces[16] = Piece(16, np.array([[0, 0, 1],
                                     [1, 1, 1],
                                     [1, 0, 0]]))  # U5
    pieces[17] = Piece(17, np.array([[1, 1],
                                     [1, 1],
                                     [1, 0]]))  # P5
    pieces[18] = Piece(18, np.array([[1, 1, 1],
                                     [1, 1, 0]]))  # W5
    pieces[19] = Piece(19, np.array([[1, 1, 0],
                                     [0, 1, 0],
                                     [0, 1, 0]]))  # V5
    pieces[20] = Piece(20, np.array([[1, 1],
                                     [0, 1],
                                     [1, 1]]))  # F5

    return pieces
