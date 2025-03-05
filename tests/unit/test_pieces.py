"""Tests for the Piece class."""

import pytest
import numpy as np
from blokus_duo.env.pieces import Piece


def test_piece_initialization():
    """Test that a piece is initialized correctly."""
    # Arrange
    piece_id = 0
    shape = np.array([[1]])

    # Act
    piece = Piece(piece_id, shape)

    # Assert
    assert piece.id == piece_id
    assert np.array_equal(piece.shape, shape)


def test_piece_rotation_0():
    """Test that rotation 0 returns the original shape."""
    # Arrange
    piece = Piece(1, np.array([[1, 1]]))  # I2 piece

    # Act
    rotated_shape = piece.rotate(0)

    # Assert
    assert np.array_equal(rotated_shape, np.array([[1, 1]]))


def test_piece_rotation_1():
    """Test that rotation 1 returns the shape rotated 90 degrees clockwise."""
    # Arrange
    piece = Piece(1, np.array([[1, 1]]))  # I2 piece

    # Act
    rotated_shape = piece.rotate(1)

    # Assert
    assert np.array_equal(rotated_shape, np.array([[1], [1]]))


def test_piece_rotation_2():
    """Test that rotation 2 returns the shape rotated 180 degrees."""
    # Arrange
    piece = Piece(2, np.array([[1, 1, 1]]))  # I3 piece

    # Act
    rotated_shape = piece.rotate(2)

    # Assert
    assert np.array_equal(rotated_shape, np.array([[1, 1, 1]]))  # I3 is symmetric


def test_piece_rotation_4():
    """Test that rotation 4 returns the shape flipped horizontally."""
    # Arrange
    piece = Piece(3, np.array([[1, 0], [1, 1]]))  # L3 piece

    # Act
    rotated_shape = piece.rotate(4)

    # Assert
    assert np.array_equal(rotated_shape, np.array([[0, 1], [1, 1]]))


def test_get_cells():
    """Test that get_cells returns the correct cell positions."""
    # Arrange
    piece = Piece(1, np.array([[1, 1]]))  # I2 piece
    position = (3, 4)
    rotation = 0

    # Act
    cells = piece.get_cells(position, rotation)

    # Assert
    assert cells == [(3, 4), (3, 5)]


def test_get_cells_with_rotation():
    """Test that get_cells returns the correct cell positions with rotation."""
    # Arrange
    piece = Piece(1, np.array([[1, 1]]))  # I2 piece
    position = (3, 4)
    rotation = 1  # 90 degrees clockwise

    # Act
    cells = piece.get_cells(position, rotation)

    # Assert
    assert cells == [(3, 4), (4, 4)]


def test_create_standard_pieces():
    """Test that create_standard_pieces returns all 21 standard pieces."""
    # Act
    from blokus_duo.env.pieces import create_standard_pieces
    pieces = create_standard_pieces()

    # Assert
    assert len(pieces) == 21
    # Check a few specific pieces
    assert np.array_equal(pieces[0].shape, np.array([[1]]))  # I1
    assert np.array_equal(pieces[1].shape, np.array([[1, 1]]))  # I2
    assert np.array_equal(pieces[2].shape, np.array([[1, 1, 1]]))  # I3
