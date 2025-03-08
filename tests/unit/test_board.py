"""Tests for the Board class."""

import numpy as np

from blokus_duo.env.board import Board


def test_board_initialization(board_size):
    """Test that the board is initialized correctly."""
    # Arrange & Act
    board = Board(board_size)

    # Assert
    assert board.size == board_size
    assert board.cells.shape == (board_size, board_size)
    assert np.all(board.cells == 0)
    assert board.is_first_move == [True, True]
    assert board.starting_positions == [(4, 4), (9, 9)]


def test_is_within_bounds_valid_position(board_size):
    """Test that a position within the board bounds is valid."""
    # Arrange
    board = Board(board_size)
    position = (5, 5)
    piece_shape = np.array([[1, 1], [1, 0]])  # L-shaped piece

    # Act
    result = board._is_within_bounds(piece_shape, position)

    # Assert
    assert result is True


def test_is_within_bounds_invalid_position(board_size):
    """Test that a position outside the board bounds is invalid."""
    # Arrange
    board = Board(board_size)
    position = (board_size - 1, board_size - 1)
    piece_shape = np.array([[1, 1], [1, 0]])  # L-shaped piece

    # Act
    result = board._is_within_bounds(piece_shape, position)

    # Assert
    assert result is False


def test_is_at_starting_position_valid(board_size):
    """Test that a position at a starting position is valid for the first move."""
    # Arrange
    board = Board(board_size)
    position = (4, 4)  # Starting position for player 0

    # Act
    result = board._is_at_starting_position(position, 0)

    # Assert
    assert result is True


def test_is_at_starting_position_invalid(board_size):
    """Test that a position not at a starting position is invalid for the first move."""
    # Arrange
    board = Board(board_size)
    position = (5, 5)  # Not a starting position

    # Act
    result = board._is_at_starting_position(position, 0)

    # Assert
    assert result is False


def test_covers_starting_position_valid(board_size):
    """Test that a piece covering the starting position is valid."""
    # Arrange
    board = Board(board_size)
    position = (3, 3)  # Position where the piece will cover (4, 4)
    piece_shape = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])  # Piece that will cover (4, 4)
    player = 0

    # Act
    result = board._covers_starting_position(piece_shape, position, player)

    # Assert
    assert result is True


def test_covers_starting_position_invalid(board_size):
    """Test that a piece not covering the starting position is invalid."""
    # Arrange
    board = Board(board_size)
    position = (3, 3)  # Position where the piece won't cover (4, 4)
    piece_shape = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])  # Piece that won't cover (4, 4)
    player = 0

    # Act
    result = board._covers_starting_position(piece_shape, position, player)

    # Assert
    assert result is False


def test_is_valid_position_first_move_at_starting_position(board_size):
    """Test that the first move at a starting position is valid."""
    # Arrange
    board = Board(board_size)
    position = (4, 4)  # Starting position for player 0
    piece_shape = np.array([[1]])  # Single cell piece
    player = 0

    # Act
    result = board.is_valid_position(piece_shape, position, player)

    # Assert
    assert result is True


def test_is_valid_position_first_move_not_at_starting_position(board_size):
    """Test that the first move not at a starting position is invalid."""
    # Arrange
    board = Board(board_size)
    position = (5, 5)  # Not a starting position
    piece_shape = np.array([[1]])  # Single cell piece
    player = 0

    # Act
    result = board.is_valid_position(piece_shape, position, player)

    # Assert
    assert result is False


def test_is_valid_position_first_move_covers_starting_position(board_size):
    """Test that the first move covering the starting position is valid."""
    # Arrange
    board = Board(board_size)
    position = (3, 3)  # Position where the piece will cover (4, 4)
    piece_shape = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])  # Piece that will cover (4, 4)
    player = 0

    # Act
    result = board.is_valid_position(piece_shape, position, player)

    # Assert
    assert result is True


def test_is_valid_position_first_move_not_covers_starting_position(board_size):
    """Test that the first move not covering the starting position is invalid."""
    # Arrange
    board = Board(board_size)
    position = (3, 3)  # Position where the piece won't cover (4, 4)
    piece_shape = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])  # Piece that won't cover (4, 4)
    player = 0

    # Act
    result = board.is_valid_position(piece_shape, position, player)

    # Assert
    assert result is False


def test_place_piece_valid_position(board_size):
    """Test that a piece can be placed at a valid position."""
    # Arrange
    board = Board(board_size)
    position = (4, 4)  # Starting position for player 0
    piece_shape = np.array([[1]])  # Single cell piece
    player = 0

    # Act
    result = board.place_piece(piece_shape, position, player)

    # Assert
    assert result is True
    assert board.cells[4, 4] == 1  # Player 0's piece
    assert board.is_first_move[0] is False  # No longer first move for player 0


def test_place_piece_invalid_position(board_size):
    """Test that a piece cannot be placed at an invalid position."""
    # Arrange
    board = Board(board_size)
    position = (5, 5)  # Not a starting position
    piece_shape = np.array([[1]])  # Single cell piece
    player = 0

    # Act
    result = board.place_piece(piece_shape, position, player)

    # Assert
    assert result is False
    assert board.cells[5, 5] == 0  # No piece placed
    assert board.is_first_move[0] is True  # Still first move for player 0
