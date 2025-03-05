"""Tests for the Blokus Duo game rules."""

import numpy as np
import pytest

from blokus_duo.env.board import Board
from blokus_duo.env.pieces import Piece, create_standard_pieces


def test_corner_to_corner_connection_valid():
    """Test that a piece can be placed with a corner-to-corner connection."""
    # Arrange
    board = Board()
    # Place first piece at starting position for player 0
    first_piece = np.array([[1]])
    board.place_piece(first_piece, (4, 4), 0)

    # Add a corner position manually for testing
    board.player_corners[0].add((2, 2))

    # Second piece with corner-to-corner connection
    second_piece = np.array([[1, 1], [1, 0]])
    position = (2, 2)  # Corner-to-corner with first piece

    # Act
    result = board.is_valid_position(second_piece, position, 0)

    # Assert
    assert result is True


def test_corner_to_corner_connection_invalid():
    """Test that a piece cannot be placed without a corner-to-corner connection."""
    # Arrange
    board = Board()
    # Place first piece at starting position for player 0
    first_piece = np.array([[1]])
    board.place_piece(first_piece, (4, 4), 0)

    # Second piece without corner-to-corner connection
    second_piece = np.array([[1, 1], [1, 0]])
    position = (6, 6)  # No corner-to-corner with first piece

    # Act
    result = board.is_valid_position(second_piece, position, 0)

    # Assert
    assert result is False


def test_no_edge_to_edge_connection():
    """Test that a piece cannot be placed with an edge-to-edge connection to own piece."""
    # Arrange
    board = Board()
    # Place first piece at starting position for player 0
    first_piece = np.array([[1]])
    board.place_piece(first_piece, (4, 4), 0)

    # Second piece with edge-to-edge connection
    second_piece = np.array([[1, 1], [1, 0]])
    position = (3, 4)  # Edge-to-edge with first piece

    # Act
    result = board.is_valid_position(second_piece, position, 0)

    # Assert
    assert result is False


def test_no_overlap_with_own_pieces():
    """Test that a piece cannot overlap with own pieces."""
    # Arrange
    board = Board()
    # Place first piece at starting position for player 0
    first_piece = np.array([[1]])
    board.place_piece(first_piece, (4, 4), 0)

    # Second piece overlapping with first piece
    second_piece = np.array([[1, 1], [1, 0]])
    position = (3, 3)  # Overlaps with first piece

    # Act
    result = board.is_valid_position(second_piece, position, 0)

    # Assert
    assert result is False


def test_can_place_adjacent_to_opponent_pieces():
    """Test that a piece can be placed adjacent to opponent pieces."""
    # Arrange
    board = Board()
    # Place piece for player 0
    board.place_piece(np.array([[1]]), (4, 4), 0)
    # Place piece for player 1
    board.place_piece(np.array([[1]]), (9, 9), 1)

    # Add a corner position manually for testing
    board.player_corners[0].add((7, 9))

    # Player 0's second piece adjacent to player 1's piece
    second_piece = np.array([[1, 1], [1, 0]])
    position = (7, 9)  # Adjacent to player 1's piece

    # Act
    result = board.is_valid_position(second_piece, position, 0)

    # Assert
    assert result is True


def test_game_over_no_valid_moves():
    """Test that the game is over when a player has no valid moves."""
    # Arrange
    board = Board(size=5)  # Smaller board for testing

    # Set up the board state manually
    board.is_first_move = [False, False]  # Both players have already moved
    board.player_corners = [set(), set()]  # No corners available

    # Fill the board completely
    for i in range(5):
        for j in range(5):
            board.cells[i, j] = 1  # Player 0's pieces

    # Act & Assert
    assert board.has_valid_moves(0) is False
    assert board.has_valid_moves(1) is False
    assert board.is_game_over() is True


def test_calculate_score():
    """Test that the score is calculated correctly."""
    # Arrange
    board = Board()

    # Manually set cells for testing
    board.cells[4, 4] = 1  # Player 0
    board.cells[2, 2] = 1  # Player 0
    board.cells[2, 3] = 1  # Player 0
    board.cells[9, 9] = 2  # Player 1

    # Act
    score_0 = board.calculate_score(0)
    score_1 = board.calculate_score(1)

    # Assert
    assert score_0 == 3  # 1 + 2 = 3
    assert score_1 == 1  # 1
