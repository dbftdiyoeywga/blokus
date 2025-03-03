"""Pytest configuration for Blokus Duo tests."""

import pytest
import numpy as np


@pytest.fixture
def seed():
    """Return a fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def board_size():
    """Return the standard board size for Blokus Duo."""
    return 14


@pytest.fixture
def empty_board(board_size):
    """Return an empty board for testing."""
    return np.zeros((board_size, board_size), dtype=np.int32)


@pytest.fixture
def starting_positions():
    """Return the starting positions for Blokus Duo."""
    return [(4, 4), (9, 9)]
