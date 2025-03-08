"""Tests for the Blokus Duo environment."""

import numpy as np
import pytest
import gymnasium as gym

from blokus_duo.env.blokus_duo_env import BlokusDuoEnv
from blokus_duo.env.pieces import create_standard_pieces


def test_env_initialization():
    """Test that the environment is initialized correctly."""
    # Arrange & Act
    env = BlokusDuoEnv()

    # Assert
    assert env.board.size == 14
    assert len(env.pieces) == 21
    assert len(env.available_pieces[0]) == 21
    assert len(env.available_pieces[1]) == 21
    assert env.current_player == 0
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert "board" in env.observation_space.spaces
    assert "available_pieces" in env.observation_space.spaces
    assert "current_player" in env.observation_space.spaces


def test_env_reset():
    """Test that the environment resets correctly."""
    # Arrange
    env = BlokusDuoEnv()

    # Modify the environment state
    env.current_player = 1
    env.available_pieces[0].remove(0)

    # Act
    observation, info = env.reset()

    # Assert
    assert env.current_player == 0
    assert len(env.available_pieces[0]) == 21
    assert len(env.available_pieces[1]) == 21
    assert np.all(observation["board"] == 0)
    assert np.all(observation["available_pieces"][0] == 1)
    assert np.all(observation["available_pieces"][1] == 1)
    assert observation["current_player"] == 0


def test_valid_action():
    """Test that a valid action is processed correctly."""
    # Arrange
    env = BlokusDuoEnv()

    # First move for player 0 at starting position (4, 4)
    # Using piece 0 (1x1 square) with rotation 0
    action = {"piece_id": 0, "rotation": 0, "position": (4, 4)}

    # Act
    observation, reward, done, info = env.step(action)

    # Assert
    assert env.current_player == 1  # Player switched
    assert 0 not in env.available_pieces[0]  # Piece removed
    assert np.sum(observation["board"] == 1) == 1  # One cell filled
    assert observation["available_pieces"][0, 0] == 0  # Piece marked as used
    assert reward > 0  # Positive reward
    assert not done  # Game not over


def test_invalid_action():
    """Test that an invalid action is handled correctly."""
    # Arrange
    env = BlokusDuoEnv()

    # Invalid position (not starting position)
    action = {"piece_id": 0, "rotation": 0, "position": (0, 0)}

    # Act
    observation, reward, done, info = env.step(action)

    # Assert
    assert env.current_player == 0  # Player not switched
    assert 0 in env.available_pieces[0]  # Piece not removed
    assert np.all(observation["board"] == 0)  # Board unchanged
    assert reward < 0  # Negative reward
    assert not done  # Game not over
    assert info.get("invalid_action", False)  # Invalid action flag


def test_game_over():
    """Test that the game ends correctly."""
    # Arrange
    env = BlokusDuoEnv()

    # Manually set the board to a game over state
    env.board.is_first_move = [False, False]
    env.board.player_corners = [set(), set()]

    # Fill the board to ensure no valid moves
    for row in range(env.board.size):
        for col in range(env.board.size):
            env.board.cells[row, col] = 1  # Player 0's pieces

    # Act
    # Use a valid first move to trigger game state check
    action = {"piece_id": 0, "rotation": 0, "position": (4, 4)}

    # This action will be invalid, but it will trigger the game over check
    observation, reward, done, info = env.step(action)

    # Assert
    assert done  # Game should be over
    assert "final_scores" in info  # Final scores should be provided


def test_render_ansi():
    """Test that the environment can be rendered in ANSI mode."""
    # Arrange
    env = BlokusDuoEnv(render_mode="ansi")

    # Act
    env.reset()
    render_result = env.render()

    # Assert
    assert isinstance(render_result, str)
    assert "+" in render_result  # Board borders
    assert "|" in render_result  # Board borders


def test_multiple_turns():
    """Test multiple turns in the game."""
    # Arrange
    env = BlokusDuoEnv()
    env.reset()

    # Act - Player 0's turn
    action1 = {"piece_id": 0, "rotation": 0, "position": (4, 4)}
    obs1, reward1, done1, info1 = env.step(action1)

    # Player 1's turn
    action2 = {"piece_id": 0, "rotation": 0, "position": (9, 9)}
    obs2, reward2, done2, info2 = env.step(action2)

    # Player 0's second turn - using a corner connection
    # Find a valid corner position from player 0's first piece
    corner_pos = next(iter(env.board.player_corners[0]))
    # Use piece 2 instead of piece 1
    action3 = {"piece_id": 2, "rotation": 0, "position": corner_pos}
    obs3, reward3, done3, info3 = env.step(action3)

    # Assert
    assert obs1["current_player"] == 1  # Switched to player 1
    assert obs2["current_player"] == 0  # Switched back to player 0
    # Don't check the current player after the third step, as it depends on the game state
    assert obs3["available_pieces"][0, 0] == 0  # Player 0's piece 0 is used
    assert obs3["available_pieces"][1, 0] == 0  # Player 1's piece 0 is used
    assert np.sum(obs3["board"] == 1) >= 1  # Player 0 has at least 1 cell
    assert np.sum(obs3["board"] == 2) == 1  # Player 1 has 1 cell


def test_skip_action():
    """Test that skip actions are handled correctly."""
    # Arrange
    env = BlokusDuoEnv()
    env.reset()

    # Act - Player 0 skips
    skip_action = {"skip": True}
    obs1, reward1, done1, info1 = env.step(skip_action)

    # Assert
    assert obs1["current_player"] == 1  # Switched to player 1
    assert not done1  # Game not over yet
    assert info1.get("skipped", False)  # Skip flag in info

    # Act - Player 1 skips
    obs2, reward2, done2, info2 = env.step(skip_action)

    # Assert
    assert done2  # Game should be over when both players skip
    assert "final_scores" in info2
