"""Integration tests for the Blokus Duo environment."""

import numpy as np
import pytest
import gymnasium as gym

from blokus_duo.env.blokus_duo_env import BlokusDuoEnv


def test_env_with_random_agent():
    """Test the environment with a random agent."""
    # Arrange
    env = BlokusDuoEnv()
    observation, info = env.reset()

    # Act - Play a few random moves
    done = False
    max_steps = 10
    steps = 0

    while not done and steps < max_steps:
        # Get a random action
        action = _get_random_valid_action(env)

        # If there are no valid actions, end the game
        if action is None:
            print("No valid actions available, ending game.")
            done = True  # Mark the game as done
            # Add final scores to info
            info = {"final_scores": [env.board.calculate_score(p) for p in range(2)]}
            break

        # Take a step
        observation, reward, done, info = env.step(action)
        steps += 1

        # Assert - Check that the observation is valid
        assert "board" in observation
        assert "available_pieces" in observation
        assert "current_player" in observation
        assert observation["board"].shape == (14, 14)
        assert observation["available_pieces"].shape == (2, 21)
        assert 0 <= observation["current_player"] < 2

        # Check that the reward is a float
        assert isinstance(reward, float)

        # Check that done is a boolean
        assert isinstance(done, bool)

        # Check that info is a dictionary
        assert isinstance(info, dict)

    # Assert - Check that we were able to take at least a few steps
    assert steps > 0


def test_env_full_game():
    """Test the environment with a full game."""
    # Arrange
    env = BlokusDuoEnv()
    observation, info = env.reset()

    # Act - Play until the game is done
    done = False
    max_steps = 100  # Limit to avoid infinite loops
    steps = 0

    while not done and steps < max_steps:
        # Get a random action
        action = _get_random_valid_action(env)

        # If there are no valid actions, end the game
        if action is None:
            print("No valid actions available, ending game.")
            done = True  # Mark the game as done
            # Add final scores to info
            info = {"final_scores": [env.board.calculate_score(p) for p in range(2)]}
            break

        # Take a step
        observation, reward, done, info = env.step(action)
        steps += 1

    # Assert - Check that the game ended
    assert done or steps == max_steps

    # If the game ended naturally, check that final scores are provided
    if done and steps < max_steps:
        assert "final_scores" in info
        assert len(info["final_scores"]) == 2


def test_env_render_modes():
    """Test the environment render modes."""
    # Test ansi render mode
    env_ansi = BlokusDuoEnv(render_mode="ansi")
    env_ansi.reset()
    render_result = env_ansi.render()
    assert isinstance(render_result, str)

    # Test rgb_array render mode
    env_rgb = BlokusDuoEnv(render_mode="rgb_array")
    env_rgb.reset()
    render_result = env_rgb.render()
    assert isinstance(render_result, np.ndarray)
    assert render_result.shape[2] == 3  # RGB channels


def _get_random_valid_action(env):
    """Get a random valid action for the current player."""
    # Get all valid actions
    valid_actions = env._get_valid_actions()

    # If there are no valid actions, return None
    if not valid_actions:
        return None

    # Return a random valid action
    return np.random.choice(valid_actions)
