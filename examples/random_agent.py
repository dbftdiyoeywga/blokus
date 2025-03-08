#!/usr/bin/env python3
"""Example of a random agent playing Blokus Duo."""

import numpy as np
import time
from typing import Dict, Any, List, Optional

from blokus_duo.env.blokus_duo_env import BlokusDuoEnv
from blokus_duo.utils.types import ActionInfo


def random_agent(env: BlokusDuoEnv) -> Dict[str, Any]:
    """Get a random valid action for the current player.

    Args:
        env: The Blokus Duo environment

    Returns:
        A random valid action, or a skip action if there are no valid actions
    """
    # Get all valid actions
    valid_actions = env._get_valid_actions()

    # If there are no valid actions, return a skip action
    if not valid_actions:
        return {"skip": True}

    # Return a random valid action
    return np.random.choice(valid_actions)


def play_game(render_mode: str = "ansi", delay: float = 0.5) -> None:
    """Play a game of Blokus Duo with random agents.

    Args:
        render_mode: The render mode to use (ansi, human, rgb_array)
        delay: The delay between moves in seconds
    """
    # Create the environment
    env = BlokusDuoEnv(render_mode=render_mode)

    # Reset the environment
    observation, info = env.reset()

    # Render the initial state
    env.render()

    # Play until the game is done
    done = False
    total_reward = [0.0, 0.0]

    while not done:
        # Get the current player
        current_player = observation["current_player"]

        # Get a random action
        action = random_agent(env)

        # Print the action
        if action.get("skip", False):
            print(f"Player {current_player} has no valid moves, skipping turn.")
        else:
            print(f"Player {current_player} plays piece {action['piece_id']} "
                  f"with rotation {action['rotation']} at position {action['position']}")

        # Take a step
        observation, reward, done, info = env.step(action)

        # Update the total reward
        total_reward[current_player] += reward

        # Render the new state
        env.render()

        # Print the reward
        print(f"Reward: {reward:.2f}")

        # Wait for a bit
        time.sleep(delay)

    # Print the final scores
    print("Game over!")
    print(f"Final scores: {info['final_scores']}")
    print(f"Total rewards: {total_reward}")


if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Play a game
    play_game("human")
