#!/usr/bin/env python3
"""Example of using StableBaselines3 with Blokus Duo environment."""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, List, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from blokus_duo.env.blokus_duo_env import BlokusDuoEnv


class BlokusDuoGymEnv(gym.Env):
    """Wrapper for BlokusDuoEnv to make it compatible with StableBaselines3.

    This wrapper converts the Dict action space to a flattened Discrete action space,
    and the Dict observation space to a flattened Box observation space.
    """

    def __init__(self, render_mode: Optional[str] = None) -> None:
        """Initialize the environment.

        Args:
            render_mode: The render mode to use (ansi, human, rgb_array)
        """
        # Create the underlying environment
        self.env = BlokusDuoEnv(render_mode=render_mode)

        # Define the action space
        # We flatten the Dict action space to a Discrete action space
        # Each action is a combination of piece_id, rotation, and position
        # Total number of actions: 21 (pieces) * 8 (rotations) * 14*14 (positions) = 32,928
        # Note: Most of these actions will be invalid, but we'll handle that in the step method
        self.action_space = gym.spaces.Discrete(21 * 8 * 14 * 14)

        # Define the observation space
        # We flatten the Dict observation space to a Box observation space
        # Board: 14x14 = 196 elements
        # Available pieces: 2 players * 21 pieces = 42 elements
        # Current player: 1 element
        # Total: 196 + 42 + 1 = 239 elements
        self.observation_space = gym.spaces.Box(
            low=0, high=2, shape=(239,), dtype=np.float32
        )

        # Set the metadata
        self.metadata = self.env.metadata

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Returns:
            observation: The initial observation
            info: Additional information
        """
        observation, info = self.env.reset(**kwargs)

        # Convert the observation to a flattened array
        flattened_obs = self._flatten_observation(observation)

        return flattened_obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: The action to take, as a flattened integer

        Returns:
            observation: The new observation
            reward: The reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Convert the flattened action to a Dict action
        dict_action = self._unflatten_action(action)

        # Take a step in the underlying environment
        observation, reward, done, info = self.env.step(dict_action)

        # Convert the observation to a flattened array
        flattened_obs = self._flatten_observation(observation)

        return flattened_obs, reward, done, info

    def render(self) -> Optional[Any]:
        """Render the environment.

        Returns:
            The rendered image or string, depending on the render mode
        """
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def _flatten_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        """Convert a Dict observation to a flattened array.

        Args:
            observation: The Dict observation

        Returns:
            The flattened observation
        """
        # Flatten the board
        board_flat = observation["board"].flatten()

        # Flatten the available pieces
        available_pieces_flat = observation["available_pieces"].flatten()

        # Convert the current player to a float
        current_player = np.array([float(observation["current_player"])])

        # Concatenate all the flattened arrays
        flattened_obs = np.concatenate([board_flat, available_pieces_flat, current_player])

        return flattened_obs.astype(np.float32)

    def _unflatten_action(self, action: int) -> Dict[str, Any]:
        """Convert a flattened action to a Dict action.

        Args:
            action: The flattened action

        Returns:
            The Dict action
        """
        # Calculate the piece_id, rotation, and position from the flattened action
        piece_id = action // (8 * 14 * 14)
        rotation = (action // (14 * 14)) % 8
        position_idx = action % (14 * 14)
        position = (position_idx // 14, position_idx % 14)

        # Create the Dict action
        dict_action = {
            "piece_id": piece_id,
            "rotation": rotation,
            "position": position,
        }

        return dict_action


def train_agent(total_timesteps: int = 10000) -> PPO:
    """Train a PPO agent on the Blokus Duo environment.

    Args:
        total_timesteps: The total number of timesteps to train for

    Returns:
        The trained PPO agent
    """
    # Create a vectorized environment
    env = make_vec_env(BlokusDuoGymEnv, n_envs=4, vec_env_cls=SubprocVecEnv)

    # Create the agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    # Close the environment
    env.close()

    return model


def evaluate_agent(model: PPO, num_episodes: int = 5) -> None:
    """Evaluate a trained agent on the Blokus Duo environment.

    Args:
        model: The trained agent
        num_episodes: The number of episodes to evaluate for
    """
    # Create the environment
    env = BlokusDuoGymEnv(render_mode="ansi")

    # Evaluate the agent
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        # Reset the environment
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        # Play until the episode is done
        while not done:
            # Get the action from the agent
            action, _ = model.predict(obs, deterministic=True)

            # Take a step in the environment
            obs, reward, done, info = env.step(action)

            # Update the total reward
            total_reward += reward

            # Render the environment
            env.render()

        # Print the final score
        print(f"Episode {episode + 1} finished with reward {total_reward:.2f}")
        print(f"Final scores: {info['final_scores']}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Train the agent
    print("Training the agent...")
    model = train_agent(total_timesteps=10000)

    # Save the agent
    model.save("blokus_duo_ppo")

    # Evaluate the agent
    print("Evaluating the agent...")
    evaluate_agent(model)

    # Load the agent
    # model = PPO.load("blokus_duo_ppo")
