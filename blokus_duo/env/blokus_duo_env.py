"""Blokus Duo environment implementation for OpenAI Gym."""

from typing import Dict, List, Optional, Set, Tuple, Union, Any, cast
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from blokus_duo.env.board import Board
from blokus_duo.env.pieces import Piece, create_standard_pieces
from blokus_duo.utils.types import Position, Rotation, PieceID, PlayerID, ActionInfo, GameState, RewardInfo


class BlokusDuoEnv(gym.Env):
    """Blokus Duo environment for OpenAI Gym.

    This environment implements the Blokus Duo board game as a reinforcement learning
    environment compatible with OpenAI Gym.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        """Initialize the Blokus Duo environment.

        Args:
            render_mode: The render mode to use (human, rgb_array, ansi)
        """
        # Initialize the board
        self.board = Board()

        # Initialize the pieces
        self.pieces = create_standard_pieces()

        # Initialize the available pieces for each player
        self.available_pieces: List[Set[PieceID]] = [set(range(21)) for _ in range(2)]

        # Initialize the current player
        self.current_player: PlayerID = 0

        # Track if the last action was a skip
        self.last_action_skip = False

        # Define the observation space
        self.observation_space = spaces.Dict(
            {
                # Board state: 14x14 array (0=empty, 1=player1, 2=player2)
                "board": spaces.Box(low=0, high=2, shape=(14, 14), dtype=np.int32),
                # Available pieces: Each player has 21 pieces (0=used, 1=available)
                "available_pieces": spaces.Box(low=0, high=1, shape=(2, 21), dtype=np.int32),
                # Current player: 0 or 1
                "current_player": spaces.Discrete(2),
            }
        )

        # Define the action space
        # We use a Dict space for better structure and clarity
        self.action_space = spaces.Dict(
            {
                "piece_id": spaces.Discrete(21),  # 21 different pieces
                "rotation": spaces.Discrete(8),  # 8 possible rotations/flips
                "position": spaces.Tuple(
                    (spaces.Discrete(14), spaces.Discrete(14))
                ),  # 14x14 board positions
            }
        )

        # Set the render mode
        self.render_mode = render_mode

        # Initialize the random number generator
        self.np_random = np.random.RandomState()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to its initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: The initial observation
            info: Additional information
        """
        # Reset the random number generator
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Reset the board
        self.board = Board()

        # Reset the available pieces
        self.available_pieces = [set(range(21)) for _ in range(2)]

        # Reset the current player
        self.current_player = 0

        # Reset the skip action flag
        self.last_action_skip = False

        # Get the initial observation
        observation = self._get_observation()

        # Return the observation and info
        return observation, {}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: The action to take, as a dictionary with keys:
                - piece_id: The ID of the piece to place (0-20)
                - rotation: The rotation to apply (0-7)
                - position: The position (row, col) to place the piece
                - skip: If True, the player skips their turn (no valid moves)

        Returns:
            observation: The new observation
            reward: The reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Handle skip action
        if action is None or action.get("skip", False):
            # If the previous action was also a skip, end the game
            if self.last_action_skip:
                done = True
                scores = [self.board.calculate_score(p) for p in range(2)]
                info = {"final_scores": scores, "skipped": True}
                return self._get_observation(), 0.0, done, info

            # Set the skip flag and switch to the next player
            self.last_action_skip = True
            self.current_player = 1 - self.current_player
            return self._get_observation(), 0.0, False, {"skipped": True}

        # Reset the skip flag for normal actions
        self.last_action_skip = False

        # Extract the action components
        piece_id = cast(PieceID, action["piece_id"])
        rotation = cast(Rotation, action["rotation"])
        position = cast(Position, action["position"])

        # Check if the action is valid
        if not self._is_valid_action(piece_id, rotation, position):
            # Check if the game is over
            done = self.board.is_game_over()

            # Prepare the info dictionary
            info = {"invalid_action": True}

            # Add final scores if the game is over
            if done:
                scores = [self.board.calculate_score(p) for p in range(2)]
                info["final_scores"] = scores

            # Invalid action, return negative reward and current observation
            return self._get_observation(), -10.0, done, info

        # Get the piece shape
        piece_shape = self.pieces[piece_id].rotate(rotation)
        piece_size = np.sum(piece_shape)

        # Place the piece on the board
        success = self.board.place_piece(piece_shape, position, self.current_player)

        # If the piece was placed successfully, remove it from the available pieces
        if success:
            self.available_pieces[self.current_player].remove(piece_id)

        # Calculate the reward
        reward_info = self._get_reward(piece_size, action)
        reward = reward_info["total"]

        # Check if the game is over
        done = self.board.is_game_over()

        # Switch to the next player
        self.current_player = 1 - self.current_player

        # If the next player has no valid moves, switch back
        if not self.board.has_valid_moves(self.current_player):
            self.current_player = 1 - self.current_player

            # If both players have no valid moves, the game is over
            if not self.board.has_valid_moves(self.current_player):
                done = True

        # Get the new observation
        observation = self._get_observation()

        # Prepare the info dictionary
        info = {"reward_info": reward_info}

        # Add final scores if the game is over
        if done:
            scores = [self.board.calculate_score(p) for p in range(2)]
            info["final_scores"] = scores

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return observation, reward, done, info

    def render(self) -> Optional[Union[str, np.ndarray]]:
        """Render the environment.

        Returns:
            The rendered image or string, depending on the render mode
        """
        if self.render_mode is None:
            return None

        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            # For human rendering, we would typically use a GUI
            # For simplicity, we'll just print the ANSI representation
            print(self._render_ansi())
            return None
        elif self.render_mode == "rgb_array":
            # For simplicity, we'll return a basic colored array
            # In a real implementation, this would be a proper visualization
            return self._render_rgb_array()

        return None

    def _render_ansi(self) -> str:
        """Render the environment as an ANSI string.

        Returns:
            The ANSI string representation of the board
        """
        # Create the board representation
        board_str = "  "
        for col in range(self.board.size):
            board_str += f"{col:2d}"
        board_str += "\n"

        board_str += "  +" + "-" * (self.board.size * 2) + "+\n"

        for row in range(self.board.size):
            board_str += f"{row:2d}|"
            for col in range(self.board.size):
                cell = self.board.cells[row, col]
                if cell == 0:
                    board_str += " ."
                elif cell == 1:
                    board_str += " X"  # Player 0
                elif cell == 2:
                    board_str += " O"  # Player 1
            board_str += " |\n"

        board_str += "  +" + "-" * (self.board.size * 2) + "+\n"

        # Add player information
        board_str += f"Current player: {self.current_player}\n"
        board_str += f"Player 0 pieces: {sorted(self.available_pieces[0])}\n"
        board_str += f"Player 1 pieces: {sorted(self.available_pieces[1])}\n"

        return board_str

    def _render_rgb_array(self) -> np.ndarray:
        """Render the environment as an RGB array.

        Returns:
            The RGB array representation of the board
        """
        # Create a simple colored array
        # In a real implementation, this would be a proper visualization
        cell_size = 30
        width = self.board.size * cell_size
        height = self.board.size * cell_size
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill the board
        for row in range(self.board.size):
            for col in range(self.board.size):
                cell = self.board.cells[row, col]
                if cell == 0:
                    # Empty cell - white
                    rgb_array[
                        row * cell_size : (row + 1) * cell_size,
                        col * cell_size : (col + 1) * cell_size,
                    ] = [255, 255, 255]
                elif cell == 1:
                    # Player 0 - blue
                    rgb_array[
                        row * cell_size : (row + 1) * cell_size,
                        col * cell_size : (col + 1) * cell_size,
                    ] = [0, 0, 255]
                elif cell == 2:
                    # Player 1 - red
                    rgb_array[
                        row * cell_size : (row + 1) * cell_size,
                        col * cell_size : (col + 1) * cell_size,
                    ] = [255, 0, 0]

        # Draw grid lines
        for i in range(self.board.size + 1):
            # Horizontal lines
            if i < self.board.size:
                rgb_array[i * cell_size, :] = [0, 0, 0]
            # Vertical lines
            if i < self.board.size:
                rgb_array[:, i * cell_size] = [0, 0, 0]

        # Highlight starting positions
        for player, (row, col) in enumerate(self.board.starting_positions):
            color = [0, 0, 200] if player == 0 else [200, 0, 0]
            # Draw a small marker
            marker_size = cell_size // 4
            center_row = row * cell_size + cell_size // 2
            center_col = col * cell_size + cell_size // 2
            rgb_array[
                center_row - marker_size : center_row + marker_size,
                center_col - marker_size : center_col + marker_size,
            ] = color

        return rgb_array

    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation.

        Returns:
            The current observation as a dictionary
        """
        # Get the board state
        board_obs = self.board.cells.copy()

        # Get the available pieces
        available_pieces = np.zeros((2, 21), dtype=np.int32)
        for player in range(2):
            for piece_id in self.available_pieces[player]:
                available_pieces[player, piece_id] = 1

        # Return the observation
        return {
            "board": board_obs,
            "available_pieces": available_pieces,
            "current_player": self.current_player,
        }

    def _is_valid_action(self, piece_id: PieceID, rotation: Rotation, position: Position) -> bool:
        """Check if an action is valid.

        Args:
            piece_id: The ID of the piece to place
            rotation: The rotation to apply
            position: The position to place the piece

        Returns:
            True if the action is valid, False otherwise
        """
        # Check if the piece is available
        if piece_id not in self.available_pieces[self.current_player]:
            return False

        # Get the piece shape
        piece_shape = self.pieces[piece_id].rotate(rotation)

        # Check if the position is valid
        return self.board.is_valid_position(piece_shape, position, self.current_player)

    def _get_reward(self, piece_size: int, action: Dict[str, Any]) -> RewardInfo:
        """Calculate the reward for an action.

        Args:
            piece_size: The size of the placed piece
            action: The action that was taken

        Returns:
            The reward information
        """
        # Basic reward: piece size
        reward = float(piece_size)

        # Bonus: corner connections
        corner_bonus = self._calculate_corner_bonus(action)

        # Bonus: blocking opponent
        blocking_bonus = self._calculate_blocking_bonus(action)

        # Total reward
        total_reward = reward + corner_bonus + blocking_bonus

        return {
            "piece_size": piece_size,
            "corner_bonus": corner_bonus,
            "blocking_bonus": blocking_bonus,
            "total": total_reward,
        }

    def _calculate_corner_bonus(self, action: Dict[str, Any]) -> float:
        """Calculate the corner connection bonus.

        Args:
            action: The action that was taken

        Returns:
            The corner connection bonus
        """
        # For simplicity, we'll just return a fixed bonus
        # In a real implementation, this would be more sophisticated
        return 0.5

    def _calculate_blocking_bonus(self, action: Dict[str, Any]) -> float:
        """Calculate the blocking bonus.

        Args:
            action: The action that was taken

        Returns:
            The blocking bonus
        """
        # For simplicity, we'll just return a fixed bonus
        # In a real implementation, this would be more sophisticated
        return 0.5

    def _get_valid_actions(self) -> List[ActionInfo]:
        """Get all valid actions for the current player.

        Returns:
            A list of valid actions
        """
        valid_actions: List[ActionInfo] = []

        # For each available piece
        for piece_id in self.available_pieces[self.current_player]:
            piece = self.pieces[piece_id]

            # For each rotation
            for rotation in range(8):
                piece_shape = piece.rotate(rotation)

                # For each position on the board
                for row in range(self.board.size):
                    for col in range(self.board.size):
                        position = (row, col)

                        # Check if the position is valid
                        if self.board.is_valid_position(piece_shape, position, self.current_player):
                            valid_actions.append({
                                "piece_id": piece_id,
                                "rotation": rotation,
                                "position": position,
                            })

        return valid_actions

    def close(self) -> None:
        """Close the environment."""
        pass
