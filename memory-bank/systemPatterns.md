# システムパターン: Blokus Duo

このドキュメントでは、Blokus Duoプロジェクト固有のシステムパターンと設計原則を定義します。共有メモリバンクの`systemPatterns.md`で定義された一般的なパターンを基に、このプロジェクトに特化した実践方法を記述します。

## Python 3.12でのテスト駆動開発パターン

### テストフレームワークと構造

1. **pytestの活用**
   - テストディスカバリの自動化
   - フィクスチャの活用
   - パラメータ化テスト
   - マーカーによるテストの分類
   - devcontainer環境での一貫したテスト実行

2. **テストディレクトリ構造**
   ```
   tests/
   ├── unit/                  # 単体テスト
   │   ├── test_board.py      # 14x14ボード関連のテスト
   │   ├── test_pieces.py     # ピース関連のテスト
   │   └── test_rules.py      # Blokus Duoルール関連のテスト
   ├── integration/           # 統合テスト
   │   ├── test_env.py        # 環境全体のテスト
   │   ├── test_agent.py      # エージェント連携テスト
   │   └── test_selfplay.py   # 自己対戦テスト
   └── conftest.py            # 共有フィクスチャ
   ```

3. **テスト命名規則**
   - `test_<対象機能>_<状況>_<期待結果>`
   - 例: `test_place_piece_valid_position_returns_true`

### Python 3.12での実装パターン

1. **Red-Green-Refactorサイクル**
   ```python
   # 1. Red: 失敗するテストを書く
   def test_board_is_valid_position_corner_returns_true():
       # Arrange
       board = Board(14, 14)  # Blokus Duoは14x14ボード
       piece = Piece.create('I1')
       position = Position(4, 4)  # Duoのスタート位置

       # Act
       result = board.is_valid_position(piece, position)

       # Assert
       assert result is True

   # 2. Green: 最小限の実装
   def is_valid_position(self, piece, position):
       # 最小限の実装でテストを通す
       return True

   # 3. Refactor: リファクタリング
   def is_valid_position(self, piece, position):
       # より堅牢な実装
       if not self._is_within_bounds(piece, position):
           return False
       if self._overlaps_existing_pieces(piece, position):
           return False
       # Blokus Duoの特殊ルール: 初手は指定された開始位置に配置
       if self.is_first_move and not self._is_at_starting_position(position):
           return False
       return True
   ```

2. **初手制限ルールの実装例（TDDアプローチ）**
   ```python
   # 1. Red: 失敗するテストを書く
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

   # 2. Green: 実装
   def _covers_starting_position(self, piece_shape: PieceArray, position: Position, player: int) -> bool:
       """Check if a piece covers the starting position for a player."""
       start_pos = self.starting_positions[player]

       # Calculate all cells covered by the piece
       for i in range(piece_shape.shape[0]):
           for j in range(piece_shape.shape[1]):
               if piece_shape[i, j] == 1:
                   cell_pos = (position[0] + i, position[1] + j)
                   if cell_pos == start_pos:
                       return True

       return False

   # is_valid_position メソッドの修正
   # 修正前
   if self.is_first_move[player] and not self._is_at_starting_position(position, player):
       return False

   # 修正後
   if self.is_first_move[player] and not self._covers_starting_position(piece_shape, position, player):
       return False

   # 3. Refactor: 環境レベルでのテスト追加
   def test_env_first_move_must_cover_starting_position():
       """Test that the first move must cover the starting position."""
       # Arrange
       env = BlokusDuoEnv()
       env.reset()

       # Create a piece that doesn't cover the starting position (4, 4)
       invalid_action = {
           "piece_id": 0,  # 1x1 piece
           "rotation": 0,
           "position": (3, 3)  # Doesn't cover (4, 4)
       }

       # Create a piece that covers the starting position (4, 4)
       valid_action = {
           "piece_id": 0,  # 1x1 piece
           "rotation": 0,
           "position": (4, 4)  # Covers (4, 4)
       }

       # Act & Assert
       # Invalid action
       obs1, reward1, done1, info1 = env.step(invalid_action)
       assert reward1 < 0  # Negative reward
       assert info1.get("invalid_action", False)  # Invalid action flag

       # Valid action
       obs2, reward2, done2, info2 = env.step(valid_action)
       assert reward2 > 0  # Positive reward
       assert not info2.get("invalid_action", False)  # Not an invalid action
   ```

2. **Python 3.12の型ヒントの活用**
   ```python
   # Python 3.12では型ヒントの構文が改善されています
   class Board:
       def __init__(self, width: int, height: int) -> None:
           self.width = width
           self.height = height
           # Python 3.12では型ヒントがより簡潔になります
           self.cells: list[list[int | None]] = [[None for _ in range(width)] for _ in range(height)]
           self.is_first_move: bool = True
           self.starting_positions: list[tuple[int, int]] = [(4, 4), (9, 9)]  # Blokus Duoの開始位置

       def is_valid_position(self, piece: Piece, position: tuple[int, int]) -> bool:
           # 実装
           pass

       def _is_at_starting_position(self, position: tuple[int, int]) -> bool:
           # Blokus Duoの開始位置チェック
           return position in self.starting_positions
   ```

3. **ruffによるコード品質管理**
   ```python
   # ruffはflake8, black, isortなどの機能を統合したツールです
   # 以下はruffによって自動的にフォーマットされたコードの例です

   from typing import Any
   import numpy as np


   class BlokusDuoGame:
       """Blokus Duoゲームのメインクラス。"""

       def __init__(self, board_size: int = 14) -> None:
           """ゲームを初期化します。

           Args:
               board_size: ボードのサイズ（デフォルトは14x14）
           """
           self.board_size = board_size
           self.board = np.zeros((board_size, board_size), dtype=np.int32)
           self.current_player = 0
           self.game_over = False

       def reset(self) -> None:
           """ゲームをリセットします。"""
           self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
           self.current_player = 0
           self.game_over = False

       def get_valid_moves(self) -> list[tuple[int, tuple[int, int], int]]:
           """現在のプレイヤーの有効な手を取得します。

           Returns:
               有効な手のリスト（ピースID、位置、回転）
           """
           # 実装
           return []
   ```

## 対戦型強化学習環境のテストパターン

### 環境の一貫性テスト

1. **状態遷移の一貫性**
   ```python
   def test_step_same_action_same_state_produces_same_result():
       # Arrange
       env1 = BlokusDuoEnv()
       env2 = BlokusDuoEnv()
       env1.reset(seed=42)
       env2.reset(seed=42)
       action = 0  # 同じ行動

       # Act
       obs1, reward1, done1, info1 = env1.step(action)
       obs2, reward2, done2, info2 = env2.step(action)

       # Assert
       assert np.array_equal(obs1, obs2)
       assert reward1 == reward2
       assert done1 == done2
   ```

2. **シード設定のテスト**
   ```python
   def test_reset_with_same_seed_produces_same_observation():
       # Arrange
       env = BlokusEnv()

       # Act
       obs1 = env.reset(seed=42)
       env.step(0)  # 状態を変更
       obs2 = env.reset(seed=42)

       # Assert
       assert np.array_equal(obs1, obs2)
   ```

### 報酬関数のテスト

1. **基本的な報酬テスト**
   ```python
   def test_reward_placing_larger_piece_gives_higher_reward():
       # Arrange
       env = BlokusDuoEnv()
       env.reset(seed=42)

       # 1マスのピースを配置する行動
       small_piece_action = env._encode_action(piece_id=0, position=(4, 4), rotation=0)

       # 5マスのピースを配置する行動
       large_piece_action = env._encode_action(piece_id=5, position=(4, 4), rotation=0)

       # Act
       _, small_reward, _, _ = env.step(small_piece_action)
       env.reset(seed=42)
       _, large_reward, _, _ = env.step(large_piece_action)

       # Assert
       assert large_reward > small_reward
   ```

2. **対戦型学習の報酬テスト**
   ```python
   def test_reward_game_end_winner_gets_bonus():
       # Arrange
       env = BlokusDuoEnv()
       env.reset()

       # Act - ゲームを終了状態まで進める
       # (テスト用のヘルパー関数を使用)
       observations, rewards, dones, infos = play_game_to_end(env, winning_player=0)

       # Assert
       assert rewards[-1][0] > 0  # 勝者は正の報酬を得る
       assert rewards[-1][1] < 0  # 敗者は負の報酬を得る
   ```

### 対戦型エージェント-環境相互作用のテスト

1. **行動空間のテスト**
   ```python
   def test_action_space_contains_all_valid_moves():
       # Arrange
       env = BlokusDuoEnv()
       env.reset()

       # Act
       action_space = env.action_space

       # Assert
       assert isinstance(action_space, gym.spaces.Discrete)
       # 全ての有効な行動が含まれていることを確認
       assert action_space.n == env._calculate_total_actions()
   ```

2. **無効な行動の処理テスト**
   ```python
   def test_invalid_action_handled_properly():
       # Arrange
       env = BlokusEnv()
       env.reset()

       # 無効な行動（範囲外）
       invalid_action = env.action_space.n + 1

       # Act & Assert
       with pytest.raises(ValueError):
           env.step(invalid_action)
   ```

2. **自己対戦学習のテスト**
   ```python
   def test_selfplay_learning_improves_performance():
       # Arrange
       env = BlokusDuoEnv()
       agent = PPO("MlpPolicy", env)

       # 初期パフォーマンスを測定
       initial_performance = evaluate_agent(agent, env, n_eval_episodes=10)

       # Act - 自己対戦学習を実行
       train_selfplay(agent, env, total_timesteps=10000)

       # 学習後のパフォーマンスを測定
       final_performance = evaluate_agent(agent, env, n_eval_episodes=10)

       # Assert
       assert final_performance > initial_performance
   ```

## OpenAI Gym互換インターフェースの設計パターン

### 環境クラス設計

1. **Blokus Duo環境の基本構造**
   ```python
   import gym
   import numpy as np
   from typing import Tuple, Dict, Any, Optional

   class BlokusDuoEnv(gym.Env):
       metadata = {'render.modes': ['human', 'rgb_array']}

       def __init__(self, board_size: int = 14):
           super().__init__()
           self.board_size = board_size
           self.num_players = 2  # Blokus Duoは2人プレイ

           # 行動空間と観測空間の定義
           self.action_space = gym.spaces.Discrete(self._calculate_total_actions())
           self.observation_space = gym.spaces.Dict({
               'board': gym.spaces.Box(
                   low=0, high=self.num_players,
                   shape=(self.board_size, self.board_size),
                   dtype=np.int32
               ),
               'available_pieces': gym.spaces.Box(
                   low=0, high=1,
                   shape=(self.num_players, 21),
                   dtype=np.int32
               ),
               'current_player': gym.spaces.Discrete(self.num_players)
           })

           # ゲーム状態の初期化
           self.board = None
           self.available_pieces = None
           self.current_player = None
           self.starting_positions = [(4, 4), (9, 9)]  # Blokus Duoの開始位置

       def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
           # 環境のリセット
           if seed is not None:
               np.random.seed(seed)

           # ゲーム状態の初期化
           self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
           self.available_pieces = np.ones((self.num_players, 21), dtype=np.int32)
           self.current_player = 0
           self.first_moves = [True, True]  # 各プレイヤーの初手フラグ

           return self._get_observation()

       def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
           # 行動のデコード
           piece_id, position, rotation = self._decode_action(action)

           # 行動の検証
           if not self._is_valid_action(piece_id, position, rotation):
               return self._get_observation(), -10.0, False, {"invalid_action": True}

           # 行動の実行
           self._place_piece(piece_id, position, rotation)

           # 初手フラグの更新
           if self.first_moves[self.current_player]:
               self.first_moves[self.current_player] = False

           # 報酬の計算
           reward = self._calculate_reward(piece_id)

           # プレイヤー交代
           self.current_player = 1 - self.current_player

           # 終了判定
           done = self._check_game_over()

           # 追加情報
           info = {
               "piece_placed": piece_id,
               "position": position,
               "rotation": rotation,
               "valid_moves_count": len(self._get_valid_moves())
           }

           return self._get_observation(), reward, done, info

       def render(self, mode: str = 'human'):
           # 環境の可視化
           pass

       def _get_observation(self) -> Dict[str, np.ndarray]:
           # 現在の観測を返す
           return {
               'board': self.board.copy(),
               'available_pieces': self.available_pieces.copy(),
               'current_player': np.array(self.current_player)
           }

       def _calculate_total_actions(self) -> int:
           # 可能な行動の総数を計算
           pass
   ```

2. **Blokus Duo用の行動エンコーディング**
   ```python
   def _encode_action(self, piece_id: int, position: tuple[int, int], rotation: int) -> int:
       # ピースID、位置、回転を一つの整数にエンコード
       max_positions = self.board_size * self.board_size
       max_rotations = 8  # 回転と反転の組み合わせ

       return piece_id * max_positions * max_rotations + \
              position[0] * self.board_size * max_rotations + \
              position[1] * max_rotations + \
              rotation

   def _decode_action(self, action: int) -> tuple[int, tuple[int, int], int]:
       # 整数からピースID、位置、回転をデコード
       max_positions = self.board_size * self.board_size
       max_rotations = 8

       piece_id = action // (max_positions * max_rotations)
       remainder = action % (max_positions * max_rotations)

       row = remainder // (self.board_size * max_rotations)
       remainder = remainder % (self.board_size * max_rotations)

       col = remainder // max_rotations
       rotation = remainder % max_rotations

       return piece_id, (row, col), rotation

   def _is_valid_action(self, piece_id: int, position: tuple[int, int], rotation: int) -> bool:
       # 行動の有効性を検証
       # Blokus Duoの特殊ルール: 初手は指定された開始位置に配置
       if self.first_moves[self.current_player] and position != self.starting_positions[self.current_player]:
           return False

       # その他の検証ロジック
       return True
   ```

## Python 3.12とpyrightを活用した型安全な開発パターン

### Python 3.12の型定義ファイル

1. **型定義の集約**
   ```python
   # types.py
   from typing import TypedDict
   import numpy as np

   # 基本型 - Python 3.12では型エイリアスがより簡潔になります
   Position = tuple[int, int]
   Rotation = int  # 0-7: 回転と反転の組み合わせ
   PieceID = int   # 0-20: 21種類のピース
   PlayerID = int  # 0-3: 4人のプレイヤー

   # 複合型
   class GameState(TypedDict):
       board: np.ndarray
       available_pieces: np.ndarray
       current_player: int

   class ActionInfo(TypedDict):
       piece_id: PieceID
       position: Position
       rotation: Rotation

   class RewardInfo(TypedDict):
       piece_size: int
       corner_bonus: float
       blocking_bonus: float
       total: float
   ```

2. **Python 3.12用のpyright設定**
   ```
   # pyrightconfig.json
   {
     "include": [
       "blokus/**/*.py",
       "tests/**/*.py"
     ],
     "exclude": [
       "**/node_modules",
       "**/__pycache__"
     ],
     "reportMissingImports": true,
     "reportMissingTypeStubs": false,
     "pythonVersion": "3.12",
     "typeCheckingMode": "strict"
   }
   ```

## ruffを活用したコード品質管理

### ruffの設定

```toml
# pyproject.toml
[tool.ruff]
# ターゲットPythonバージョン
target-version = "py312"

# 有効にするルール
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "UP",  # pyupgrade
    "N",   # pep8-naming
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "TCH", # flake8-type-checking
    "TID", # flake8-tidy

### 型安全なコード例

1. **クラスとメソッド**
   ```python
   from typing import List, Tuple, Optional
   import numpy as np
   from .types import Position, Rotation, PieceID

   class Piece:
       def __init__(self, id: PieceID, shape: np.ndarray) -> None:
           self.id = id
           self.shape = shape

       @classmethod
       def create(cls, piece_name: str) -> 'Piece':
           # ピース名から形状を生成
           # ...
           return cls(id=0, shape=np.array([[1]]))

       def get_cells(self, position: Position, rotation: Rotation) -> List[Position]:
           # 指定された位置と回転でのセル位置のリストを返す
           rotated_shape = self._rotate(rotation)
           cells: List[Position] = []

           for i in range(rotated_shape.shape[0]):
               for j in range(rotated_shape.shape[1]):
                   if rotated_shape[i, j] == 1:
                       cells.append((position[0] + i, position[1] + j))

           return cells

       def _rotate(self, rotation: Rotation) -> np.ndarray:
           # 回転と反転を適用した形状を返す
           # ...
           return self.shape
   ```

2. **関数とジェネリクス**
   ```python
   from typing import TypeVar, List, Generic, Callable

   T = TypeVar('T')

   class PriorityQueue(Generic[T]):
       def __init__(self, priority_fn: Callable[[T], float]) -> None:
           self.items: List[T] = []
           self.priority_fn = priority_fn

       def push(self, item: T) -> None:
           self.items.append(item)
           self.items.sort(key=self.priority_fn)

       def pop(self) -> T:
           if not self.items:
               raise IndexError("Priority queue is empty")
           return self.items.pop(0)
   ```

このシステムパターンドキュメントは、Blokusプロジェクトの開発における具体的なパターンと実践方法を定義します。共有メモリバンクの`systemPatterns.md`と併せて参照してください。
