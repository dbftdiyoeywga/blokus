# 技術コンテキスト: Blokus Duo

このドキュメントでは、Blokus Duoプロジェクトの技術的な基盤、制約、および開発環境について定義します。共有メモリバンクの`techContext.md`と併せて参照してください。

## 開発環境

### Python環境

- **Python バージョン**: 3.12
- **仮想環境管理**: devcontainer内でuvを使用
- **パッケージ管理**: uv
- **コード品質ツール**:
  - ruff: リンター、フォーマッター、インポート順序の最適化（Black, isort, flake8の代替）
  - pyright: 静的型チェック
  - pytest: テストフレームワーク
  - pytest-cov: コードカバレッジ測定

### 開発ツール

- **エディタ/IDE**: VSCode
  - Python拡張機能
  - Pylance (pyright)
  - Ruff拡張機能
  - Test Explorer
  - Git 統合
  - Dev Containers拡張機能
- **バージョン管理**: Git
- **CI/CD**: GitHub Actions
  - 自動テスト実行
  - 型チェック
  - ruffによるコード品質チェック
  - コードカバレッジレポート

## 技術スタック

### コア技術

- **Python 3.12**: 単一言語での実装
- **NumPy**: 数値計算と配列操作
- **OpenAI Gym**: 強化学習環境インターフェース
- **StableBaselines3**: 強化学習アルゴリズム実装
- **PyTorch**: ニューラルネットワーク（StableBaselines3のバックエンド）

### 補助ライブラリ

- **Matplotlib/Seaborn**: データ可視化
- **Pandas**: データ分析
- **tqdm**: プログレスバー
- **PyYAML**: 設定ファイル管理
- **Click/Typer**: CLIインターフェース

### 可視化ツール（オプション）

- **Pygame**: 簡易的なゲーム状態可視化
- **Jupyter Notebook**: 実験と分析
- **Tensorboard**: 学習曲線の可視化

## プロジェクト構造

```
blokus/
├── pyproject.toml          # プロジェクト設定
├── README.md               # プロジェクト概要
├── .gitignore              # Git除外設定
├── .devcontainer/          # Devcontainer設定
│   ├── devcontainer.json   # Devcontainer設定
│   └── Dockerfile          # Dockerイメージ定義
├── .github/                # GitHub Actions設定
│   └── workflows/
│       └── ci.yml          # CI設定
├── blokus_duo/             # メインパッケージ
│   ├── __init__.py
│   ├── env/                # 環境実装
│   │   ├── __init__.py
│   │   ├── blokus_duo_env.py # OpenAI Gym環境
│   │   ├── board.py        # 14x14ボード実装
│   │   ├── pieces.py       # ピース定義
│   │   └── rules.py        # Blokus Duoルール
│   ├── agents/             # エージェント実装
│   │   ├── __init__.py
│   │   ├── models.py       # カスタムモデル
│   │   └── selfplay.py     # 自己対戦学習
│   ├── utils/              # ユーティリティ
│   │   ├── __init__.py
│   │   ├── types.py        # 型定義
│   │   └── visualization.py # 可視化ツール
│   └── config/             # 設定
│       ├── __init__.py
│       └── default.yaml    # デフォルト設定
├── tests/                  # テスト
│   ├── __init__.py
│   ├── unit/               # 単体テスト
│   │   ├── test_board.py
│   │   ├── test_pieces.py
│   │   └── test_rules.py
│   ├── integration/        # 統合テスト
│   │   ├── test_env.py
│   │   ├── test_agent.py
│   │   └── test_selfplay.py # 自己対戦テスト
│   └── conftest.py         # テスト共通設定
├── examples/               # 使用例
│   ├── train_agent.py      # エージェント学習スクリプト
│   ├── train_selfplay.py   # 自己対戦学習スクリプト
│   └── evaluate_agent.py   # エージェント評価スクリプト
└── notebooks/              # Jupyter Notebooks
    ├── exploration.ipynb   # 環境探索
    └── visualization.ipynb # 結果可視化
```

## 依存関係管理

### uvによる依存関係管理

```bash
# 依存関係のインストール
uv pip install -e ".[dev]"

# 新しい依存関係の追加
uv pip install numpy

# 開発依存関係の追加
uv pip install --dev pytest

# 依存関係の更新
uv pip sync requirements.txt
```

### 必須依存関係

```toml
[project]
name = "blokus_duo"
version = "0.1.0"
description = "Blokus Duo environment for reinforcement learning"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26.0",
    "gymnasium>=0.29.0",
    "stable-baselines3>=2.0.0",
    "torch>=2.0.0",
    "pyyaml>=6.0",
    "typer>=0.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "pandas>=2.1.0",
]
```

## コード品質管理

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
    "TID", # flake8-tidy-imports
    "RUF", # ruff-specific rules
]

# 無視するルール
ignore = []

# 行の長さ
line-length = 88

# 自動修正を有効にする
fix = true

# isort設定
[tool.ruff.isort]
known-first-party = ["blokus_duo"]
```

## 型ヒントと静的解析

### Python 3.12の型ヒント方針

- すべての関数とメソッドに型ヒントを付ける
- 複雑な型は`types.py`に集約
- Python 3.12の新しい型ヒント構文を活用
  - `list[T]` 代わりに `List[T]`
  - `dict[K, V]` 代わりに `Dict[K, V]`
  - `int | None` 代わりに `Optional[int]`
- `TypedDict`を活用した構造化データの型定義
- ジェネリクスを活用した汎用コンポーネント

### pyrightの設定

```json
{
  "include": [
    "blokus_duo/**/*.py",
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

## Devcontainer設定

```json
// .devcontainer/devcontainer.json
{
  "name": "Blokus Duo Development",
  "image": "mcr.microsoft.com/devcontainers/python:3.12",
  "postCreateCommand": "pip install uv && uv pip install -e '.[dev]'",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff"
      ],
      "settings": {
        "python.linting.enabled": true,
        "python.linting.ruffEnabled": true,
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
          "source.fixAll.ruff": true,
          "source.organizeImports.ruff": true
        }
      }
    }
  }
}
```

## OpenAI Gym環境設計

### Blokus Duo観測空間

```python
observation_space = gym.spaces.Dict({
    'board': gym.spaces.Box(
        low=0, high=2,  # 2人プレイ
        shape=(14, 14),  # 14x14ボード
        dtype=np.int32
    ),
    'available_pieces': gym.spaces.Box(
        low=0, high=1,
        shape=(2, 21),  # 2人プレイ
        dtype=np.int32
    ),
    'current_player': gym.spaces.Discrete(2)  # 2人プレイ
})
```

### Blokus Duo行動空間

```python
# 離散行動空間
# ピースID × 位置 × 回転の組み合わせ
action_space = gym.spaces.Discrete(21 * 14 * 14 * 8)
```

### Blokus Duo報酬設計

- **基本報酬**: 配置したピースのサイズに比例
- **ボーナス報酬**: 特定の状況（コーナー占有、相手の動きを妨げるなど）
- **終了報酬**: ゲーム終了時の最終スコア差に基づく（勝者に正、敗者に負の報酬）
- **ペナルティ**: 無効な行動に対するペナルティ

## 対戦型強化学習

### 自己対戦学習

```python
def train_selfplay(
    agent: BaseAlgorithm,
    env: BlokusDuoEnv,
    total_timesteps: int = 100000,
    n_eval_episodes: int = 10,
) -> BaseAlgorithm:
    """自己対戦学習を行う。

    Args:
        agent: 学習するエージェント
        env: Blokus Duo環境
        total_timesteps: 総ステップ数
        n_eval_episodes: 評価エピソード数

    Returns:
        学習済みエージェント
    """
    # 実装
    return agent
```

### StableBaselines3連携

#### 対応アルゴリズム

- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor Critic)
- DQN (Deep Q-Network)
- SAC (Soft Actor-Critic)

#### カスタムポリシーネットワーク

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy

class BlokusDuoCNN(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 14 * 14, feature_dim)  # 14x14ボード

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class CustomBlokusDuoPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # カスタム実装
```

#### 学習設定

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=1_000_000)
model.save("blokus_duo_ppo")
```

## パフォーマンス最適化

### 計算効率

- NumPyベクトル化操作の活用
- 効率的なボード表現（ビットボードなど）
- キャッシュを活用した合法手生成
- 並列環境での学習
- Python 3.12の最適化機能の活用

### メモリ効率

- 観測の効率的な表現
- 状態のコピーを最小限に抑える
- 大きな行動空間の効率的な処理

## 可視化ツール

### Blokus Duoゲーム状態の可視化

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_board(board: np.ndarray, figsize: tuple[int, int] = (8, 8)) -> plt.Figure:
    """ボード状態を可視化する"""
    plt.figure(figsize=figsize)
    cmap = plt.cm.get_cmap('tab10', 3)  # 3色（プレイヤー2色+空白）
    plt.imshow(board, cmap=cmap, vmin=-0.5, vmax=2.5)
    plt.colorbar(ticks=range(3), label='Player')
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.title('Blokus Duo Board State')

    # 開始位置をマーク
    plt.plot(4, 4, 'rx', markersize=10)  # プレイヤー1の開始位置
    plt.plot(9, 9, 'bx', markersize=10)  # プレイヤー2の開始位置

    plt.tight_layout()
    return plt.gcf()
```

### 対戦結果の可視化

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def plot_match_results(
    results: List[dict],
    figsize: tuple[int, int] = (12, 8)
) -> plt.Figure:
    """対戦結果を可視化する"""
    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 勝率の推移
    sns.lineplot(x='episode', y='win_rate', data=df, ax=ax1)
    ax1.set_title('Win Rate over Episodes')
    ax1.set_ylabel('Win Rate')
    ax1.set_xlabel('Episode')

    # スコア分布
    sns.histplot(x='score_diff', data=df, ax=ax2)
    ax2.set_title('Score Difference Distribution')
    ax2.set_xlabel('Score Difference (Player 1 - Player 2)')

    plt.tight_layout()
    return fig
```

## 技術的制約

1. **計算リソース**
   - 学習には大量の計算リソースが必要
   - 効率的なシミュレーションが重要
   - Python 3.12の最適化機能を活用

2. **行動空間の大きさ**
   - 大きな離散行動空間（21ピース × 14x14ボード × 回転）
   - 効率的な行動表現と探索が必要

3. **学習の安定性**
   - 対戦型強化学習の収束性の課題
   - 自己対戦学習の安定性
   - 適切なハイパーパラメータ調整が必要

4. **評価の複雑さ**
   - 2人対戦ゲームの評価は難しい
   - 自己対戦と外部エージェントとの対戦のバランス

5. **開発環境の一貫性**
   - devcontainer環境の再現性確保
   - uvによる依存関係管理の一貫性
   - ruffによるコード品質の一貫性

このテクニカルコンテキストは、Blokus Duoプロジェクトの技術的な基盤と制約を定義し、一貫した技術的アプローチを確保します。共有メモリバンクの`techContext.md`と併せて参照してください。
