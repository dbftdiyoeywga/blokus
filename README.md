# Blokus Duo

Blokus Duo は、テスト駆動開発（TDD）の原則に基づいた強化学習エージェント開発のための環境です。2人用の戦略的ボードゲーム「Blokus Duo」をOpenAI Gym互換環境として実装し、効率的かつ堅牢な強化学習エージェント開発プロセスを確立します。

## 重要: 開発環境について

**このプロジェクトは必ずdevcontainer内で開発を行う必要があります。**

これは以下の理由から絶対に守るべき制約です：
- 環境の一貫性: Python 3.12とすべての依存関係が正しくインストールされた環境を保証
- 再現性の確保: すべての開発者が同一の環境で作業できることを保証
- ツールチェーンの統合: uv、ruff、pyrightなどのツールが正しく設定された環境を提供

## 開発環境のセットアップ

### 前提条件
- Docker
- Visual Studio Code
- Visual Studio Code Remote - Containers 拡張機能

### セットアップ手順

1. リポジトリをクローン
   ```bash
   git clone <repository-url>
   cd blokus
   ```

2. VS Codeでフォルダを開く
   ```bash
   code .
   ```

3. VS Codeが「Reopen in Container」を提案するポップアップを表示します。これをクリックするか、コマンドパレット（F1）から「Remote-Containers: Reopen in Container」を選択します。

4. devcontainerのビルドが完了するまで待ちます。これにより、Python 3.12、uv、pytest、ruffなどの必要なツールがすべてインストールされます。

5. 環境が正しく設定されているか確認するには、以下のコマンドを実行します：
   ```bash
   ./scripts/check_devcontainer.py
   ```
   このスクリプトは、現在の環境がdevcontainer内で実行されているかどうかを確認します。

## プロジェクト構造

```
blokus/
├── pyproject.toml          # プロジェクト設定
├── README.md               # プロジェクト概要
├── .gitignore              # Git除外設定
├── .devcontainer/          # Devcontainer設定
│   ├── devcontainer.json   # Devcontainer設定
│   └── Dockerfile          # Dockerイメージ定義
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
└── memory-bank/            # プロジェクト文書
    ├── projectbrief.md     # プロジェクト概要
    ├── productContext.md   # プロダクトコンテキスト
    ├── activeContext.md    # アクティブコンテキスト
    ├── systemPatterns.md   # システムパターン
    ├── techContext.md      # 技術コンテキスト
    └── progress.md         # 進捗状況
```

## 開発ワークフロー

このプロジェクトはテスト駆動開発（TDD）の原則に従います：

1. **Red**: まず失敗するテストを書く
2. **Green**: テストが通るように最小限の実装をする
3. **Refactor**: コードをリファクタリングして改善する

### テストの実行

Makefileを使用して、環境に応じた適切な方法でテストを実行できます：

```bash
# すべてのテストを実行
make test

# 特定のテストを実行
make test ARGS="tests/unit/test_board.py"

# カバレッジレポートを生成
make test-cov

# 特定のテストでカバレッジレポートを生成
make test-cov ARGS="tests/unit/test_board.py"
```

### コード品質チェック

```bash
# リンティング
make lint

# 型チェック
make typecheck

# すべての検証（テスト、リント、型チェック）を実行
make validate

# 開発環境の確認（単独実行）
make check-env
# または
./scripts/check_devcontainer.py
```

Makefileは自動的に実行環境を検出し、devcontainer内またはDocker Compose経由で適切にコマンドを実行します。

## 環境の使用例

```python
import gymnasium as gym
import blokus_duo

# 環境の作成
env = gym.make('blokus_duo/BlokusDuo-v0')

# 環境のリセット
observation = env.reset()

# ゲームのステップ実行
for _ in range(100):
    action = env.action_space.sample()  # ランダムアクション
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
```

## 貢献ガイドライン

1. **devcontainer内で開発**: すべての開発作業はdevcontainer内で行ってください（`./scripts/check_devcontainer.py`で確認）
2. **TDDの原則に従う**: 実装前にテストを書き、Red-Green-Refactorサイクルを守る
3. **型ヒントを活用**: すべての関数とメソッドに型ヒントを付ける
4. **ruffによるコード品質管理**: コミット前にruffによるチェックを実行する
5. **ドキュメント更新**: 機能追加や変更時にはメモリバンクのドキュメントも更新する
