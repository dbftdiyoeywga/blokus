# Makefile for Blokus Duo project

# 環境チェック
check-env:
	@echo "環境チェックを実行中..."
	@if [ -f ./scripts/check_devcontainer.py ]; then \
		python ./scripts/check_devcontainer.py; \
		if [ $$? -ne 0 ]; then \
			echo "警告: devcontainer環境外で実行されています。"; \
			echo "Docker Compose環境を使用します。"; \
		fi \
	fi

# テスト実行（環境に応じて適切な方法で実行）
test: check-env
	@if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || [ "$$REMOTE_CONTAINERS" = "true" ]; then \
		echo "コンテナ内でテストを実行します..."; \
		python -m pytest $(ARGS); \
	else \
		echo "Docker Compose経由でテストを実行します..."; \
		docker compose exec -T app python -m pytest $(ARGS); \
	fi

# カバレッジレポート付きテスト
test-cov: check-env
	@if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || [ "$$REMOTE_CONTAINERS" = "true" ]; then \
		echo "コンテナ内でカバレッジテストを実行します..."; \
		python -m pytest --cov=blokus_duo $(ARGS); \
	else \
		echo "Docker Compose経由でカバレッジテストを実行します..."; \
		docker compose exec -T app python -m pytest --cov=blokus_duo $(ARGS); \
	fi

# リンティング
lint: check-env
	@if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || [ "$$REMOTE_CONTAINERS" = "true" ]; then \
		echo "コンテナ内でリンティングを実行します..."; \
		ruff check .; \
	else \
		echo "Docker Compose経由でリンティングを実行します..."; \
		docker compose exec -T app ruff check .; \
	fi

# 型チェック
typecheck: check-env
	@if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || [ "$$REMOTE_CONTAINERS" = "true" ]; then \
		echo "コンテナ内で型チェックを実行します..."; \
		pyright; \
	else \
		echo "Docker Compose経由で型チェックを実行します..."; \
		docker compose exec -T app pyright; \
	fi

# すべての検証を実行
validate: test lint typecheck

.PHONY: check-env test test-cov lint typecheck validate
