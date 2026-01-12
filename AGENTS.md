# AGENTS.md

## 出力言語
- すべての出力は日本語（回答/ログ/コメント）

## セットアップコマンド
- 依存導入: `uv sync --dev`
- pre-commit: `uv run pre-commit install`

## 実験実行
- 直接実行: `uv run python experiments/expXXX_name/run.py exp=YYY`
- Makefile: `make uv-exp EXP=XXX CFG=YYY`

## 実験ログ
- 実験後に `docs/experiments.md` に 1 行追加（新しい行を上に）
- 自動追記: `make log-exp EXP=XXX CFG=YYY [AUTHOR=...] [RUN_DIR=...]`

## テスト/品質
- フォーマット: `make fmt` (Ruff)
- Lint: `make lint` (Ruff)
- 型チェック: `make type` (Ty)
- 全チェック: `make check` (Lint + Type)
- テスト: `make test`

## リポジトリ固有ルール
- `data/`, `outputs/`, `submissions/` はコミットしない（`.gitignore` 済み）
- Kaggle トークンは `~/.kaggle/kaggle.json` に配置
- `.env` などの秘密情報はコミットしない
- 提出ファイルは `submissions/` に置く（例: `submissions/submission.csv`）

## ディレクトリ構成
- `experiments/`: 実験本体と設定
- `src/`: 再利用コード
- `scripts/`: 補助スクリプト
- `notebooks/`: EDA/試行
- `docs/`: 実験ログ/運用ドキュメント

## 可視化
- 基本実行 (メトリクス + N=200): `uv run scripts/visualize_experiments.py`
- 特定グループ (N=50): `uv run scripts/visualize_experiments.py --group 50`
- 全グループ一括: `uv run scripts/visualize_experiments.py --group-all`
- 提出ファイル指定: `uv run scripts/visualize_experiments.py --submission-path submissions/best.csv`
