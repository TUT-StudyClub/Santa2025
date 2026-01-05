# AGENTS.md

## 出力言語
- すべての出力は日本語（回答/ログ/コメント）

## セットアップコマンド
- 依存導入: `uv sync --dev`
- pre-commit: `uv run pre-commit install`

## 実験実行
- 直接実行: `uv run python experiments/expXXX_name/run.py exp=YYY`
- Makefile: `make uv-exp EXP=XXX CFG=YYY`
- Hydra 設定: `exp=YYY` は `experiments/expXXX_name/exp/YYY.yaml` を選択

## 実験ログ
- 実験後に `docs/experiments.md` に 1 行追加（新しい行を上に）
- 自動追記: `make log-exp EXP=XXX CFG=YYY [AUTHOR=...] [RUN_DIR=...]`

## テスト/品質
- フォーマット: `make fmt`
- Lint: `make lint`
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
