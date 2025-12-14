## 概要
Kaggle の Santa コンペ用チームリポジトリ（ローカル実行が主）

## クイックスタート
### uv（推奨）
```bash
uv sync --dev
pre-commit install
```

※ Kaggle 環境に寄せたい場合は Python 3.11.13 推奨（`.python-version`）。

### venv + pip
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pre-commit install
```

## 実験の回し方（Hydra 例）
```bash
python experiments/exp000_sample/run.py exp=000
python experiments/exp000_sample/run.py exp=001 exp.debug=true
```

## Docker（任意 / Kaggle に寄せたい人向け）
```bash
make docker-build
make docker-jupyter
# CPU の場合: make docker-build CPU=1
```

## Kaggle Dataset へのアップロード（任意）
```bash
python tools/upload_dataset.py --dataset <username>/<dataset-slug> --src-dir outputs/runs/exp000_sample/000
```

## 運用ルール（最小）
- `main` へ直pushしない（PR + レビュー + CI）
- `data/`・`outputs/`・`submissions/` は原則コミットしない（`.gitignore` 済）
- 実験結果は `docs/experiments.md` に1行で残す

## 主要ディレクトリ
- `src/`: 再利用コード
- `notebooks/`: EDA / 試行（出力はコミットしない運用を推奨）
- `scripts/`: 実行スクリプト
- `configs/`: 設定ファイル
- `docs/`: 運用/実験ログ
