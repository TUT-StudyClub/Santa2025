# Santa2025（Kaggle チーム用リポジトリ）
ローカル実行を前提に「実験を再現できる形で残す」「チームで安全に共有する」ことを目的にしたテンプレです。

## まずはここだけ（最短で動かす）
```bash
cd Santa2025
uv sync --dev
uv run pre-commit install
uv run python experiments/exp000_sample/run.py exp=001 exp.debug=true
```

## コマンド早見表
| やりたいこと | コマンド例 | 補足 |
| --- | --- | --- |
| 依存関係を揃える | `uv sync --dev` | 初回/依存変更後 |
| Python を実行する | `uv run python <file.py> ...` | activate 不要 |
| 実験を回す | `make uv-exp EXP=000 CFG=001` | `experiments/` を実行 |
| ノートブックを開く | `make notebook` | JupyterLab |
| フォーマット | `make fmt` | `ruff format` |
| Lint | `make lint` | `ruff check` |
| テスト | `make test` | `pytest` |

## このリポジトリで大事にしていること
### 1) データや生成物は Git に入れない
以下は巨大化・情報漏洩・衝突の原因になりやすいので、デフォルトでコミットしない設計です（`.gitignore` 済）。
- `data/`（Kaggle から落としたデータ）
- `outputs/`（モデル・特徴量・ログ・図など）
- `submissions/`（提出ファイル）
- `.env` / `kaggle.json` などの秘密情報

※ もし誤って追跡してしまうと、CI が落ちるようにしています。

### 2) 「何を変えてスコアが変わったか」を必ず残す
実験や提出をしたら `docs/experiments.md` に 1 行で記録します。
後から「なぜ上がったか/下がったか」を追えるのがチームでは特に重要です。

## 事前準備（初回だけ）
### 必要なもの
- Git
- Python（推奨: 3.11.13。`./.python-version` に合わせると Kaggle に寄せやすい）
- `uv`（推奨: Python 環境管理。なければ venv + pip でもOK）
- Kaggle アカウント（データDLや提出をするなら）
- Docker（任意: Kaggle 互換環境で回したい人向け）

### Kaggle API トークンの設定（データDL/アップロードする場合）
Kaggle CLI を使うには `kaggle.json` が必要です。
1. Kaggle の「Account」ページ → 「Create New API Token」→ `kaggle.json` をダウンロード
2. `~/.kaggle/kaggle.json` に配置
3. 権限を絞る（macOS/Linux）:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

※ コンペによっては「Join」やルール同意が必要です（同意してないとダウンロードが失敗します）。

## セットアップ（チーム共通）
### 方法A: uv（推奨 / 迷いにくい）
`uv` は lock された依存関係（`uv.lock`）で全員の環境を揃えやすいです。
```bash
cd Santa2025
uv sync --dev
uv run pre-commit install
```

ポイント:
- `uv run ...` を使えば、仮想環境を activate しなくても実行できます
- 依存関係は `pyproject.toml` と `uv.lock` で管理します（`uv.lock` は Git に入れる）

### 方法B: venv + pip（uv を使わない場合）
```bash
cd Santa2025
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pre-commit install
```

## `.env` の用意（任意だけど推奨）
`.env` は「個人PC依存の設定」を置くためのファイルです（Git には入りません）。
```bash
cp .env.example .env
```

例（必要なものだけでOK）:
```env
COMPETITION_SLUG=your-competition-slug
SEED=42
```

## データの準備
### データの置き場所
- 生データ: `data/raw/`
- 中間生成物: `data/interim/`（必要なら）

詳細: `data/README.md`

### Kaggle からダウンロード（推奨）
`.env` の `COMPETITION_SLUG` を参照します。
```bash
uv run python scripts/download_competition_data.py
```

slug を直接指定する場合:
```bash
uv run python scripts/download_competition_data.py --competition <competition-slug>
```

## 実験を回す（Hydra）
このリポジトリは「実験の再現性」を上げるために、以下の形で実験を管理します。
- major（実験単位）: `experiments/exp000_sample/` のようなフォルダ
- minor（設定）: `experiments/exp000_sample/exp/000.yaml` のような yaml

### 実行例（サンプル）
```bash
uv run python experiments/exp000_sample/run.py exp=000
uv run python experiments/exp000_sample/run.py exp=001 exp.debug=true
```

### 出力物はどこに出る？
実行すると、例えば以下に成果物が保存されます。
`outputs/runs/<major>/<minor>/<timestamp>/`

中には最低限これが入ります:
- `*.log`: 実行ログ
- `config_*.yaml`: 実行時の設定（再現に必須）
- `metrics.json`: 指標（CV などをここに保存する運用にすると便利）

### 新しい実験（major）を作る手順（おすすめ）
1. `experiments/exp000_sample/` をコピーして `experiments/exp001_<name>/` を作る
2. `run.py` を「その実験の入口」にして、処理は適宜 `src/` に寄せる
3. 実験設定は `exp/000.yaml` のように増やしていく
4. 実験を回したら `docs/experiments.md` に結果を 1 行で残す

## 提出と記録（チームで重要）
### 提出ファイル
- 置き場所: `submissions/`（Git には入れない）
- 命名例: `submissions/20251214_1230_exp001_cv0.8123.csv`

### 実験ログ
- `docs/experiments.md` に「何を変えたか / CV / LB / 再現コマンド」を 1 行で残す

詳細な運用ルール: `docs/TEAM_GUIDE.md`

## 開発コマンド（よく使う）
```bash
make fmt     # format
make lint    # lint
make test    # tests
make uv-exp EXP=000 CFG=001   # 実験実行（uv）
```

## 依存関係を追加したい（例）
### uv を使っている場合（推奨）
```bash
uv add lightgbm
uv add --dev pytest-xdist
```

### venv + pip の場合
```bash
# 1) requirements.txt にパッケージ名を追記（例: lightgbm）
# 2) 依存を入れ直す
python -m pip install -r requirements-dev.txt
```

## Docker（任意 / Kaggle に寄せたい人向け）
ローカル環境が人によって違って不安な場合に使います。
```bash
make docker-build
make docker-jupyter
# CPU の場合: make docker-build CPU=1
```

## Kaggle Dataset へのアップロード（任意）
チーム内で「学習済みモデルや特徴量など」を共有したい場合に使います。
```bash
uv run python tools/upload_dataset.py \
  --dataset <username>/<dataset-slug> \
  --src-dir outputs/runs/exp000_sample/000
```

## 分析スクリプト
`scripts/` ディレクトリには、提出ファイルの分析用スクリプトがあります。

### `analyze_scores.py`
各グループのスコアを分析して、改善の余地があるグループを特定します。

```bash
uv run python scripts/analyze_scores.py
```

**出力内容:**
- 総スコア
- スコアが最も悪い（高い）20グループ
- スコアが最も良い（低い）20グループ
- グループサイズ別のスコア分布

### `theoretical_analysis.py`
理論的な最小スコアと現在のベースラインの効率を分析します。

```bash
uv run python scripts/theoretical_analysis.py
```

**出力内容:**
- 木の形状パラメータ（高さ、幅、面積）
- 異なるパッキング効率での理論的最小スコア
- グループごとのスコア、サイド長、幅、高さ、アスペクト比、効率
- アスペクト比が悪い（改善の余地がある）グループの特定

## 主要ディレクトリ
- `experiments/`: 実験（major/minor で再現）
- `src/`: 再利用コード（長く使うロジックはここへ）
- `notebooks/`: EDA / 試行（出力セルはコミットしない運用推奨）
- `scripts/`: 補助スクリプト（DL、分析など）
- `configs/`: 共通設定（必要に応じて）
- `docs/`: 運用/実験ログ

## 困ったとき（よくある詰まり）
- `uv: command not found` → uv をインストールしてからやり直してください（公式: https://docs.astral.sh/uv/）
- Kaggle のダウンロードが失敗する → コンペの Join/ルール同意、`~/.kaggle/kaggle.json` の配置、`chmod 600` を確認
- `pre-commit` が落ちる → まず `uv run pre-commit run -a` で修正し、再コミット
- うっかり `data/` や `outputs/` を Git に入れた → `git rm -r --cached data outputs submissions` してからコミット
