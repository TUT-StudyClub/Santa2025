# チーム運用ガイド（ローカル実行前提）

## 目的
- `main` を常に動く状態に保つ（壊れたら全員が困る）
- 実験の再現性を上げる（「何を変えて上がったか」を追える）
- データ/生成物でリポジトリを肥大化させない

## Git 運用（最小ルール）
- `main` への直pushは禁止（PR 必須）
- PR は最低 1 人レビュー + CI 通過でマージ
- ブランチ名例:
  - `feat/<name>/<topic>`: 機能追加
  - `exp/<name>/<yyyymmdd>_<tag>`: 実験
  - `fix/<name>/<topic>`: バグ修正

## 実験の残し方（重要）
- 実験ごとに `docs/experiments.md` に 1 行で記録
  - いつ / 誰が / 何を変えた / CV / LB / 再現手順（スクリプト or config）
- 生成物（モデル/特徴量/ログ/図）は `outputs/` 配下に置く
  - 例: `outputs/runs/20251214_1230_baseline/`
- 提出ファイルは `submissions/` に置いて、スコアをログに残す

## 実験ディレクトリ（Hydra 運用）
- `experiments/expXXX_<name>/` を major（実験単位）として管理
- `experiments/expXXX_<name>/exp/000.yaml` のようなファイルを minor（実験設定）として管理
- 実行例:
  - `python experiments/exp000_sample/run.py exp=000`
  - `python experiments/exp000_sample/run.py exp=001 exp.debug=true`
- 新しい major を切るときは、まず `exp000_sample` をコピーして中身を整理すると早い

## データ/秘密情報の扱い
- `data/`（配布データ）と `outputs/`（生成物）と `submissions/` は基本コミットしない（`.gitignore` 済）
- Kaggle API トークンは `~/.kaggle/kaggle.json` に置く（リポジトリに入れない）
- `.env` などの秘密情報はコミットしない（必要なら `.env.example` だけ共有）

## Notebook 運用
- Notebook は `notebooks/` に置く
- 出力セルをコミットしない運用を推奨（`pre-commit` で strip するのが楽）
- ロジックはできるだけ `src/` と `scripts/` に寄せる（差分がレビューしやすい）
