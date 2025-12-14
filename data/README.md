# data/
このディレクトリ配下はデフォルトで git 管理しません（`data/README.md` のみ追跡）。

想定構成（例）:
- `data/raw/`: Kaggle から落とした生データ
- `data/interim/`: 前処理後の中間生成物（必要なら）

Kaggle CLI でのダウンロード例:
```bash
kaggle competitions download -c <competition-slug> -p data/raw
unzip -o data/raw/<zipfile>.zip -d data/raw
```

スクリプト（`.env` の `COMPETITION_SLUG` を参照）:
```bash
python scripts/download_competition_data.py
```

※ Kaggle API トークンは `~/.kaggle/kaggle.json` に置き、リポジトリには入れないでください。
