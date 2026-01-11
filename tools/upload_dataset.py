from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()


def copy_directory(source_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=False)


@click.command()
@click.option("--dataset", "-d", required=True, help="例: username/santa2025-exp000")
@click.option("--title", "-t", default=None, help="Kaggle 上での表示タイトル（省略可）")
@click.option(
    "--src-dir",
    "-s",
    type=click.Path(path_type=Path),
    required=True,
    help="アップロード対象ディレクトリ",
)
@click.option("--notes", "-n", default="", help="バージョンノート（更新時）")
@click.option("--new", is_flag=True, help="新規作成（既存の場合は更新）")
@click.option("--public", is_flag=True, help="公開データセットとして作成（デフォルトは private）")
def main(  # noqa: PLR0913
    dataset: str,
    title: str | None,
    src_dir: Path,
    notes: str,
    new: bool,
    public: bool,
) -> None:
    """
    ディレクトリをまるごと Kaggle Dataset としてアップロードする。

    注意: Kaggle API の認証が必要（~/.kaggle/kaggle.json）。
    """
    if not src_dir.exists():
        raise SystemExit(f"src-dir not found: {src_dir}")

    tmp_root = Path("tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_root / "kaggle_dataset"
    copy_directory(src_dir, tmp_dir)

    dataset_metadata: dict[str, Any] = {
        "id": dataset,
        "licenses": [{"name": "CC0-1.0"}],
        "title": title or dataset.split("/", 1)[-1],
    }
    (tmp_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    api = KaggleApi()
    api.authenticate()

    if new:
        api.dataset_create_new(
            folder=str(tmp_dir),
            dir_mode="tar",
            convert_to_csv=False,
            public=public,
        )
    else:
        api.dataset_create_version(
            folder=str(tmp_dir),
            version_notes=notes,
            dir_mode="tar",
            convert_to_csv=False,
        )

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
