from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def unzip_all(zip_dir: Path, dest_dir: Path) -> None:
    for zip_path in sorted(zip_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", "-c", default=None, help="Kaggle competition slug")
    parser.add_argument("--out", "-o", default="data/raw", help="Download directory")
    args = parser.parse_args()

    competition = args.competition or os.getenv("COMPETITION_SLUG")
    if not competition:
        raise SystemExit(
            "competition slug is required: use --competition or set COMPETITION_SLUG in .env"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(out_dir)],
        check=True,
    )
    unzip_all(out_dir, out_dir)
    print(f"Downloaded & extracted: {competition} -> {out_dir}")


if __name__ == "__main__":
    main()
