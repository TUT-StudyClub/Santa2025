# sparroWASM（sparrow-cli）で最適化する

このリポジトリの木（polygon）配置を、外部のネスティングエンジン **sparroWASM**（`sparrow-cli`）で最適化するためのメモです。

## 1. `sparrow-cli` の用意

`sparrow-cli` は Rust 製のネイティブ CLI です（このリポジトリには同梱していません）。

### Rust をインストール

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

### `sparrow-cli` をビルド

例（`sparroWASM-fork` からビルド）:

```bash
git clone https://github.com/truyentu/sparroWASM-fork.git third_party/sparroWASM-fork
cd third_party/sparroWASM-fork
cargo build --release --bin sparrow-cli

# macOS/Linux の場合
./target/release/sparrow-cli --help
```

`sparrow-cli` のパス例:
- `third_party/sparroWASM-fork/target/release/sparrow-cli`

## 2. 最適化スクリプトの実行

`scripts/optimize_with_sparrowwasm.py` は、既存の提出 CSV の一部グループを `sparrow-cli` で再配置し、スコアが改善したグループだけ差し替えます。

### 例: 21〜100 だけ試す

```bash
uv run python scripts/optimize_with_sparrowwasm.py \
  --input submissions/submission.csv \
  --output submissions/sparrowwasm_try.csv \
  --sparrow-cli third_party/sparroWASM-fork/target/release/sparrow-cli \
  --groups 21-100 \
  --timeout 10 \
  --workers 4 \
  --orient-step 5 \
  --sparrow-scale 1000 \
  --strip-height-scales 0.9,0.95,1.0 \
  --trials 3 \
  --print-best \
  --early-termination
```

### 例: 高速に雑に試す（角度を粗く・短時間）

```bash
uv run python scripts/optimize_with_sparrowwasm.py \
  --input submissions/submission.csv \
  --output submissions/sparrowwasm_fast.csv \
  --sparrow-cli third_party/sparroWASM-fork/target/release/sparrow-cli \
  --groups 1-200 \
  --timeout 2 \
  --workers 4 \
  --orient-step 10 \
  --early-termination
```

## 3. 注意点

- `sparrow-cli` は「固定高さのストリップ」を前提に幅を最小化するため、正方形最小化（`max(width,height)`）と目的が完全一致ではありません。
  - このスクリプトでは、出力レイアウトに対して **全体回転の探索** を追加して、正方形側長が小さくなる角度を探します。
- `sparrow-cli` の出力 `position_x/position_y` は「shape の重心位置」として解釈して、提出CSVの `x/y`（幹の付け根中央）へ変換しています。
- `--orient-step` を細かくするほど（例: `1`）回転の自由度は上がりますが、探索が重くなりやすいです（`0` 以下なら回転制約なし）。
- 形状サイズが小さすぎると `sparrow-cli` 側の探索が不安定になる場合があるため、`--sparrow-scale 1000` のようにスケールアップしてから実行します（出力は自動で元スケールに戻します）。
