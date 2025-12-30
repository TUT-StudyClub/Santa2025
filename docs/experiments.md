# 実験ログ
「再現できる最小情報」を 1 行で残す運用を想定しています。

推奨: 新しい行を上に追加。

## 実験結果まとめ（日本語）
- 現時点の最良スコア: 109.8679（exp=030, heavy search + recenter）
- スコア推移: 178.9197 → 159.3497 → 117.2827 → 109.8679
- 目標スコア70には未到達

| Date | Author | Branch/PR | Change | Seed | CV | LB | Command/Config | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-12-29 | haruto | exp/kotou/20251229 | exp002_baseline Change SA movement range from fixed to dynamic | 42 | - | core=71.360590748689 | `uv run ./experiments/exp002_baseline/run.py exp=001` | - |
| 2025-12-29 | yuhei | - | exp001_baseline heavy search + recenter | 42 | score=109.8679 | - | `uv run experiments/exp001_baseline/run.py exp=030` | n=200, metrics.json |
| 2025-12-28 | yuhei | - | exp001_baseline compaction+reinsert | 42 | score=117.2827 | - | `uv run experiments/exp001_baseline/run.py exp=020` | n=200, metrics.json |
| 2025-12-28 | yuhei | - | exp001_baseline compaction | 42 | score=159.3497 | - | `uv run experiments/exp001_baseline/run.py exp=010` | n=200, metrics.json |
| 2025-12-28 | yuhei | - | exp001_baseline greedy packing | 42 | score=178.9197 | - | `uv run experiments/exp001_baseline/run.py exp=000` | n=200, metrics.json |
