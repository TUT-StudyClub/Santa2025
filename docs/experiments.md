# 実験ログ
「再現できる最小情報」を 1 行で残す運用を想定しています。

推奨: 新しい行を上に追加。

## 実験結果まとめ（日本語）
- 現時点の最良スコア: 70.909422580984（exp006_hill exp=001）
- スコア推移: 178.9197 → 159.3497 → 117.2827 → 109.8679 → 70.935929406660 → 82.898630 → 70.924429
- 目標スコア: 69

| Date | Author | Branch/PR | Change | Seed | CV | LB | Command/Config | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-04 | haruto | exp/kotou/20260104 | exp006_hill 貪欲法の実装 | 42 | - | score=70.909422580984 | `uv run ./experiments/exp006_hill/run.py exp=001` | - |
| 2026-01-04 | yuhei | - | exp004_baseline shrink on baseline.csv | 42 | - | score=70.924429 | `uv run ./experiments/exp004_baseline/run.py exp=004` | baseline.csv, shrink |
| 2026-01-03 | yuhei | - | exp004_baseline shear+multistart+polish | 42 | - | score=82.898630 | `uv run ./experiments/exp004_baseline/run.py exp=003` | fallback baseline autogen |
| 2026-01-03 | haruto | exp/kotou/20260103 | exp005_reheat Reheatingの実装 | 42 | - | score=70.924430272852 | `uv run ./experiments/exp005_reheat/run.py exp=001` | - |
| 2026-01-01 | haruto | exp/kotou/20260101 | exp004_baseline Modifying the baseline file | 42 | - | score=70.935929406660 | `uv run ./experiments/exp004_baseline/run.py exp=001` | - |
| 2025-12-30 | haruto | exp/kotou/20251230 | exp003_es Introduction of Early Stopping | 42 | - | score=71.361575421075 | `uv run ./experiments/exp003_es/run.py exp=001` | - |
| 2025-12-29 | haruto | exp/kotou/20251229 | exp002_baseline Change SA movement range from fixed to dynamic | 42 | - | score=71.360590748689 | `uv run ./experiments/exp002_baseline/run.py exp=002` | - |
| 2025-12-29 | yuhei | - | exp001_baseline heavy search + recenter | 42 | score=109.8679 | - | `uv run experiments/exp001_baseline/run.py exp=030` | n=200, metrics.json |
| 2025-12-28 | yuhei | - | exp001_baseline compaction+reinsert | 42 | score=117.2827 | - | `uv run experiments/exp001_baseline/run.py exp=020` | n=200, metrics.json |
| 2025-12-28 | yuhei | - | exp001_baseline compaction | 42 | score=159.3497 | - | `uv run experiments/exp001_baseline/run.py exp=010` | n=200, metrics.json |
| 2025-12-28 | yuhei | - | exp001_baseline greedy packing | 42 | score=178.9197 | - | `uv run experiments/exp001_baseline/run.py exp=000` | n=200, metrics.json |
