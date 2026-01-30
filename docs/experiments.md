# 実験ログ
「再現できる最小情報」を 1 行で残す運用を想定しています。

推奨: 新しい行を上に追加。

## 実験結果まとめ（日本語）
- 現時点の最良スコア: 70.628314（submissions/submission.csv）
- スコア推移: 178.9197 → 159.3497 → 117.2827 → 109.8679 → 70.935929406660 → 82.898630 → 70.924429 → 70.909422580984 → 70.892099 → 70.836885 → 70.781495651088 → 70.778910759459 → 70.778841 → 70.658982 → 70.658810 → 70.628314
- 目標スコア: 69

| Date | Author | Branch/PR | Change | Seed | CV | LB | Command/Config | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01-27 | haruto | - | exp021_ssa パラメータチューニング |  | - | score=70.569941 | `uv run python experiments/exp021_ssa/run.py exp=000` | 改善なし |
| 2026-01-21 | haruto | - | exp022_seed_ensemble Seed Ensemble |  | - | score=70.630028 | `uv run python experiments/exp022_seed_ensemble/run.py exp=000` | 重なり発生のためLB改善なし |
| 2026-01-21 | yuhei | - | exp021_ssa SSA局所探索（group=008 試運転） | 2042 | - | score=70.628314 | `uv run python experiments/exp021_ssa/run.py exp=001` | 改善なし（出力なし） |
| 2026-01-21 | yuhei | - | submission.csv 現状確認（best更新） | - | - | score=70.628314 | `uv run python scripts/analyze_scores.py --input submissions/submission.csv` | 生成経路は要追跡（Kaggle側でtouchがoverlap扱いになる場合あり） |
| 2026-01-20 | yuhei | - | exp018_gap_fill top_k=40 (21-100) | - | - | score=70.658810 | `uv run python experiments/exp018_gap_fill/run.py exp=007` | group=022 改善、submission.csv 更新 |
| 2026-01-20 | haruto | - | exp021_squeeze Guided SA with Squeeze | 42 | - | score=70.781496 | `python experiments/exp021_squeeze/run.py` | 改善なし、既存submission=70.781496で上書きなし |
| 2026-01-20 | yuhei | - | sparroWASM group=022 微改善 | - | - | score=70.658982 | `uv run python scripts/diff_submissions.py --base submissions/archive/submission.csv --new submissions/submission.csv` | group=022 のみ差し替え（diff=-0.000046） |
| 2026-01-13 | yuhei | - | exp020_symmetric_slide 重心中心 | 42 | - | score=70.781496 | `uv run python experiments/exp020_symmetric_slide/run.py exp=001` | 改善なし、既存submission=70.744149で上書きなし |
| 2026-01-13 | yuhei | - | exp020_symmetric_slide 点対称+揺らし+スライド | 42 | - | score=70.781496 | `uv run python experiments/exp020_symmetric_slide/run.py exp=000` | 改善なし、既存submission=70.744149で上書きなし |
| 2026-01-13 | yuhei | - | exp017_shake_slide 21-100/121-140集中 | 42 | - | score=70.781496 | `uv run python experiments/exp017_shake_slide/run.py exp=001` | 改善なし、既存submission=70.744149で上書きなし |
| 2026-01-13 | yuhei | - | exp018_gap_fill baseline固定+slide強化 | - | - | score=70.781496 | `uv run python experiments/exp018_gap_fill/run.py exp=006` | 改善なし、既存submission=70.744149で上書きなし |
| 2026-01-13 | yuhei | - | exp018_gap_fill score_tolerance=1e-6 | - | - | score=70.744149 | `uv run python experiments/exp018_gap_fill/run.py exp=005` | 改善なし、既存submission=70.744149で上書きなし |
| 2026-01-16 | yuhei | - | exp019_rrt SA局所探索（小グループ集中） | 42 | - | score=70.781496 | `uv run python experiments/exp019_rrt/run.py exp=002` | 改善なし、既存submission=70.771942で上書きなし |
| 2026-01-16 | yuhei | - | exp019_rrt SA局所探索 | 42 | - | score=70.781496 | `uv run python experiments/exp019_rrt/run.py exp=001` | 改善なし、既存submission=70.771942で上書きなし |
| 2026-01-16 | yuhei | - | exp019_rrt RRT局所探索 | 42 | - | score=70.781496 | `uv run python experiments/exp019_rrt/run.py exp=000` | 改善なし、既存submission=70.771942で上書きなし |
| 2026-01-13 | yuhei | - | exp018_gap_fill 探索強化 | - | - | score=70.771942 | `uv run python experiments/exp018_gap_fill/run.py exp=004` | 改善なし、既存submission=70.771942で上書きなし |
| 2026-01-13 | yuhei | - | exp018_gap_fill 全体実行 | - | - | score=70.771942 | `uv run python experiments/exp018_gap_fill/run.py exp=003` | 改善なし、既存submission=70.771942で上書きなし |
| 2026-01-13 | yuhei | - | exp018_gap_fill 境界候補追加 | - | - | score=70.771942 | `uv run python experiments/exp018_gap_fill/run.py exp=002` | 改善なし、既存submission=70.771942で上書きなし |
| 2026-01-13 | yuhei | - | exp018_gap_fill ギャップ吸引ログ | - | - | score=70.771942 | `uv run python experiments/exp018_gap_fill/run.py exp=001` | 改善なし、既存submission=70.771942で上書きなし |
| 2026-01-12 | yuhei | - | exp017_shake_slide 揺らし+スライド | 42 | - | score=70.781496 | `uv run python experiments/exp017_shake_slide/run.py exp=000` | 改善なし、既存submission=70.773909で上書きなし |
| 2026-01-12 | yuhei | - | exp016_gap_slide 強め押し込み | - | - | score=70.781496 | `uv run python experiments/exp016_gap_slide/run.py exp=002` | 改善なし、既存submission=70.773909で上書きなし |
| 2026-01-12 | yuhei | - | exp016_gap_slide 斜め方向スライド | - | - | score=70.781496 | `uv run python experiments/exp016_gap_slide/run.py exp=001` | 改善なし、既存submission=70.773909で上書きなし |
| 2026-01-12 | yuhei | - | exp015_pattern_penalty spacing拡張 | 4242 | - | score=70.778841 | `uv run python experiments/exp015_pattern_penalty/run.py exp=001` | 改善なし、既存submission=70.778841で上書きなし |
| 2026-01-12 | yuhei | - | exp015_pattern_penalty パターン初期化+ペナルティSA | 4242 | - | score=70.778841 | `uv run python experiments/exp015_pattern_penalty/run.py exp=000` | 改善なし、既存submission=70.778841で上書きなし |
| 2026-01-12 | yuhei | - | exp014_pressure_penalty 圧縮+ペナルティSA | 4242 | - | score=70.778841 | `uv run python experiments/exp014_pressure_penalty/run.py exp=000` | 改善なし、既存submission=70.778841で上書きなし |
| 2026-01-12 | yuhei | - | exp013_boundary_pack 21-100ワースト20 | 4242 | - | score=70.778841 | `uv run python experiments/exp013_boundary_pack/run.py exp=002` | 改善なし、既存submission=70.778841で上書きなし |
| 2026-01-12 | yuhei | - | exp012_vertex_penalty top30集中 | 42 | - | score=70.781496 | `uv run python experiments/exp012_vertex_penalty/run.py exp=003` | 改善なし、既存submission=70.779244で上書きなし |
| 2026-01-12 | yuhei | - | exp012_vertex_penalty ペナルティSA | 42 | - | score=70.781496 | `uv run python experiments/exp012_vertex_penalty/run.py exp=001` | 改善なし、既存submission=70.779244で上書きなし |
| 2026-01-12 | yuhei | - | exp007_intensive 非対称シード | 42 | - | score=70.781236 | `uv run python experiments/exp007_intensive/run.py exp=003` | 改善あり（+0.000260）だが既存submission=70.779244のため上書きなし |
| 2026-01-11 | yuhei | - | exp011_dense_pack 非対称グリッド | 42 | - | score=70.781496 | `uv run python experiments/exp011_dense_pack/run.py exp=003` | 改善なし、既存submission=70.779244で上書きなし |
| 2026-01-11 | yuhei | - | exp011_dense_pack 非対称ジッター | 42 | - | score=70.781496 | `uv run python experiments/exp011_dense_pack/run.py exp=002` | 改善なし、既存submission=70.779244で上書きなし |
| 2026-01-12 | haruto | exp/kotou_20260112 | exp0012_vertex_penalty 連続ペナルティ法の実装 | 42 | - | score=70.778910759459 | `python ./experiments/exp0012_vertex_penalty/run.py` | - |
| 2026-01-11 | yuhei | - | exp007_intensive 21-100限定 | 42 | - | score=70.781236 | `uv run ./experiments/exp007_intensive/run.py exp=002` | baseline 70.781496 → 70.781236（+0.000260）、submission.csv 出力 |
| 2026-01-11 | haruto | exp/kotou_20260111 | exp0012_vertex_penalty 頂点ペナルティ法の実装 | 42 | - | score=70.781495651088 | `python ./experiments/exp0012_vertex_penalty/run.py` | - |
| 2026-01-10 | haruto | exp/kotou_20260110 | exp0011_dense_pack グリッド形状の探索範囲を拡大 | 42 | - | score=70.829248956305 | `python ./experiments/exp0011_dense_pack/run.py` | - |
| 2026-01-08 | yuhei | - | exp007_intensive 強化SA + 回転後shrink | 42 | - | score=70.836885 | `uv run ./experiments/exp007_intensive/run.py exp=000` | baseline.csv, 強化パラメータ |
| 2026-01-05 | yuhei | - | exp004_baseline anisotropic shrink + rotate | 42 | - | score=70.892099 | `uv run ./experiments/exp004_baseline/run.py exp=007` | baseline.csv, shrink_aniso, rotate |
| 2026-01-05 | haruto | exp/kotou/20260105 | exp006_hill 貪欲法の改良 | 42 | - | score=70.892100493950 | `python ./experiments/exp006_hill/run.py exp=001` | - |
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
