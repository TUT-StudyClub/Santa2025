## 変更内容
- 頂点ペナルティ法
  - 従来のSAは「重なったらNG」というルールのもとであり、その厳しさを撤廃し、「重なってもいいから、少しずつ押し返して最適化をする」という物理挙動を実装
- 重なり=ペナルティ
  - 現在のベスト配置の座標を中心に向かって縮小することで、意図的に重なりを作る
  - このペナルティを0にするように木を動かして、空いた隙間に木を埋め込むという施策

## 再現手順
- 実行コマンド: python ./experiments/exp012_vertex_penalty/run.py
- 使った config / notebook / script: ./experiments/exp012_vertex_penalty/run.py ./experiments/exp012_vertex_penalty/exp/000.yaml

## 結果（あれば）
- CV:
- LB: 70.781495651088

## チェックリスト
- [ ] `data/`・`outputs/`・`submissions/` をコミットしていない
- [ ] `docs/experiments.md` に記録した（実験/提出をした場合）
- [ ] `make check` / `pytest` が通る（CI）
