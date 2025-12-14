# experiments/
Hydra を使った「major（実験フォルダ）× minor（設定ファイル）」で実験を再現しやすくする構成です。

## 例
- major: `experiments/exp000_sample/`
- minor: `experiments/exp000_sample/exp/000.yaml`

実行例:
```bash
python experiments/exp000_sample/run.py exp=000
python experiments/exp000_sample/run.py exp=001 exp.debug=true
```

