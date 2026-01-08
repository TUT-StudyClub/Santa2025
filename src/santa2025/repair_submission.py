import os

import pandas as pd

# ==========================================
# ファイルパスの設定 (環境に合わせて変更してください)
# ==========================================
# 重なりエラーが出ている現在のファイル
target_csv = "submissions/submission.csv"
# 安全なベースラインファイル (サンプル投稿)
baseline_csv = "submissions/baseline.csv"
# 出力ファイル名
output_csv = "submissions/submission_fix.csv"


def main():
    print("Loading files...")
    if not os.path.exists(target_csv) or not os.path.exists(baseline_csv):
        print("Error: File not found. Check paths.")
        return

    df_target = pd.read_csv(target_csv)
    df_base = pd.read_csv(baseline_csv)

    # IDをインデックスにして操作しやすくする
    df_target.set_index("id", inplace=True)
    df_base.set_index("id", inplace=True)

    # ---------------------------------------------------------
    # 強制的に修正するグループ番号のリスト
    # エラーが出続けるグループ番号をここに追加してください
    # ---------------------------------------------------------
    groups_to_force_revert = [3]

    print(f"Force reverting groups: {groups_to_force_revert}")

    for n in groups_to_force_revert:
        # IDのプレフィックスを作成 (例: 154 -> "154_", 5 -> "005_")
        prefix = f"{n:03d}_"

        # このグループに属するIDを抽出
        ids_to_fix = [i for i in df_target.index if i.startswith(prefix)]

        if not ids_to_fix:
            print(f"Warning: No IDs found for group {n}")
            continue

        print(f"  - Reverting Group {n} ({len(ids_to_fix)} trees)...")

        # ベースラインのデータで上書き (座標データなど全列)
        df_target.loc[ids_to_fix] = df_base.loc[ids_to_fix]

    # インデックスを戻して保存
    df_target.reset_index(inplace=True)
    df_target.to_csv(output_csv, index=False)
    print("=" * 40)
    print(f"Done! Saved to: {output_csv}")
    print("This file contains the safe baseline data for Group 154.")


if __name__ == "__main__":
    main()
