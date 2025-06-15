import argparse
import csv
import subprocess
import sys
from pathlib import Path


TASK_GROUPS = {
    "gsm8k_plus": {
        "columns": [
            "gsm_plus",
        ],
        "script": "scripts/abci3/evaluate_gsm8k_plus.sh",
    },
    "english_general": {
        "columns": [
            "gsm8k",
            "squad2",
            "triviaqa",
            "hellaswag",
            "openbookqa",
            "xwinograd_en",
        ],
        "script": "scripts/abci3/evaluate_english_general.sh",
    },
    "english_bbh": {
        "columns": [
            "bbh_cot",
        ],
        "script": "scripts/abci3/evaluate_english_bbh.sh",
    },
    "arc_0shot": {
        "columns": [
            "arc_challenge",
            "arc_easy",
        ],
        "script": "scripts/abci3/evaluate_arc_0shot.sh",
    },
    "arc_25shot": {
        "columns": [
            "arc_challenge_25shot",
            "arc_easy_25shot",
        ],
        "script": "scripts/abci3/evaluate_arc_25shot.sh",
    },
    "piqa_10shot": {
        "columns": [
            "piqa_10shot",
        ],
        "script": "scripts/abci3/evaluate_piqa_10shot.sh",
    },
    "evaluate_gsm8k_cot": {
        "columns": [
            "gsm8k_8shot",
            "gsm8k_0shot",
            "gsm8k_8shot_cot",
            "gsm8k_4shot_cot",
        ],
        "script": "scripts/abci3/evaluate_gsm8k_cot.sh",
    },
    "gsm8k_cot_zeroshot": {
        "columns": [
            "gsm8k_cot_zeroshot",
        ],
        "script": "scripts/abci3/evaluate_gsm8k_cot_0shot.sh",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=Path, required=True, help="show_result.py で生成されたCSVファイル"
    )
    ap.add_argument(
        "--task-group",
        choices=list(TASK_GROUPS.keys()),
        required=True,
        help="評価するタスクグループ",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="qsubコマンドを書き出すシェルスクリプト（省略時は自動生成）",
    )
    ap.add_argument("--queue", action="store_true", help="即座にqsubを実行する")
    args = ap.parse_args()

    # タスクグループの設定を取得
    task_info = TASK_GROUPS[args.task_group]
    target_columns = task_info["columns"]
    qsub_script = task_info["script"]

    # デフォルトの出力ファイル名
    if args.out is None:
        args.out = Path(f"scripts/missing_{args.task_group}_jobs.sh")

    # CSVを読み込む
    missing_models = []

    with args.csv.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_path = row["model"]

            # 指定されたタスクグループのカラムで -1.0 があるかチェック
            has_missing = any(
                float(row.get(col, 0)) == -1.0 for col in target_columns if col in row
            )

            if has_missing:
                missing_models.append(model_path)

    if not missing_models:
        print(f"✓ すべてのモデルで {args.task_group} の評価が完了しています。")
        return

    print(
        f"⚠️  {len(missing_models)} 個のモデルで未評価の {args.task_group} タスクがあります。"
    )

    # qsubコマンドを生成
    qsub_commands = []
    for model_path in missing_models:
        # 52bモデルの場合はTP=4, RTYPE=rt_HF
        if "52b" in model_path.lower():
            rtype = "rt_HF"
            tp = 4
        else:
            rtype = "rt_HG"
            tp = 1

        cmd = f"qsub -v RTYPE={rtype},MODEL_NAME_PATH={model_path},TP={tp},DP=1 {qsub_script}"
        qsub_commands.append(cmd)
        print(f"  - {model_path} (TP={tp}, RTYPE={rtype})")

    if args.queue:
        # 即座に実行
        print(f"\n▶ {args.task_group} 評価の qsub コマンドを実行します...")
        for cmd in qsub_commands:
            print(f" $ {cmd}")
            try:
                res = subprocess.run(cmd, shell=True, check=False)
                if res.returncode != 0:
                    print(f"  ⚠️  failed (exit {res.returncode})", file=sys.stderr)
            except Exception as e:
                print(f"  ⚠️  error: {e}", file=sys.stderr)
    else:
        # シェルスクリプトに書き出し
        args.out.parent.mkdir(parents=True, exist_ok=True)

        script_content = "#!/bin/bash\n\n"
        script_content += f"# Missing {args.task_group} evaluation jobs\n"
        script_content += f"# Generated from: {args.csv}\n"
        script_content += f"# Total models: {len(missing_models)}\n"
        script_content += f"# Target columns: {', '.join(target_columns)}\n"
        script_content += "# Note: 52b models use TP=4 and RTYPE=rt_HF\n\n"

        for cmd in qsub_commands:
            script_content += cmd + "\n"

        args.out.write_text(script_content)
        args.out.chmod(0o755)

        print(f"\n✓ qsub コマンドを {args.out} に書き出しました。")
        print(f"   実行するには: ./{args.out} または bash {args.out}")


if __name__ == "__main__":
    main()
