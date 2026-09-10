"""L7 の測定。**位相を見える要約に替えると勝てるか**(P-07)と、CH4 の切り分け。

L2 との違いは要約の取り方**だけ**である。層の数も幅も学習率の掃引の仕方も
評価の土俵も同じで、`model.POOL_MODE` の一つの切替しか動かさない ——
制御された ablation にするため。

置く系は四つ。

| 系 | 要約 | 母集団 | 何を測るか |
|---|---|---|---|
| `cnn_harmonic` | 位相あり | 805 | **P-07 の判定**(相手は 2 特徴 kNN の 10.95 度) |
| `cnn_invariant_paired` | 位相なし | 739 | CH4 比較の基準(L2 の再掲) |
| `cnn_ch4_paired` | 位相なし | 739 | CH4 を足した系(L2 の再掲) |
| `cnn_ch4_shuffled_paired` | 位相なし | 739 | **CH4 を年ごとに入れ替えた対照** |

最後の一つが §7.11 の宿題に答える —— CH4 で下がったのが「緯度の情報が増えた」からか
「第 2 チャンネルが学習を安定させた」からかを分ける。
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

from pipeline import curves, experiment, model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "phase.json"


def _jsonable(value):
    return experiment._jsonable(value)


def run_system(examples, x, y, channels: int, label: str, pool_mode: str) -> dict:
    """一系を測る。**要約の切替はここでだけ触り、終わったら必ず戻す。**"""
    original = model.POOL_MODE
    model.POOL_MODE = pool_mode
    try:
        return {
            **experiment.evaluate(examples, x, y, channels=channels, label=label),
            "pool_mode": pool_mode,
        }
    finally:
        model.POOL_MODE = original


def main() -> int:
    baseline = json.loads((ROOT / "data" / "baseline.json").read_text(encoding="utf-8"))
    opponent_name = baseline["best_two_feature_system"]
    opponent_mae = baseline["systems"][opponent_name]["mae"]

    paired_baseline = json.loads(
        (ROOT / "data" / "baseline_paired.json").read_text(encoding="utf-8")
    )
    paired_opponent = paired_baseline["systems"][paired_baseline["best_two_feature_system"]]["mae"]

    previous = json.loads((ROOT / "data" / "model.json").read_text(encoding="utf-8"))

    co2 = curves.build_examples("co2")
    x1, y1 = model.to_arrays(co2)

    # **再開できるようにする。** 既に報告へ入っている系は測り直さない。
    # 高価な計算は落ちる前提で組む —— loop_007 では 12 走を終えた直後に
    # プロセスが無言で消え、書き出してあった分だけが残った(TOOL-ENV)。
    resumed: dict = {}
    previous_sweep = None
    if OUT.exists():
        earlier = json.loads(OUT.read_text(encoding="utf-8"))
        resumed = earlier.get("systems", {})
        previous_sweep = earlier.get("lr_sweep")
        if resumed:
            print(f"再開: 済み {sorted(resumed)}")

    report = {
        "generated_from": "pipeline/phase_experiment.py",
        "status": "running",
        "max_epochs": experiment.MAX_EPOCHS,
        "seeds": list(experiment.SEEDS),
        "opponent": {"system": opponent_name, "mae": opponent_mae},
        "opponent_paired": {"mae": paired_opponent},
        "previous": {
            "cnn_co2": previous["systems"]["cnn_co2"]["mae"],
            "cnn_co2_paired": previous["systems"]["cnn_co2_paired"]["mae"],
            "cnn_co2_ch4_paired": previous["systems"]["cnn_co2_ch4_paired"]["mae"],
        },
        "systems": dict(resumed),
    }
    if previous_sweep is not None:
        report["lr_sweep"] = previous_sweep
    OUT.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --- P-07: 位相を見える要約 -------------------------------------------------
    print(f"主系統(位相あり): {len(co2)} 地点年 / {len({e.group for e in co2})} 群")
    if "cnn_harmonic" in resumed:
        harmonic = resumed["cnn_harmonic"]
        experiment.LEARNING_RATE = report["lr_sweep"]["chosen_lr"]
        print(f"  済み(再開): cnn_harmonic {harmonic['mae']:.2f} / lr={experiment.LEARNING_RATE:g}")
    else:
        print("学習率の掃引(選定は内側検証のみ・種 0):")
        original = model.POOL_MODE
        model.POOL_MODE = "harmonic"
        try:
            sweep = experiment.sweep_learning_rate(co2, x1, y1, channels=1, seed=0)
            experiment.LEARNING_RATE = sweep["chosen_lr"]
            print(f"  → 採用 lr={experiment.LEARNING_RATE:g}")
            report["lr_sweep"] = sweep
            harmonic = {
                **experiment.evaluate(co2, x1, y1, channels=1, label="cnn_harmonic"),
                "pool_mode": "harmonic",
            }
        finally:
            model.POOL_MODE = original
        report["systems"]["cnn_harmonic"] = harmonic
    OUT.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --- CH4 の切り分け --------------------------------------------------------
    ch4 = {(e.code, e.year): e for e in curves.build_examples("ch4")}
    paired = [e for e in co2 if (e.code, e.year) in ch4]
    second = [ch4[(e.code, e.year)] for e in paired]
    shuffled = experiment.shuffled_second_channel(paired, second)

    x_one, y_pair = model.to_arrays(paired)
    x_two, _ = model.to_arrays(paired, second=second)
    x_shuf, _ = model.to_arrays(paired, second=shuffled)

    # 入れ替えが本当に起きていることを、入力の差で確かめる(宣言でなく実測 — HC-071)
    changed = int(np.sum(np.any(x_two[:, :, 1] != x_shuf[:, :, 1], axis=1)))
    print(f"CH4 の切り分け: {len(paired)} 地点年 / 入れ替わった例 {changed}")
    assert changed == len(paired), "入れ替えが全例に効いていない"

    for label, arr, channels in (
        ("cnn_invariant_paired", x_one, 1),
        ("cnn_ch4_paired", x_two, 2),
        ("cnn_ch4_shuffled_paired", x_shuf, 2),
    ):
        if label in resumed:
            print(f"  済み(再開): {label} {resumed[label]['mae']:.2f}")
            continue
        report["systems"][label] = run_system(paired, arr, y_pair, channels, label, "invariant")
        OUT.write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    report["ch4_decomposition"] = {
        "real_minus_base": report["systems"]["cnn_ch4_paired"]["mae"]
        - report["systems"]["cnn_invariant_paired"]["mae"],
        "shuffled_minus_base": report["systems"]["cnn_ch4_shuffled_paired"]["mae"]
        - report["systems"]["cnn_invariant_paired"]["mae"],
        "note": (
            "入れ替えた CH4 でも同じだけ下がるなら、効いていたのは情報でなく"
            "「第 2 チャンネルがある」ことである"
        ),
    }
    report["status"] = "complete"
    OUT.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"相手(2 特徴 kNN): {opponent_mae:.2f}")
    print(f"位相なし(L2): {report['previous']['cnn_co2']:.2f}")
    print(f"位相あり(L7): {harmonic['mae']:.2f}(種ごと {[f'{m:.2f}' for m in harmonic['mae_per_seed']]})")
    print(f"P-07 → {'成立' if harmonic['mae'] < opponent_mae else '不成立'}")
    print()
    d = report["ch4_decomposition"]
    print(f"CH4 本物: {d['real_minus_base']:+.2f} 度 / 入れ替え: {d['shuffled_minus_base']:+.2f} 度")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
