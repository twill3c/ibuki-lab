"""L8 の測定。**学習率の nested 選定**と、**CH4 の何が効いたかの切り分け**。

判定の線は測る前に SPEC §7.17 に書いた。ここはそれを測って `data/nested.json` に書くだけで、
線を動かさない。

置く系は次のとおり。

| 系 | 要約 | 母集団 | 何を測るか |
|---|---|---|---|
| 再現の検算 | 位相なし | 739 | 本物の CH4 の種 0 が L7 の値と一致するか(学習率の既定値の事故を捕まえる) |
| `cnn_ch4_rotated_paired` | 位相なし | 739 | CH4 を巡回でずらす(CO2 との位相差を壊す) |
| `cnn_ch4_equalised_paired` | 位相なし | 739 | CH4 の振幅を揃える(振幅の大小を消す) |
| nested `harmonic` | 位相あり | 805 | P-07 の再判定・P-08・P-09 |
| nested `invariant` | 位相なし | 805 | P-11(位相の効果が選定規則の産物でないか) |

**高価な計算は落ちる前提で組む。** 系ごと、nested は種ごとに書き出し、再開できる。
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

from pipeline import curves, experiment, model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nested.json"

#: 再現の検算の許容(度)。同じ計算を同じ機械で回すので本来は一致する。
#: ずれるなら土俵(学習率・要約・母集団)がどこか違っている。
REPRODUCTION_TOLERANCE = 0.05


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def save(report: dict) -> None:
    OUT.write_text(
        json.dumps(experiment._jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def closer_to_real(control: float, real: float, base: float) -> bool:
    """対照の MAE が、基準より本物に近いか(P-10 の種ごとの判定)。"""
    return abs(control - real) < abs(control - base)


def ch4_controls(report: dict) -> None:
    phase = load("phase.json")
    lr = load("model.json")["learning_rate"]
    experiment.LEARNING_RATE = lr
    assert model.POOL_MODE == "invariant", "CH4 の表は位相なしで測った"

    co2 = curves.build_examples("co2")
    ch4 = {(e.code, e.year): e for e in curves.build_examples("ch4")}
    paired = [e for e in co2 if (e.code, e.year) in ch4]
    second = [ch4[(e.code, e.year)] for e in paired]
    x_real, y = model.to_arrays(paired, second=second)
    print(f"CH4 の切り分け: {len(paired)} 地点年 / lr={lr:g}", flush=True)

    # --- 再現の検算(ここで落ちれば何時間も使う前に止まる) ------------------------
    if "reproduction" not in report:
        started = time.time()
        result = experiment.run_logo(paired, x_real, y, 2, seed=0, strict_budget=False)
        mae = experiment._per_site_mae(paired, range(len(paired)), result["preds"], y)
        expected = phase["systems"]["cnn_ch4_paired"]["mae_per_seed"][0]
        report["reproduction"] = {
            "system": "cnn_ch4_paired", "seed": 0, "mae": mae, "expected": expected,
            "delta": mae - expected, "tolerance": REPRODUCTION_TOLERANCE,
            "seconds": round(time.time() - started),
        }
        save(report)
        print(f"  再現の検算: {mae:.4f}(L7 {expected:.4f})", flush=True)
    rep = report["reproduction"]
    if abs(rep["delta"]) >= REPRODUCTION_TOLERANCE:
        raise RuntimeError(f"L7 の値を再現しない(差 {rep['delta']:+.4f} 度)。土俵が違う")

    variants = {
        "cnn_ch4_rotated_paired": experiment.rotated_second_channel(second),
        "cnn_ch4_equalised_paired": experiment.equalised_second_channel(second),
    }
    # 対照が実際に効いていることを入力の側で確かめる(宣言でなく実測 — HC-071)
    checks = {}
    for label, variant in variants.items():
        x_var, _ = model.to_arrays(paired, second=variant)
        changed = int(np.sum(np.any(np.abs(x_var[:, :, 1] - x_real[:, :, 1]) > 1e-6, axis=1)))
        checks[label] = {"changed_examples": changed, "n": len(paired)}
        print(f"  {label}: 入力が変わった例 {changed}/{len(paired)}", flush=True)
    report["ch4_control_checks"] = checks
    # 巡回は全例で変わるはず。揃えは標準偏差がちょうど中央値の例だけ変わらない(1 例まで許す)
    assert checks["cnn_ch4_rotated_paired"]["changed_examples"] == len(paired)
    assert checks["cnn_ch4_equalised_paired"]["changed_examples"] >= len(paired) - 1

    for label, variant in variants.items():
        if label in report["ch4_systems"]:
            print(f"  済み(再開): {label} {report['ch4_systems'][label]['mae']:.2f}", flush=True)
            continue
        x_var, _ = model.to_arrays(paired, second=variant)
        report["ch4_systems"][label] = experiment.evaluate(paired, x_var, y, channels=2, label=label)
        save(report)

    base = phase["systems"]["cnn_invariant_paired"]
    real = phase["systems"]["cnn_ch4_paired"]
    denominator = base["mae"] - real["mae"]
    decomposition = {}
    for label in variants:
        ctrl = report["ch4_systems"][label]
        sides = [
            closer_to_real(c, r, b)
            for c, r, b in zip(ctrl["mae_per_seed"], real["mae_per_seed"], base["mae_per_seed"])
        ]
        decomposition[label] = {
            "mae": ctrl["mae"],
            "spread": ctrl["mae_spread"],
            "retained": (base["mae"] - ctrl["mae"]) / denominator,
            "closer_to_real_per_seed": sides,
            "all_seeds_agree": len(set(sides)) == 1,
        }
    rot = decomposition["cnn_ch4_rotated_paired"]
    eq = decomposition["cnn_ch4_equalised_paired"]
    holds = eq["retained"] < 0.5 and rot["retained"] >= 0.5
    identified = (
        holds
        and eq["all_seeds_agree"] and not eq["closer_to_real_per_seed"][0]
        and rot["all_seeds_agree"] and rot["closer_to_real_per_seed"][0]
    )
    report["ch4_decomposition"] = {
        "base_mae": base["mae"], "real_mae": real["mae"], "denominator": denominator,
        "controls": decomposition,
        "p10_holds_numerically": holds,
        "p10_identified": identified,
    }
    save(report)


def nested_system(report: dict, key: str, examples, x, y, pool_mode: str) -> dict:
    entry = report["nested"].setdefault(key, {"pool_mode": pool_mode, "seeds": []})
    done = {row["seed"] for row in entry["seeds"]}
    original = model.POOL_MODE
    model.POOL_MODE = pool_mode
    try:
        for seed in experiment.SEEDS:
            if seed in done:
                print(f"  済み(再開): {key} seed={seed}", flush=True)
                continue
            print(f"  {key} seed={seed}:", flush=True)
            result = experiment.run_nested(examples, x, y, 1, seed)
            row = experiment.summarise_nested(examples, y, result, seed)
            entry["seeds"].append(row)
            save(report)
            print(
                f"  {key} seed={seed}: nested {row['mae']:.2f} / 止め時の規則 {row['mae_stop_rule']:.2f} / "
                f"採用 {row['chosen_counts']} ({row['seconds']} 秒)",
                flush=True,
            )
    finally:
        model.POOL_MODE = original

    rows = sorted(entry["seeds"], key=lambda r: r["seed"])
    maes = [r["mae"] for r in rows]
    stop = [r["mae_stop_rule"] for r in rows]
    entry.update(
        mae=statistics.mean(maes),
        mae_per_seed=maes,
        mae_spread=max(maes) - min(maes),
        mae_stop_rule=statistics.mean(stop),
        mae_stop_rule_per_seed=stop,
        mae_fixed_lr={
            k: statistics.mean(r["mae_fixed_lr"][k] for r in rows) for k in rows[0]["mae_fixed_lr"]
        },
        n_examples=len(examples),
        n_sites=len({e.code for e in examples}),
        n_groups=len({e.group for e in examples}),
    )
    save(report)
    return entry


def main() -> int:
    baseline = load("baseline.json")
    phase = load("phase.json")
    opponent_mae = baseline["systems"][baseline["best_two_feature_system"]]["mae"]

    report = {
        "generated_from": "pipeline/nested_experiment.py",
        "status": "running",
        "max_epochs": experiment.MAX_EPOCHS,
        "seeds": list(experiment.SEEDS),
        "lr_candidates": list(experiment.LR_CANDIDATES),
        "stop_fraction": experiment.NESTED_STOP_FRACTION,
        "select_fraction": experiment.NESTED_SELECT_FRACTION,
        "opponent_mae": opponent_mae,
        "l7": {
            "harmonic_mae": phase["systems"]["cnn_harmonic"]["mae"],
            "harmonic_spread": phase["systems"]["cnn_harmonic"]["mae_spread"],
        },
        "ch4_systems": {},
        "nested": {},
    }
    if OUT.exists():
        earlier = json.loads(OUT.read_text(encoding="utf-8"))
        for field in ("reproduction", "ch4_systems", "nested", "unit"):
            if field in earlier:
                report[field] = earlier[field]
        print(f"再開: CH4 {sorted(report['ch4_systems'])} / nested {sorted(report['nested'])}", flush=True)
    save(report)  # 書き出し経路を先に通す(loop_002 GEN-LOGIC)

    co2 = curves.build_examples("co2")
    x1, y1 = model.to_arrays(co2)

    # 一単位を測ってから本番規模へ入る(HC-207 / HC-213)
    if "unit" not in report:
        original = model.POOL_MODE
        model.POOL_MODE = "harmonic"
        try:
            unit = experiment.measure_unit(co2, x1, y1, channels=1)
        finally:
            model.POOL_MODE = original
        runs = len(experiment.LR_CANDIDATES) * len(experiment.SEEDS)
        report["unit"] = {
            "seconds_per_epoch_all_folds": unit,
            "nested_runs_per_system": runs,
            "estimate_hours_per_system": unit * experiment.MAX_EPOCHS * runs / 3600,
        }
        save(report)
    print(
        f"一単位 {report['unit']['seconds_per_epoch_all_folds'] * 1000:.0f} ms → "
        f"nested 1 系あたり見積もり {report['unit']['estimate_hours_per_system']:.1f} 時間",
        flush=True,
    )

    ch4_controls(report)

    print(f"nested: {len(co2)} 地点年 / {len({e.group for e in co2})} 群", flush=True)
    harmonic = nested_system(report, "harmonic", co2, x1, y1, "harmonic")
    invariant = nested_system(report, "invariant", co2, x1, y1, "invariant")

    # 判定は**作った時点で** Python の bool に落とす。平均は numpy.float64 なので比較は numpy.bool になり、
    # 変換を書き出し関数の中だけに置くと、表示のような別の出口で落ちる(loop_008 GEN-LOGIC)
    report["verdicts"] = {
        "p07_numerically": bool(harmonic["mae"] < opponent_mae),
        "p07_beats_beyond_spread": bool(opponent_mae - harmonic["mae"] > harmonic["mae_spread"]),
        "p08": bool(harmonic["mae_spread"] < phase["systems"]["cnn_harmonic"]["mae_spread"]),
        "p09": bool(harmonic["mae"] < harmonic["mae_stop_rule"]),
        "p10_numerically": bool(report["ch4_decomposition"]["p10_holds_numerically"]),
        "p10_identified": bool(report["ch4_decomposition"]["p10_identified"]),
        "p11": bool(harmonic["mae"] < invariant["mae"]),
    }
    report["status"] = "complete"
    save(report)

    print()
    print(f"相手 {opponent_mae:.2f} / nested 位相あり {harmonic['mae']:.2f}(幅 {harmonic['mae_spread']:.2f})"
          f" / 位相なし {invariant['mae']:.2f}")
    print(f"止め時の規則なら 位相あり {harmonic['mae_stop_rule']:.2f}")
    # 判定は numpy の比較から来るので numpy.bool が混ざる。表示も書き出しと同じ変換を通す
    # (loop_008: 報告は書き終えていたのに、この 1 行で終了コード 1 になった)
    print(json.dumps(experiment._jsonable(report["verdicts"]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
