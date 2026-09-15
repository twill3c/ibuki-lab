"""L10 の測定。未測定のまま残っていた P-04 と P-05 を、SPEC §7.19 に測る前に書いた線で判定する。

| 系 | 資料形 | 母集団 | 何を測るか |
|---|---|---|---|
| `cnn_invariant_805` | event | 805 | P-04(地点別誤差の緯度帯)・L2 の再現の検算 |
| `knn_805` | event | 805 | P-04 の参考(判定には使わない) |
| `cnn_event_monthpop` / `knn_event_monthpop` | event | 月次もそろう地点年 | P-05 の event 側 |
| `cnn_month_monthpop` / `knn_month_monthpop` | **月次(平滑済み)** | 同上 | P-05 の月次側 |

**月次の曲線を学習に使うのはこのスクリプトだけである**(§7.19 の G-06 の例外)。
出荷する模型の学習経路は event 由来のままで、月次を読まないことは T-068 が走査で固定する。

高価な計算は落ちる前提で組む。系ごと・種ごとに書き出し、再開できる。
"""
from __future__ import annotations

import dataclasses
import json
import random
import statistics
import time
from pathlib import Path

from pipeline import baseline, curves, experiment, external, model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "closing.json"

#: 緯度帯の境(度)。北回帰線・南回帰線。
TROPIC = 23.44
BANDS = ("北の中高緯度", "熱帯", "南の中高緯度")

#: 置換検定の回数と種。**測る前に決めて動かさない**(§7.19)。
PERMUTATIONS = 10_000
PERMUTATION_SEED = 20260915

#: 再現の検算の許容(度)。L2 と同じ計算を同じ機械で回すので、本来は一致する。
REPRODUCTION_TOLERANCE = 0.05


# ------------------------------------------------------------------ 純粋な部品(T-065〜T-067)

def band_of(latitude: float) -> str:
    """地点の緯度を三つの帯に排他に分ける。境の値は中高緯度の側に入れる。"""
    if latitude >= TROPIC:
        return BANDS[0]
    if latitude <= -TROPIC:
        return BANDS[2]
    return BANDS[1]


def site_errors(examples, preds_by_seed, labels) -> dict:
    """地点ごとに、すべての種・すべての年の絶対誤差を平均する。返り値は {地点: (緯度, 誤差)}。"""
    collected: dict[str, tuple] = {}
    for preds in preds_by_seed:
        assert len(preds) == len(examples) == len(labels), "予測と例の数が合わない"
        for ex, p, t in zip(examples, preds, labels):
            collected.setdefault(ex.code, (ex.latitude, []))[1].append(abs(float(p) - float(t)))
    return {code: (lat, statistics.mean(errs)) for code, (lat, errs) in collected.items()}


def band_summary(errors: dict) -> dict:
    grouped: dict[str, list] = {b: [] for b in BANDS}
    for _code, (lat, err) in sorted(errors.items()):
        grouped[band_of(lat)].append(err)
    return {
        b: {"sites": len(v), "mean": statistics.mean(v) if v else None}
        for b, v in grouped.items()
    }


def _statistic(values, bands) -> float:
    """S = min(熱帯の平均, 南の平均) − 北の平均。帯が空なら計算しない。"""
    sums = {b: 0.0 for b in BANDS}
    counts = {b: 0 for b in BANDS}
    for v, b in zip(values, bands):
        sums[b] += v
        counts[b] += 1
    assert all(counts[b] for b in BANDS), "空の緯度帯がある"
    mean = {b: sums[b] / counts[b] for b in BANDS}
    return min(mean[BANDS[1]], mean[BANDS[2]]) - mean[BANDS[0]]


def permutation_test(errors: dict, repeats: int = PERMUTATIONS, seed: int = PERMUTATION_SEED) -> dict:
    """地点の帯の割り当てを入れ替える置換検定。片側(S が観測値以上になる割合)。"""
    codes = sorted(errors)
    values = [errors[c][1] for c in codes]
    bands = [band_of(errors[c][0]) for c in codes]
    observed = _statistic(values, bands)
    rng = random.Random(seed)
    pool = list(bands)
    at_least = 0
    for _ in range(repeats):
        rng.shuffle(pool)
        if _statistic(values, pool) >= observed:
            at_least += 1
    return {"statistic": observed, "p": (at_least + 1) / (repeats + 1), "repeats": repeats, "seed": seed}


def monthly_pairs(event_examples, month_curves: dict):
    """event の例のうち、同じ地点・同じ年の月次の曲線がそろうものを対にする。

    返り値は (event 側の例, 月次側の例)。月次側は **値だけを月次の曲線に差し替えた** 写しで、
    地点・年・群・緯度は event 側と同じ。並びも同じにする(LOGO の fold が一致するように)。
    """
    by_key = {
        (code, year): curve
        for code, entries in month_curves.items()
        for year, curve in entries
    }
    event_side, month_side = [], []
    for ex in event_examples:
        curve = by_key.get((ex.code, ex.year))
        if curve is None:
            continue
        assert len(curve) == curves.N_BINS, "月次の曲線の長さが 24 でない"
        event_side.append(ex)
        month_side.append(
            dataclasses.replace(
                ex,
                values=tuple(float(v) for v in curve),
                n_samples=0,
                filled_bins=curves.N_BINS,
                interpolated=0,
            )
        )
    return event_side, month_side


# ------------------------------------------------------------------ 測定

def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def save(report: dict) -> None:
    OUT.write_text(
        json.dumps(experiment._jsonable(report), ensure_ascii=False) + "\n", encoding="utf-8"
    )


def cnn_system(report: dict, label: str, examples) -> dict:
    """位相なしの 1D CNN を種 3 で LOGO。種ごとの予測を残す。"""
    entry = report["systems"].setdefault(label, {"kind": "cnn", "seeds": {}})
    x, y = model.to_arrays(examples)
    for seed in experiment.SEEDS:
        if str(seed) in entry["seeds"]:
            print(f"  済み(再開): {label} seed={seed}", flush=True)
            continue
        started = time.time()
        result = experiment.run_logo(examples, x, y, 1, seed, strict_budget=False)
        mae = float(experiment._per_site_mae(examples, range(len(examples)), result["preds"], y))
        entry["seeds"][str(seed)] = {
            "preds": [round(float(p), 5) for p in result["preds"]],
            "mae": mae,
            "folds_at_budget": int(result["folds_at_budget"]),
            "seconds": round(time.time() - started),
        }
        save(report)
        print(f"  {label} seed={seed}: 地点MAE {mae:.2f}({entry['seeds'][str(seed)]['seconds']} 秒)", flush=True)
    maes = [entry["seeds"][str(s)]["mae"] for s in experiment.SEEDS]
    entry.update(
        mae=statistics.mean(maes),
        mae_per_seed=maes,
        mae_spread=max(maes) - min(maes),
        n_examples=len(examples),
        n_sites=len({e.code for e in examples}),
        n_groups=len({e.group for e in examples}),
    )
    save(report)
    return entry


def knn_system(report: dict, label: str, examples) -> dict:
    """2 特徴 kNN(L1 と同じ k の選び方)。決定的なので一度だけ測る。"""
    if label in report["systems"]:
        return report["systems"][label]
    labels = [ex.latitude for ex in examples]
    preds = baseline._predictions("harmonic_knn", examples, labels)
    scored = baseline._score(examples, labels, preds)
    entry = {
        "kind": "knn",
        "preds": [round(float(p), 5) for p in preds],
        "mae": float(scored["mae"]),
        "n_examples": len(examples),
        "n_sites": scored["sites"],
    }
    report["systems"][label] = entry
    save(report)
    print(f"  {label}: 地点MAE {entry['mae']:.2f}", flush=True)
    return entry


def main() -> int:
    previous = load("model.json")
    experiment.LEARNING_RATE = previous["learning_rate"]
    assert model.POOL_MODE == "invariant", "L2 と同じ位相なしで測る"

    report = {
        "generated_from": "pipeline/closing_experiment.py",
        "status": "running",
        "learning_rate": experiment.LEARNING_RATE,
        "max_epochs": experiment.MAX_EPOCHS,
        "seeds": list(experiment.SEEDS),
        "tropic_deg": TROPIC,
        "systems": {},
    }
    if OUT.exists():
        earlier = json.loads(OUT.read_text(encoding="utf-8"))
        for field in ("systems", "unit"):
            if field in earlier:
                report[field] = earlier[field]
        print(f"再開: 済み {sorted(report['systems'])}", flush=True)
    save(report)  # 書き出し経路を先に通す

    co2 = list(curves.build_examples("co2"))
    labels = [ex.latitude for ex in co2]

    # P-05 の母集団を先に決める。月次が全例でそろうなら event 側は P-04 と**同じ計算**になるので
    # 学習し直さずに使い回す(並びと値が同一であることを確かめてから)
    event_side, month_side = monthly_pairs(co2, external.noaa_monthly_curves("co2"))
    same_as_805 = [(e.code, e.year, e.values) for e in event_side] == [(e.code, e.year, e.values) for e in co2]
    cnn_runs = 6 if same_as_805 else 9

    # 一単位を測ってから本番規模へ入る(HC-207 / HC-213)
    if "unit" not in report:
        x1, y1 = model.to_arrays(co2)
        unit = experiment.measure_unit(co2, x1, y1, channels=1)
        report["unit"] = {
            "seconds_per_epoch_all_folds": unit,
            "cnn_runs": cnn_runs,
            "estimate_hours": unit * experiment.MAX_EPOCHS * cnn_runs / 3600,
        }
        save(report)
    print(f"一単位 {report['unit']['seconds_per_epoch_all_folds'] * 1000:.0f} ms → "
          f"見積もり {report['unit']['estimate_hours']:.1f} 時間(並行セッション次第で 3〜4 倍)", flush=True)

    # --- P-04 ---------------------------------------------------------------
    print(f"P-04: {len(co2)} 地点年", flush=True)
    cnn805 = cnn_system(report, "cnn_invariant_805", co2)
    expected = previous["systems"]["cnn_co2"]["mae_per_seed"]
    deltas = [a - b for a, b in zip(cnn805["mae_per_seed"], expected)]
    report["reproduction"] = {"expected": expected, "measured": cnn805["mae_per_seed"], "deltas": deltas,
                              "tolerance": REPRODUCTION_TOLERANCE}
    save(report)
    if max(abs(d) for d in deltas) >= REPRODUCTION_TOLERANCE:
        raise RuntimeError(f"L2 の値を再現しない(差 {deltas})。土俵が違うので判定しない")

    knn805 = knn_system(report, "knn_805", co2)
    cnn_errors = site_errors(co2, [cnn805["seeds"][str(s)]["preds"] for s in experiment.SEEDS], labels)
    knn_errors = site_errors(co2, [knn805["preds"]], labels)
    test = permutation_test(cnn_errors)
    bands = band_summary(cnn_errors)
    north, tropics, south = (bands[b]["mean"] for b in BANDS)
    report["p04"] = {
        "cnn_bands": bands,
        "knn_bands": band_summary(knn_errors),
        "cnn_site_errors": {c: {"latitude": lat, "mae": e} for c, (lat, e) in sorted(cnn_errors.items())},
        "permutation": test,
        "holds_numerically": bool(tropics > north and south > north),
        "sayable": bool(tropics > north and south > north and test["p"] < 0.05),
    }
    save(report)

    # --- P-05 ---------------------------------------------------------------
    changed = sum(1 for a, b in zip(event_side, month_side) if a.values != b.values)
    print(f"P-05: 月次もそろう地点年 {len(event_side)} / 値が差し替わった例 {changed} / "
          f"event 側は 805 と同一: {same_as_805}", flush=True)
    assert changed == len(month_side), "月次の曲線に差し替わっていない例がある"
    report["p05_population"] = {
        "n_examples": len(event_side),
        "n_sites": len({e.code for e in event_side}),
        "n_groups": len({e.group for e in event_side}),
        "changed": changed,
        "event_side_identical_to_805": same_as_805,
    }
    save(report)

    if same_as_805:
        knn_event, cnn_event = knn805, cnn805
    else:
        knn_event = knn_system(report, "knn_event_monthpop", event_side)
        cnn_event = cnn_system(report, "cnn_event_monthpop", event_side)
    knn_month = knn_system(report, "knn_month_monthpop", month_side)
    cnn_month = cnn_system(report, "cnn_month_monthpop", month_side)

    gap_event = cnn_event["mae"] - knn_event["mae"]
    gap_month = cnn_month["mae"] - knn_month["mae"]
    margin = max(cnn_event["mae_spread"], cnn_month["mae_spread"])
    report["p05"] = {
        "gap_event": gap_event,
        "gap_month": gap_month,
        "narrowing": gap_event - gap_month,
        "spread_margin": margin,
        "holds_numerically": bool(gap_month < gap_event),
        "sayable": bool(gap_event - gap_month > margin),
    }
    report["status"] = "complete"
    save(report)

    print()
    print(json.dumps(experiment._jsonable({"p04": {k: v for k, v in report["p04"].items()
                                                   if k not in ("cnn_site_errors",)},
                                           "p05": report["p05"]}), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
