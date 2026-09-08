"""外部ホールドアウトでの検証(L5 / G-11)。

出荷模型(L3)を、**学習のどの段階でも見ていない気象庁 3 地点**に当てる。

そのままでは何を測ったのか言えない。気象庁が配るのは月平均で、模型が学習したのは
event 由来の半月曲線だからである。**外れたとして、網が違うからか、資料形が違うからか。**
分けるために三つを並べる。

| 系 | 網 | 資料形 | 何を測るか |
|---|---|---|---|
| `noaa_event_reference` | NOAA | event | 同じ模型の基準(**学習に使った例なので楽観的**) |
| `noaa_monthly_control` | NOAA | 月次 | 基準との差が**資料形の効果** |
| `jma_holdout` | 気象庁 | 月平均 | 対照との差が**網の効果** |

NOAA の二つは学習に使った例なので絶対値は楽観的だが、**同じ例の同じ模型**なので
差は資料形の効果を分離できる。気象庁の絶対値は L2 の leave-one-group-out 実測
(地点あたり MAE 12.40 度)と比べる —— これが未知の地点に対する見込みだからである。
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pipeline import curves, ingest, jma, model

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data" / "model_weights.json"
OUT = ROOT / "data" / "external.json"


def load_weights() -> dict:
    """出荷した重みを JSON から読む。**学習し直さない。**"""
    raw = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    return {
        name: jnp.asarray(np.asarray(raw[name], dtype=np.float32))
        for name in raw
        if name != "metadata"
    }, raw["metadata"]


def predict(params, curve) -> float:
    x = np.asarray([[[v] for v in curve]], dtype=np.float32)
    return float(np.asarray(model.forward(params, jnp.asarray(x)))[0])


# ------------------------------------------------------------------ NOAA の月次

def noaa_monthly_curves(gas: str = "co2") -> dict:
    """NOAA の `*_month.txt` から、気象庁と**同じ処理**で年ごとの曲線を作る。

    月平均 12 点 → 半月 24 点へ反復で広げ、直線を差し引く。
    こうすると気象庁との違いが「網」だけになる。

    **これは平滑済みのデータである**(§2.2 の Thoning 当てはめ)。学習には使わない ——
    ここで使うのは資料形の効果を測るためだけで、`curves.py` は今もこの経路を持たない。
    """
    out: dict[str, list] = {}
    for site in ingest.build_sites(gas):
        if site.lat_source != "header":
            continue
        path = ingest.month_path(gas, site.code)
        rows: dict[tuple, float] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 4:
                continue
            rows[(int(fields[1]), int(fields[2]))] = float(fields[3])

        years = sorted({y for (y, _m) in rows})
        for year in years:
            months = [rows.get((year, m)) for m in range(1, 13)]
            if any(v is None for v in months):
                continue
            curve = curves.detrend(jma.months_to_bins([float(v) for v in months]))
            out.setdefault(site.code, []).append((year, list(curve)))
    return out


def noaa_event_curves(gas: str = "co2") -> dict:
    """学習に使ったのと同じ event 由来の曲線。地点ごとにまとめ直すだけ。"""
    out: dict[str, list] = {}
    for ex in curves.build_examples(gas):
        out.setdefault(ex.code, []).append((ex.year, list(ex.values)))
    return out


# ------------------------------------------------------------------ 評価

def score(params, by_site: dict, latitudes: dict, source: str) -> dict:
    """地点あたりの MAE。地点ごとに年を平均してから地点間で平均する(L2 と同じ指標)。"""
    per_site = []
    rows = []
    for code, entries in sorted(by_site.items()):
        truth = latitudes[code]
        errors = []
        predictions = []
        for _year, curve in entries:
            prediction = predict(params, curve)
            predictions.append(prediction)
            errors.append(abs(prediction - truth))
        per_site.append(statistics.mean(errors))
        rows.append(
            {
                "code": code,
                "true_latitude": round(truth, 4),
                "predicted_latitude": round(statistics.mean(predictions), 2),
                "predicted_sd": round(statistics.pstdev(predictions), 2) if len(predictions) > 1 else 0.0,
                "mae": round(statistics.mean(errors), 2),
                "years": len(entries),
            }
        )
    return {
        "source": source,
        "mae": statistics.mean(per_site),
        "median_ae": statistics.median(per_site),
        "worst_site_ae": max(per_site),
        "n_sites": len(per_site),
        "n_examples": sum(len(v) for v in by_site.values()),
        "sites": rows,
    }


def main() -> int:
    params, metadata = load_weights()
    logo = metadata["generalisation_estimate_mae"]

    fixed = {
        s.code: s.latitude for s in ingest.build_sites("co2") if s.lat_source == "header"
    }

    event = noaa_event_curves()
    monthly = {c: v for c, v in noaa_monthly_curves().items() if c in event}
    # 対照と基準は**同じ地点**に載せる。母集団が違えば差の内訳が読めない(HC-064)
    shared = sorted(set(event) & set(monthly))
    event = {c: event[c] for c in shared}
    monthly = {c: monthly[c] for c in shared}

    jma_curves = {code: list(jma.annual_curves(code)) for code in jma.STATIONS}
    jma_lat = {code: st["latitude"] for code, st in jma.STATIONS.items()}

    systems = {
        "noaa_event_reference": score(params, event, fixed, "event"),
        "noaa_monthly_control": score(params, monthly, fixed, "month"),
        "jma_holdout": score(params, jma_curves, jma_lat, "month"),
    }

    product_effect = systems["noaa_monthly_control"]["mae"] - systems["noaa_event_reference"]["mae"]
    network_effect = systems["jma_holdout"]["mae"] - systems["noaa_monthly_control"]["mae"]

    report = {
        "generated_from": "pipeline/external.py",
        "model": {
            "weights": "data/model_weights.json",
            "trained_on_all_examples": metadata["trained_on_all_examples"],
            "generalisation_estimate_mae": logo,
            "note": (
                "NOAA の二系は**学習に使った例**なので絶対値は楽観的である。"
                "気象庁の絶対値と比べるのは L2 の leave-one-group-out 実測のほう"
            ),
        },
        "systems": systems,
        "decomposition": {
            "product_effect_deg": product_effect,
            "network_effect_deg": network_effect,
            "note": (
                "資料形の効果 = NOAA の月次 − NOAA の event(網が同じ・資料形だけ違う)。"
                "網の効果 = 気象庁 − NOAA の月次(資料形が同じ・網だけ違う)"
            ),
        },
        "jma_coverage": {code: jma.coverage(code) for code in jma.STATIONS},
        "stations": {
            code: {
                k: v
                for k, v in station.items()
                if k in ("name", "latitude", "longitude", "since", "source", "note")
            }
            for code, station in jma.STATIONS.items()
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'系':>26} {'MAE(地点)':>10} {'地点':>5} {'例':>6}")
    for name, s in systems.items():
        print(f"{name:>26} {s['mae']:>10.2f} {s['n_sites']:>5} {s['n_examples']:>6}")
    print()
    print(f"資料形の効果(月次 − event): {product_effect:+.2f} 度")
    print(f"網の効果(気象庁 − NOAA月次): {network_effect:+.2f} 度")
    print(f"L2 の leave-one-group-out 実測: {logo:.2f} 度")
    print()
    for row in systems["jma_holdout"]["sites"]:
        station = jma.STATIONS[row["code"]]
        print(
            f"  {station['name']:>6} 真 {row['true_latitude']:>6.2f} / "
            f"予測 {row['predicted_latitude']:>6.2f} (±{row['predicted_sd']:.1f}) / "
            f"誤差 {row['mae']:>5.2f} / {row['years']} 年"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
