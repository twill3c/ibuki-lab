"""画面②「積み上がる息」のデータを書き出す(L5)。

見せるのは二つ。

- **積み上がり**。1970 年代からの上昇を、生の測定結果のまま並べる。NOAA の平滑済み
  月次ではなく `*_event.txt` の保持標本を使う(G-06)
- **一呼吸**。長期の上昇を差し引くと、毎年同じ形の脈が残る

あわせて **気象庁 3 地点での外部検証**(F-08)の材料を渡す。こちらは
`pipeline/external.py` が作った `data/external.json` をそのまま読む。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from pipeline import curves, ingest, jma

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "data" / "screen2.json"

#: 画面に出す地点。長い記録があり、緯度が離れているものを選ぶ。
FEATURED = ("brw", "mlo", "spo")

#: 上昇の見せ方に使う多項式の次数。**曲線を当てはめるのではなく、
#: 「長期の上がり」だけを取り出すための最小限**。二次にすると加速も表せる。
TREND_DEGREE = 2

#: 書き出す桁。**濃度は 0.1 ppm 刻みで測られているので 2 桁で足りる。**
#: 時刻は 4 桁(約 1 時間)あれば年内の順序が保てる。桁を落とすのは飾りではなく、
#: 出荷物を小さくするため —— 3 桁のままだと 748 KB だった。
VALUE_DIGITS = 2
TIME_DIGITS = 4


def _round(value, digits: int = VALUE_DIGITS):
    if isinstance(value, (list, tuple)):
        return [_round(v, digits) for v in value]
    return round(float(value), digits)


def polyfit(xs, ys, degree: int) -> list:
    """正規方程式で多項式を当てる。外部依存を増やさないため自前で解く。"""
    n = degree + 1
    matrix = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    rhs = [sum((x**i) * y for x, y in zip(xs, ys)) for i in range(n)]

    a = [matrix[i][:] + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("特異な設計行列")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def evaluate(coef, x: float) -> float:
    return sum(c * x**i for i, c in enumerate(coef))


def site_series(gas: str = "co2") -> list:
    """地点ごとの (時刻, 値) と、当てはめた長期の上がり。

    標本は保持されたものだけ。同一時刻(フラスコ対)は先に平均する ——
    §2.3 の手順 1 と同じ順序で、点の重みを揃える。
    """
    sites = {s.code: s for s in ingest.build_sites(gas)}
    out = []
    for series in ingest.training_series(gas):
        if series.code not in FEATURED:
            continue
        by_time: dict[float, list] = {}
        for time_decimal, value in series.values:
            by_time.setdefault(time_decimal, []).append(value)
        points = sorted((t, sum(v) / len(v)) for t, v in by_time.items())

        xs = [t for t, _ in points]
        ys = [v for _, v in points]
        origin = xs[0]
        coef = polyfit([x - origin for x in xs], ys, TREND_DEGREE)
        trend = [evaluate(coef, x - origin) for x in xs]

        site = sites[series.code]
        out.append(
            {
                "code": series.code,
                "name": site.name or series.code,
                "latitude": _round(site.latitude),
                "times": _round(xs, TIME_DIGITS),
                "values": _round(ys),
                "trend": _round(trend),
                "n_samples": len(points),
                "year_min": math.floor(xs[0]),
                "year_max": math.floor(xs[-1]),
                "rise_ppm": _round(trend[-1] - trend[0]),
                "amplitude_ppm": _round(
                    max(y - t for y, t in zip(ys, trend)) - min(y - t for y, t in zip(ys, trend))
                ),
            }
        )
    return sorted(out, key=lambda r: -r["latitude"])


def jma_series() -> list:
    """気象庁 綾里の月平均。**外部ホールドアウトの実物**を並べて見せる。"""
    out = []
    for code in ("ryo",):
        rows = jma.read_monthly(code)
        points = sorted(rows.items())
        xs = [y + (m - 0.5) / 12 for (y, m), _ in points]
        ys = [v for _, v in points]
        coef = polyfit([x - xs[0] for x in xs], ys, TREND_DEGREE)
        trend = [evaluate(coef, x - xs[0]) for x in xs]
        station = jma.STATIONS[code]
        out.append(
            {
                "code": code,
                "name": station["name"],
                "latitude": _round(station["latitude"]),
                "times": _round(xs, TIME_DIGITS),
                "values": _round(ys),
                "trend": _round(trend),
                "n_samples": len(points),
                "year_min": math.floor(xs[0]),
                "year_max": math.floor(xs[-1]),
                "rise_ppm": _round(trend[-1] - trend[0]),
                "amplitude_ppm": _round(
                    max(y - t for y, t in zip(ys, trend)) - min(y - t for y, t in zip(ys, trend))
                ),
            }
        )
    return out


def main() -> int:
    external = json.loads((ROOT / "data" / "external.json").read_text(encoding="utf-8"))

    payload = {
        "generated_from": "pipeline/build_screen2.py",
        "trend_degree": TREND_DEGREE,
        "noaa": site_series(),
        "jma": jma_series(),
        "external": {
            "systems": {
                name: {k: v for k, v in system.items() if k != "sites"}
                for name, system in external["systems"].items()
            },
            "jma_sites": external["systems"]["jma_holdout"]["sites"],
            "decomposition": external["decomposition"],
            "model": external["model"],
            "stations": external["stations"],
            "coverage": external["jma_coverage"],
        },
        "provenance": {
            "noaa": "NOAA GML surface-flask **event**(保持標本・平滑なし)",
            "jma": "気象庁 月平均値。**学習にも閾値の決定にも使っていない外部ホールドアウト**",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for row in payload["noaa"] + payload["jma"]:
        print(
            f"  {row['name'][:28]:>28} 緯度 {row['latitude']:>7.2f} / "
            f"{row['year_min']}-{row['year_max']} / 標本 {row['n_samples']:>5} / "
            f"上昇 {row['rise_ppm']:>6.1f} ppm / 一呼吸 {row['amplitude_ppm']:>5.1f} ppm"
        )
    print(f"{OUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
