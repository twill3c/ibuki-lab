"""画面③「当ててみる」のデータを書き出す(L6)。

選べる曲線を三群に分けて渡す。**学習に使ったかどうかを群ごとに明示する** ——
使った地点で当たるのは当たり前で、見どころは使っていない側だからである。

| 群 | 学習に使ったか |
|---|---|
| NOAA の固定地点(65) | **使った** |
| 太平洋の帯(14) | 使っていない(一年で 24 ビンが埋まらないので学習から外れた) |
| 気象庁(3) | 使っていない(外部ホールドアウト) |
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline import curves, ingest, jma

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "data" / "screen3.json"

DIGITS = 3


def _round(value):
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return round(float(value), DIGITS)


def jma_climatology(code: str) -> list:
    """気象庁地点の気候値。採用した年の平均を取る。"""
    rows = list(jma.annual_curves(code))
    n = curves.N_BINS
    return [sum(curve[i] for _y, curve in rows) / len(rows) for i in range(n)]


def main() -> int:
    names = {
        s.code: (s.name or s.code)
        for s in ingest.build_sites("co2")
        if s.lat_source == "header"
    }

    choices = []
    for row in curves.site_climatology("co2"):
        choices.append(
            {
                "code": row["code"],
                "label": f"{names.get(row['code'], row['code'])}",
                "group": "NOAA の固定地点",
                "latitude": _round(row["latitude"]),
                "values": _round(list(row["values"])),
                "in_training": True,
                "note": f"{row['year_min']}-{row['year_max']} の {row['years']} 年ぶんの平均",
            }
        )

    for row in curves.band_climatology("co2"):
        if row["parent"] != "poc":
            continue
        choices.append(
            {
                "code": row["code"],
                "label": f"太平洋 {abs(row['latitude']):.0f}°{'N' if row['latitude'] >= 0 else 'S'}",
                "group": "太平洋の帯(学習に使っていない)",
                "latitude": _round(row["latitude"]),
                "values": _round(list(row["values"])),
                "in_training": False,
                "note": f"{row['years']} 年ぶんをまとめた気候値。一年では 24 ビンが埋まらないので学習から外れた",
            }
        )

    for code, station in jma.STATIONS.items():
        cov = jma.coverage(code)
        choices.append(
            {
                "code": f"jma_{code}",
                "label": station["name"],
                "group": "気象庁(外部ホールドアウト)",
                "latitude": _round(station["latitude"]),
                "values": _round(jma_climatology(code)),
                "in_training": False,
                "note": f"{cov['year_min']}-{cov['year_max']} の {cov['years_accepted']} 年ぶんの平均。{station['note']}",
            }
        )

    payload = {
        "generated_from": "pipeline/build_screen3.py",
        "n_bins": curves.N_BINS,
        "choices": choices,
        "default_code": "jma_mnm",
        "provenance": {
            "note": (
                "曲線はいずれも年内の直線を差し引いた偏差(ppm)。NOAA は event 由来、"
                "気象庁は月平均を半月へ反復で広げたもの"
            ),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    groups: dict[str, int] = {}
    for choice in choices:
        groups[choice["group"]] = groups.get(choice["group"], 0) + 1
    for group, count in groups.items():
        print(f"  {group}: {count} 件")
    print(f"{OUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
