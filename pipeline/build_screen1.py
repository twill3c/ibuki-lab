"""画面①「呼吸する地球」のデータを書き出す(L4)。

出すのは二つ。

- **固定地点の気候値**(65 地点)。学習に使ったのと同じ作り方の系列を平均したもの。
  緯度 × 半月の帯にすると、北で深く・赤道で浅く・南で逆向きの脈が一枚に出る
- **太平洋の緯度梯子**(POC 14 帯)。同じ船・同じ海・同じ研究室で緯度だけを振った列で、
  振幅が赤道で潰れて南で戻る形をそのまま見せられる

**どちらも `*_event.txt` 由来で、NOAA の平滑を経ていない**(G-06)。画面にもそう書く。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from pipeline import curves, ingest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "data" / "screen1.json"

#: 帯の年数がこれ未満なら「記録が短い」と印を付ける。画面で薄く出すため。
SHORT_RECORD_YEARS = 12

DIGITS = 4


def _round(value):
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return round(float(value), DIGITS)


def bin_labels() -> list:
    """半月ビンの表示名。年内の位置から機械的に作る(図のラベルは図と同じ源から — HC-045)。"""
    labels = []
    for i in range(curves.N_BINS):
        start = i / curves.N_BINS * 12
        month = int(start) + 1
        half = "前" if start - int(start) < 0.5 else "後"
        labels.append(f"{month}月{half}")
    return labels


def phase_of(values) -> dict:
    """谷と山がどのビンに来るか。位相の反転を絵で言うための量。"""
    low = min(range(len(values)), key=lambda i: values[i])
    high = max(range(len(values)), key=lambda i: values[i])
    return {"trough_bin": low, "peak_bin": high}


def site_rows(gas: str) -> list:
    names = {}
    for site in ingest.build_sites(gas):
        if site.lat_source == "header":
            names[site.code] = site.name or site.code

    rows = []
    for row in curves.site_climatology(gas):
        values = list(row["values"])
        rows.append(
            {
                "code": row["code"],
                "name": names.get(row["code"], row["code"]),
                "latitude": _round(row["latitude"]),
                "values": _round(values),
                "amplitude": _round(max(values) - min(values)),
                "years": row["years"],
                "year_min": row["year_min"],
                "year_max": row["year_max"],
                **phase_of(values),
            }
        )
    return rows


def band_rows(gas: str) -> list:
    rows = []
    for row in curves.band_climatology(gas):
        values = list(row["values"])
        rows.append(
            {
                "code": row["code"],
                "parent": row["parent"],
                "latitude": _round(row["latitude"]),
                "values": _round(values),
                "amplitude": _round(max(values) - min(values)),
                "years": row["years"],
                "samples_per_bin_min": row["samples_per_bin_min"],
                "short_record": row["years"] < SHORT_RECORD_YEARS,
                **phase_of(values),
            }
        )
    return rows


def main() -> int:
    sites = site_rows("co2")
    bands = band_rows("co2")
    coverage = curves.coverage_report("co2")

    payload = {
        "generated_from": "pipeline/build_screen1.py",
        "n_bins": curves.N_BINS,
        "bin_labels": bin_labels(),
        "unit": "ppm(年内の直線を差し引いた偏差)",
        "sites": sites,
        "bands": bands,
        "coverage": coverage,
        "provenance": {
            "source": "NOAA GML surface-flask event data (CC0 1.0)",
            "smoothed": False,
            "note": (
                "NOAA の月次ファイルは Thoning 平滑を経た値だが、ここに出しているのは "
                "event(保持された生の測定結果)から組んだ曲線である。補間もしていない"
            ),
            "site_curves": "採用した地点年(24 ビンすべてが実測)の平均",
            "band_curves": (
                "周航は一年で 24 ビンを埋められないので、複数年をまとめた気候値。"
                "年内平均からの偏差をビンごとに平均している —— 地点の曲線とは作り方が違う"
            ),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    poc = [b for b in bands if b["parent"] == "poc"]
    print(f"地点 {len(sites)} / 帯 {len(bands)}(うち POC {len(poc)})")
    print(f"振幅の範囲: 地点 {min(s['amplitude'] for s in sites):.2f} 〜 "
          f"{max(s['amplitude'] for s in sites):.2f} ppm")
    print(f"太平洋の梯子: 30N {next(b['amplitude'] for b in poc if b['latitude'] == 30):.2f} → "
          f"赤道 {next(b['amplitude'] for b in poc if b['latitude'] == 0):.2f} → "
          f"35S {next(b['amplitude'] for b in poc if b['latitude'] == -35):.2f} ppm")
    north = [s for s in sites if s["latitude"] > 45]
    south = [s for s in sites if s["latitude"] < -45]
    print(f"谷のビン: 北緯45度以北の中央値 {sorted(s['trough_bin'] for s in north)[len(north)//2]} / "
          f"南緯45度以南 {sorted(s['trough_bin'] for s in south)[len(south)//2]}"
          f"(24 ビン中。半年ずれていれば位相が逆)")
    print(f"{OUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
