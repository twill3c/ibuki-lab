"""気象庁の月平均 CO2 を読む(L5)。

**この系統は外部ホールドアウトである。** 学習にも、閾値の決定にも、学習率の選定にも
使っていない。それを保証するのは宣言ではなく、`tests/test_jma.py::T-048` の走査 ——
学習経路のモジュールがこのファイルを import していないことを機械的に確かめる。

気象庁が配るのは**月平均値 12 点**で、模型が学習した**半月 24 点の event 由来曲線**とは
形も資料の種類も違う。したがってここで測れるのは二つの移り方の**合成**である:

1. 別の観測網へ移れるか(気象庁は NOAA と独立に較正・運用されている)
2. 別の資料形へ移れるか(月平均は生の測定結果ではない)

分けるために `pipeline/external.py` が **NOAA 自身の月次ファイルへ同じ処理を当てた対照**を
置く。そちらは網が同じで資料形だけが違うので、差の内訳が読める。
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "jma"

#: 観測地点。座標は気象庁の観測地点一覧の**度分表記をそのまま**持ち、
#: 小数はそこから導く。二つを別々に持って突合するので、書き写しの誤りが残らない(T-047)。
STATIONS = {
    "ryo": {
        "name": "綾里",
        "name_en": "Ryori",
        "latitude_dms": (39, 2),
        "longitude_dms": (141, 49),
        "latitude": 39 + 2 / 60,
        "longitude": 141 + 49 / 60,
        "since": 1987,
        "source": "https://www.data.jma.go.jp/env/ghg_obs/station/",
        "note": "岩手県大船渡市。2002 年に観測点が移転し、標高が 230m → 260m に変わっている",
    },
    "mnm": {
        "name": "南鳥島",
        "name_en": "Minamitorishima",
        "latitude_dms": (24, 17),
        "longitude_dms": (153, 59),
        "latitude": 24 + 17 / 60,
        "longitude": 153 + 59 / 60,
        "since": 1993,
        "source": "https://www.data.jma.go.jp/env/ghg_obs/station/",
        "note": "東京から南東へ約 2,000 km の孤島。GAW 全球観測所",
    },
    "yon": {
        "name": "与那国島",
        "name_en": "Yonagunijima",
        "latitude_dms": (24, 28),
        "longitude_dms": (123, 1),
        "latitude": 24 + 28 / 60,
        "longitude": 123 + 1 / 60,
        "since": 1997,
        "source": "https://www.data.jma.go.jp/env/ghg_obs/station/",
        "note": "沖縄県八重山郡。観測は 2024 年 3 月末で終了。南鳥島とほぼ同緯度で経度が 31 度違う",
    },
}

#: 気象庁の CSV は cp932。UTF-8 で開くと化けたまま数字だけ読めてしまうので明示する。
ENCODING = "cp932"

#: 欠測の印。
MISSING = "--"

_ROW = re.compile(r"^\s*(\d{4})\s*,\s*(\d{1,2})\s*,\s*([^,]*?)\s*,?\s*$")


class JmaError(Exception):
    """気象庁データについての仮定が崩れたときに投げる。"""


def csv_path(code: str) -> Path:
    path = RAW / f"co2_monthave_{code}.csv"
    if not path.exists():
        raise JmaError(f"{path} が無い。README の取得手順を実行すること")
    return path


@functools.lru_cache(maxsize=None)
def read_monthly(code: str) -> dict:
    """`(年, 月) → ppm` を返す。欠測と註記行は落とす。

    末尾に日本語の註記が数行付くので、**行の形で選ぶ**(年, 月, 数値)。
    註記を数値として読み込まないことは、正規表現がそれを拒むことで担保する。
    """
    if code not in STATIONS:
        raise JmaError(f"知らない地点 {code}")

    out = {}
    seen_header = False
    for line in csv_path(code).read_text(encoding=ENCODING).splitlines():
        if not line.strip():
            continue
        if not seen_header:
            if line.startswith("年,"):
                seen_header = True
            continue
        match = _ROW.match(line)
        if not match:
            continue
        year, month, raw = int(match.group(1)), int(match.group(2)), match.group(3)
        if raw == MISSING or not raw:
            continue
        # 速報値には ')' が付く。値としては使うが、印は落とす
        cleaned = raw.replace(")", "").replace("(", "").strip()
        try:
            out[(year, month)] = float(cleaned)
        except ValueError:
            continue

    if not out:
        raise JmaError(f"{code}: 数値が 1 件も読めなかった(文字コードか書式が変わった)")
    return out


def months_to_bins(values) -> list:
    """月平均 12 点を半月 24 点へ広げる。**補間ではなく反復である。**

    補間すればそれは平滑であり、§2.2 で `*_month.txt` を避けたのと同じ問題が入る。
    各月の値をその月の 2 ビンに同じ値で置くだけにして、平滑を混ぜない(T-049)。
    """
    if len(values) != 12:
        raise JmaError(f"月平均が 12 点でない({len(values)} 点)")
    out = []
    for value in values:
        out.append(value)
        out.append(value)
    return out


def detrend(values) -> list:
    """最小二乗直線を差し引く。**学習側の `curves.detrend` と同じ式**である。

    別実装にすると、外れたときに『前処理が違うから』という逃げ道ができる。
    ここは学習側を呼ぶ —— ただし `curves` は気象庁を知らないままである(向きが逆なので
    T-048 の走査には掛からない)。
    """
    from pipeline import curves

    return list(curves.detrend(values))


@functools.lru_cache(maxsize=None)
def annual_curves(code: str) -> tuple:
    """`(年, 24 点の偏差)` の列。**12 か月すべてが揃った年だけ**を採る。

    欠けた月を埋めない。埋めれば平滑であり、落とした年は数えて出す(G-16 と同じ規律)。
    """
    rows = read_monthly(code)
    years = sorted({y for (y, _m) in rows})
    out = []
    for year in years:
        months = [rows.get((year, m)) for m in range(1, 13)]
        if any(v is None for v in months):
            continue
        out.append((year, detrend(months_to_bins([float(v) for v in months]))))
    return tuple(out)


def coverage(code: str) -> dict:
    rows = read_monthly(code)
    years = sorted({y for (y, _m) in rows})
    accepted = [y for y, _ in annual_curves(code)]
    return {
        "years_with_any_month": len(years),
        "years_accepted": len(accepted),
        "years_dropped_incomplete": len(years) - len(accepted),
        "year_min": min(accepted) if accepted else None,
        "year_max": max(accepted) if accepted else None,
        "months_total": len(rows),
    }


def main() -> int:
    for code, station in STATIONS.items():
        cov = coverage(code)
        print(
            f"{station['name']:>6}({code}) 緯度 {station['latitude']:.3f} / "
            f"採用 {cov['years_accepted']} 年 ({cov['year_min']}-{cov['year_max']}) / "
            f"欠測で落とした {cov['years_dropped_incomplete']} 年 / 月数 {cov['months_total']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
