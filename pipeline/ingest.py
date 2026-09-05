"""NOAA GML 地上フラスコ網の取り込み(L0)。

このモジュールが守る約束は三つある。

1. **緯度がどこから来たかを地点ごとに記録する**(SPEC §2.1)。
   固定地点は event ヘッダの `site_latitude`、船舶帯は標本の実緯度からの再導出。
   どちらでもない地点は `lat_source = "none"` として残し、黙って捨てない。
2. **仮定が崩れたら落ちる検算をコードの中に置く**(HC-075)。
   列名・帯の刻み・分類の網羅性は、そのつど例外で止める。黙って通る道を作らない。
3. **平滑済みの系列を学習経路へ出さない**(SPEC §2.2 / G-06)。
   `*_month.txt` は Thoning 平滑の産物であり、`smoothed=True` の印を負う。
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXT = DATA / "raw" / "ext"

#: NOAA が欠測を表すのに使う値。実データでは -1e+34 が site_latitude に入る。
FILL_THRESHOLD = -1.0e30

#: 帯の緯度を符号名と突き合わせるときの許容(度)。
#: 符号名からの推定を無検算で通さないための検算であり、帯の幅ではない。
BAND_NAME_CHECK_DEG = 0.5

#: 帯の符号。例: pocn30 → 親 poc・北緯 30 度、pocs05 → 親 poc・南緯 5 度、poc000 → 0 度。
BAND_CODE = re.compile(r"^(?P<parent>[a-z]{3})(?P<hemi>[ns]?)(?P<deg>\d{2,3})$")

#: 「# key : value」「# key: value」の双方を読む。key 側に : を含む行(latitude:units 等)も通る。
HEADER_LINE = re.compile(r"^#\s*(?P<key>.+?)\s*:\s+(?P<value>.*?)\s*$")


class IngestError(Exception):
    """外部形式についての仮定が崩れたときに投げる。黙って違う結果を返さないための例外。"""


@dataclass(frozen=True)
class Site:
    code: str
    latitude: float | None
    lat_source: str  # "header" | "band" | "none"
    name: str | None = None
    country: str | None = None
    elevation: float | None = None
    parent: str | None = None  # 船舶帯のときの親系統


@dataclass
class Series:
    code: str
    source: str  # "event" | "month"
    smoothed: bool
    values: list = field(default_factory=list)


# --------------------------------------------------------------------- 走査

def gas_dir(gas: str) -> Path:
    d = EXT / f"{gas}_surface-flask_ccgg_text"
    if not d.is_dir():
        raise IngestError(f"展開済みディレクトリが無い: {d}")
    return d


def discover_month_site_codes(gas: str) -> set[str]:
    """`*_month.txt` のファイル名から地点符号を得る。ファイル名だけを見る経路。"""
    codes = set()
    for p in gas_dir(gas).glob(f"{gas}_*_month.txt"):
        codes.add(p.name.split("_")[1])
    return codes


def event_path(gas: str, code: str) -> Path | None:
    matches = sorted(gas_dir(gas).glob(f"{gas}_{code}_*event.txt"))
    if not matches:
        return None
    if len(matches) > 1:
        raise IngestError(f"{code}: event ファイルが一意でない {[m.name for m in matches]}")
    return matches[0]


def month_path(gas: str, code: str) -> Path:
    matches = sorted(gas_dir(gas).glob(f"{gas}_{code}_*_month.txt"))
    if len(matches) != 1:
        raise IngestError(f"{code}: month ファイルが一意でない {[m.name for m in matches]}")
    return matches[0]


# --------------------------------------------------------------------- ヘッダ

@functools.lru_cache(maxsize=None)
def read_event_header(path: Path) -> dict:
    header = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = HEADER_LINE.match(line.rstrip("\n"))
            if m and m.group("key") != "comment":
                header[m.group("key")] = m.group("value")
    if "site_code" not in header:
        raise IngestError(f"{path.name}: site_code がヘッダに無い")
    return header


def _header_latitude(header: dict) -> float | None:
    raw = header.get("site_latitude")
    if raw is None:
        return None
    value = float(raw)
    if value < FILL_THRESHOLD or not (-90.0 <= value <= 90.0):
        return None  # 欠測値。移動プラットフォームの公称位置は使えない(G-05)
    return value


def event_headers_with_fill_latitude(gas: str) -> set[str]:
    """公称緯度が欠測値になっている系統の符号。T-003 の陽性対照が実在を確かめる。"""
    out = set()
    for p in gas_dir(gas).glob(f"{gas}_*event.txt"):
        header = read_event_header(p)
        raw = header.get("site_latitude")
        if raw is not None and float(raw) < FILL_THRESHOLD:
            out.add(header["site_code"].lower())
    return out


# --------------------------------------------------------------------- 船舶帯

def band_nominal_latitude(code: str) -> float:
    """符号名が主張している緯度。これは**候補**であって、検算を経るまで採用しない。"""
    m = BAND_CODE.match(code)
    if not m:
        raise IngestError(f"{code}: 帯の符号として読めない")
    deg = float(m.group("deg"))
    return -deg if m.group("hemi") == "s" else deg


@functools.lru_cache(maxsize=None)
def observed_marks(gas: str, parent: str) -> tuple[float, ...]:
    """親系統の緯度分布だけから、船が実際に止まった目標点を取り出す。

    **符号名を一切見ない。** ここが循環を切る要である。

    採り方: 保持標本の緯度を値ごとに数え、件数の降順に並べたときの
    **最大の倍率ギャップ**より上を目標点とする。閾値を人が置かないので、
    刻みの違う系統(POC は 5 度・SCS は 3 度)へ同じ定数を持ち込む事故が起きない。

    実測 2026-09-05: POC は目標点 92〜625 件に対し非目標点の最大が 24 件(倍率 3.8)、
    SCS は 153〜188 件に対し 28 件(倍率 5.5)で、どちらもギャップは一箇所で明瞭に開く。
    """
    path = event_path(gas, parent)
    if path is None:
        raise IngestError(f"{parent}: 親の event ファイルが無い")
    counts = {}
    for s in read_event_samples(path):
        if is_retained(s):
            counts[s.latitude] = counts.get(s.latitude, 0) + 1
    if len(counts) < 3:
        raise IngestError(f"{parent}: 緯度の異なり数が少なすぎて目標点を導けない")

    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    best_gap, cut = 0.0, 1
    for i in range(1, len(ordered)):
        ratio = ordered[i - 1][1] / ordered[i][1]
        if ratio > best_gap:
            best_gap, cut = ratio, i
    if best_gap < 2.0:
        raise IngestError(f"{parent}: 目標点と非目標点が分離しない(最大倍率 {best_gap:.2f})")
    return tuple(sorted(lat for lat, _ in ordered[:cut]))


@functools.lru_cache(maxsize=None)
def mark_step(gas: str, parent: str) -> float:
    """目標点の刻み。等差でなければ止める(仮定が崩れたら落ちる検算 — HC-075)。"""
    marks = observed_marks(gas, parent)
    if len(marks) < 2:
        raise IngestError(f"{parent}: 目標点が 2 点未満で刻みを導けない")
    gaps = {round(b - a, 6) for a, b in zip(marks, marks[1:])}
    if len(gaps) != 1:
        raise IngestError(f"{parent}: 目標点の刻みが一定でない {sorted(gaps)}")
    return gaps.pop()


def band_half_width(parent: str, gas: str = "co2") -> float:
    """帯の半値幅。**観測された目標点の刻み**から導く。

    README §7.7 が明示しているのは POC の ±2.5 のみである。刻みの違う SCS へ
    2.5 を持ち込まないため、値は名前ではなくデータから決める。
    """
    return mark_step(gas, parent) / 2.0


def derive_band_latitude(code: str, marks) -> float:
    """符号名が主張する緯度に対応する**観測された目標点**を返す。無ければ止める。

    引数の `marks` はデータのみから導いた目標点であり、符号名から絞り込んでいない。
    したがってこの一致は恒等式ではない —— 名前と観測が独立に決まった二つの量である。
    """
    nominal = band_nominal_latitude(code)
    hits = [m for m in marks if abs(m - nominal) <= BAND_NAME_CHECK_DEG]
    if not hits:
        raise IngestError(
            f"{code}: 符号名の緯度 {nominal:.1f} に対応する目標点が観測されない"
            f"(観測された目標点 {sorted(marks)})"
        )
    if len(hits) > 1:
        raise IngestError(f"{code}: 符号名の緯度 {nominal:.1f} に目標点が複数対応する {hits}")
    return hits[0]


# --------------------------------------------------------------------- 標本

@dataclass(frozen=True)
class Sample:
    code: str
    year: int
    month: int
    time_decimal: float
    value: float
    latitude: float
    qcflag: str


_EVENT_REQUIRED = ("site_code", "year", "month", "time_decimal", "value", "latitude", "qcflag")


@functools.lru_cache(maxsize=None)
def read_event_samples(path: Path) -> tuple[Sample, ...]:
    """event ファイルの全標本(棄却フラグのものも含む)。列名は実物から読む。"""
    columns = None
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split()
            if not fields:
                continue
            if columns is None:
                if fields[0] != "site_code":
                    raise IngestError(f"{path.name}: 列名の行が見つからない(先頭 {fields[0]})")
                columns = fields
                missing = [c for c in _EVENT_REQUIRED if c not in columns]
                if missing:
                    raise IngestError(f"{path.name}: 必要な列が無い {missing}")
                continue
            if len(fields) != len(columns):
                raise IngestError(
                    f"{path.name}: 列数が合わない(期待 {len(columns)} 実 {len(fields)})"
                )
            row = dict(zip(columns, fields))
            rows.append(
                Sample(
                    code=row["site_code"].lower(),
                    year=int(row["year"]),
                    month=int(row["month"]),
                    time_decimal=float(row["time_decimal"]),
                    value=float(row["value"]),
                    latitude=float(row["latitude"]),
                    qcflag=row["qcflag"],
                )
            )
    if columns is None:
        raise IngestError(f"{path.name}: データ行が 1 件も無い")
    return tuple(rows)


def is_retained(sample: Sample) -> bool:
    """QC フラグ第 1 列(棄却)が `.` のものだけを残す(README §7.5)。"""
    return bool(sample.qcflag) and sample.qcflag[0] == "."


@functools.lru_cache(maxsize=None)
def qc_counts(gas: str) -> dict:
    total = rejected = 0
    for p in sorted(gas_dir(gas).glob(f"{gas}_*event.txt")):
        for s in read_event_samples(p):
            total += 1
            if not is_retained(s):
                rejected += 1
    return {"total": total, "rejected": rejected, "retained": total - rejected}


# --------------------------------------------------------------------- 地点

@functools.lru_cache(maxsize=None)
def build_sites(gas: str) -> tuple[Site, ...]:
    """月次ファイルが存在する全地点を、緯度の出所つきで返す。

    **どの地点も捨てない。** 緯度が付かない地点は `lat_source="none"` で残す。
    三分類が月次ファイル数を尽くすことは build_census() が検算する。
    """
    sites = []
    for code in sorted(discover_month_site_codes(gas)):
        ev = event_path(gas, code)
        if ev is not None:
            header = read_event_header(ev)
            lat = _header_latitude(header)
            if lat is not None:
                sites.append(
                    Site(
                        code=code,
                        latitude=lat,
                        lat_source="header",
                        name=header.get("site_name"),
                        country=header.get("site_country"),
                        elevation=_maybe_float(header.get("site_elevation")),
                    )
                )
                continue
            sites.append(Site(code=code, latitude=None, lat_source="none"))
            continue

        m = BAND_CODE.match(code)
        parent = m.group("parent") if m else None
        if parent is None or event_path(gas, parent) is None:
            sites.append(Site(code=code, latitude=None, lat_source="none"))
            continue

        latitude = derive_band_latitude(code, observed_marks(gas, parent))
        header = read_event_header(event_path(gas, parent))
        sites.append(
            Site(
                code=code,
                latitude=latitude,
                lat_source="band",
                name=header.get("site_name"),
                country=header.get("site_country"),
                parent=parent,
            )
        )
    return tuple(sites)


def _maybe_float(raw):
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return None if value < FILL_THRESHOLD else value


# --------------------------------------------------------------------- 系列

def assert_trainable(series) -> None:
    """学習経路へ渡せるのは event 由来の未平滑系列だけである(G-06)。"""
    for s in series:
        if s.source != "event" or s.smoothed:
            raise IngestError(
                f"{s.code}: 平滑済み/非 event の系列を学習経路へ渡そうとした "
                f"(source={s.source} smoothed={s.smoothed})"
            )


@functools.lru_cache(maxsize=None)
def training_series(gas: str) -> tuple[Series, ...]:
    """地点ごとの生標本系列。年周曲線への整形は L1 で行う。"""
    out = []
    for site in build_sites(gas):
        ev = event_path(gas, site.code)
        if ev is None:
            if site.lat_source != "band":
                continue
            # 帯への標本の割当は、観測から決まった目標点 site.latitude を中心に行う。
            # 符号名は割当にも使わない。
            half = band_half_width(site.parent, gas)
            samples = [
                s
                for s in read_event_samples(event_path(gas, site.parent))
                if is_retained(s) and abs(s.latitude - site.latitude) <= half
            ]
        else:
            samples = [s for s in read_event_samples(ev) if is_retained(s)]
        if not samples:
            continue
        out.append(
            Series(
                code=site.code,
                source="event",
                smoothed=False,
                values=[(s.time_decimal, s.value) for s in samples],
            )
        )
    assert_trainable(out)
    return tuple(out)


# --------------------------------------------------------------------- census

def count_by_filename_path() -> dict:
    """ファイル名だけを見る独立経路(T-008 の二経路一致の片側)。"""
    return {
        "co2_month_files": len(list(gas_dir("co2").glob("co2_*_month.txt"))),
        "ch4_month_files": len(list(gas_dir("ch4").glob("ch4_*_month.txt"))),
    }


def build_census() -> dict:
    sites = build_sites("co2")
    by_source = {"header": 0, "band": 0, "none": 0}
    for s in sites:
        by_source[s.lat_source] += 1

    month_files = len(list(gas_dir("co2").glob("co2_*_month.txt")))
    if len(sites) != month_files:
        raise IngestError(f"地点数 {len(sites)} が月次ファイル数 {month_files} と一致しない")
    if sum(by_source.values()) != month_files:
        raise IngestError("緯度の出所による三分類が月次ファイル数を尽くしていない")

    census = {
        "generated_from": "pipeline/ingest.py",
        "co2_month_files": month_files,
        "ch4_month_files": len(list(gas_dir("ch4").glob("ch4_*_month.txt"))),
        "fixed_sites": by_source["header"],
        "ship_band_sites": by_source["band"],
        "sites_without_latitude": by_source["none"],
        "co2_event_samples_total": qc_counts("co2")["total"],
        "co2_event_samples_rejected": qc_counts("co2")["rejected"],
        "co2_event_samples_retained": qc_counts("co2")["retained"],
        "sites_with_both_gases": len(
            discover_month_site_codes("co2") & discover_month_site_codes("ch4")
        ),
        "latitude_range": [
            min(s.latitude for s in sites if s.latitude is not None),
            max(s.latitude for s in sites if s.latitude is not None),
        ],
    }
    return census


def main() -> int:
    census = build_census()
    (DATA / "census.json").write_text(
        json.dumps(census, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for key, value in census.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
