"""年周曲線の構成と、地点群ばらしの分割(L1)。

SPEC §2.3 の五手順をそのまま実装する。要点は二つ。

- **補間しない。** 欠けたビンのある地点年は落とす。埋めれば平滑になり、
  模型は大気でなく補間器を学ぶ(SPEC §2.2 と同じ理由)。
- **直線を引いて差し引く。** 絶対濃度は半球間の勾配そのもので、
  緯度の手がかりが最も強く漏れる経路である(G-02)。
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from pipeline import ingest

#: 年内を何等分するか。12/24/52 の被覆を実測して 24 を選んだ(SPEC §2.3)。
N_BINS = 24

#: 地点年として採る最小の保持標本数。
MIN_SAMPLES_PER_YEAR = 40

#: この距離より近い地点は同じ群に入れる(G-04)。
GROUP_RADIUS_KM = 500.0

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class RawYear:
    """ビン平均まで済ませた地点年。まだ直線を引いていない。"""

    code: str
    year: int
    bins: tuple  # 長さ N_BINS。埋まらなかったビンは None
    n_samples: int

    @property
    def filled(self) -> int:
        return sum(1 for b in self.bins if b is not None)


@dataclass(frozen=True)
class Example:
    """学習・評価に渡る一例。"""

    code: str
    year: int
    group: str
    latitude: float
    values: tuple  # 長さ N_BINS の ppm 偏差
    n_samples: int
    filled_bins: int
    interpolated: int

    @property
    def key(self) -> str:
        return f"{self.code}:{self.year}"


# ------------------------------------------------------------------ 曲線の構成

def bin_index(time_decimal: float) -> int:
    """年内の小数位置を N_BINS 等分したときのビン番号。"""
    frac = time_decimal - math.floor(time_decimal)
    return min(N_BINS - 1, int(frac * N_BINS))


@functools.lru_cache(maxsize=None)
def raw_bin_means(gas: str) -> tuple:
    """地点年ごとのビン平均。同一時刻の標本を先に平均してからビンに入れる。

    順序は NOAA 自身の月次平均と同じ(README §7.7)。実測 2026-09-06 で標本の
    49.0% が同一時刻の重複であり、先に潰さないと採取密度の偏りがビン平均に効く。
    """
    out = []
    for series in ingest.training_series(gas):
        by_year: dict[int, dict[float, list]] = {}
        for time_decimal, value in series.values:
            year = int(math.floor(time_decimal))
            by_year.setdefault(year, {}).setdefault(time_decimal, []).append(value)

        for year, by_time in sorted(by_year.items()):
            n_samples = sum(len(v) for v in by_time.values())
            if n_samples < MIN_SAMPLES_PER_YEAR:
                continue
            buckets: list[list] = [[] for _ in range(N_BINS)]
            for time_decimal, values in by_time.items():
                buckets[bin_index(time_decimal)].append(sum(values) / len(values))
            bins = tuple(
                (sum(b) / len(b)) if b else None for b in buckets
            )
            out.append(RawYear(code=series.code, year=year, bins=bins, n_samples=n_samples))
    return tuple(out)


def accept_bins(bins) -> bool:
    """全ビンが実測で埋まっているか。補間はしないので、これが採否そのものである。"""
    return len(bins) == N_BINS and all(b is not None for b in bins)


def least_squares_slope(values) -> float:
    """等間隔の系列に対する最小二乗直線の傾き(単位はビンあたり)。"""
    n = len(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    sxx = sum((i - mean_x) ** 2 for i in range(n))
    sxy = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    return sxy / sxx


def detrend(values) -> tuple:
    """最小二乗直線を差し引く。水準(絶対濃度)と年内の増加が同時に消える。"""
    n = len(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    slope = least_squares_slope(values)
    return tuple(v - (mean_y + slope * (i - mean_x)) for i, v in enumerate(values))


# ------------------------------------------------------------------ 地点と群

@functools.lru_cache(maxsize=None)
def site_positions(gas: str) -> dict:
    """固定地点の緯度経度。船舶帯は経度が定まらないので群分けの対象外。"""
    out = {}
    for site in ingest.build_sites(gas):
        if site.lat_source != "header":
            continue
        header = ingest.read_event_header(ingest.event_path(gas, site.code))
        out[site.code] = (float(header["site_latitude"]), float(header["site_longitude"]))
    return out


def great_circle_km(a, b) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(h)))


@functools.lru_cache(maxsize=None)
def site_groups(gas: str) -> dict:
    """近接地点を併合した群。同じ空気を測っている地点を train/test に跨らせない。"""
    positions = site_positions(gas)
    codes = sorted(positions)
    parent = {c: c for c in codes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(codes):
        for b in codes[i + 1 :]:
            if great_circle_km(positions[a], positions[b]) < GROUP_RADIUS_KM:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return {c: find(c) for c in codes}


# ------------------------------------------------------------------ 例の組み立て

@functools.lru_cache(maxsize=None)
def build_examples(gas: str) -> tuple:
    """学習・評価に渡る例。固定地点のみ(船舶帯は §2.3 の理由で外部ホールドアウトへ)。"""
    sites = {s.code: s for s in ingest.build_sites(gas)}
    groups = site_groups(gas)

    out = []
    for raw in raw_bin_means(gas):
        if not accept_bins(raw.bins):
            continue
        if raw.code not in groups:
            continue  # 船舶帯
        out.append(
            Example(
                code=raw.code,
                year=raw.year,
                group=groups[raw.code],
                latitude=sites[raw.code].latitude,
                values=detrend(raw.bins),
                n_samples=raw.n_samples,
                filled_bins=raw.filled,
                interpolated=0,
            )
        )
    return tuple(out)


@functools.lru_cache(maxsize=None)
def band_climatology(gas: str) -> tuple:
    """船舶帯の気候値曲線。**複数年をまとめて一本にする。**

    周航は各緯度を年に数回しか通らないので、一年では 24 ビンが埋まらない
    (SPEC §2.3)。学習には使えないが、**緯度梯子の絵**(F-02)には使える ——
    同じ船・同じ海・同じ研究室で緯度だけを振った列だからである。

    地点年ごとの直線除去ができないので、ここでは**ビンごとに全年の偏差を平均する**。
    偏差は「その年のその帯の平均」からの差で取る。つまり年ごとの水準と長期の増加は
    年内平均を引く時点で落ちるが、**年内の増加ぶんは残る** —— 学習の系列とは
    作り方が違うので、混ぜて使わない。
    """
    sites = {s.code: s for s in ingest.build_sites(gas)}
    out = []
    for series in ingest.training_series(gas):
        site = sites[series.code]
        if site.lat_source != "band":
            continue

        by_year: dict[int, dict[float, list]] = {}
        for time_decimal, value in series.values:
            year = int(math.floor(time_decimal))
            by_year.setdefault(year, {}).setdefault(time_decimal, []).append(value)

        sums = [0.0] * N_BINS
        counts = [0] * N_BINS
        years_used = 0
        for year, by_time in sorted(by_year.items()):
            averaged = [
                (t, sum(v) / len(v)) for t, v in sorted(by_time.items())
            ]
            if len(averaged) < 4:
                continue
            level = sum(v for _, v in averaged) / len(averaged)
            for time_decimal, value in averaged:
                index = bin_index(time_decimal)
                sums[index] += value - level
                counts[index] += 1
            years_used += 1

        if not all(counts):
            continue
        out.append(
            {
                "code": series.code,
                "parent": site.parent,
                "latitude": site.latitude,
                "values": tuple(s / c for s, c in zip(sums, counts)),
                "years": years_used,
                "samples_per_bin_min": min(counts),
                "samples_total": sum(counts),
            }
        )
    return tuple(sorted(out, key=lambda r: -r["latitude"]))


def site_climatology(gas: str) -> tuple:
    """固定地点の気候値曲線。採用した地点年の平均を取るだけ。

    学習に使ったのと**同じ作り方の系列**を平均しているので、画面①の帯と
    模型が見ているものは同じ形である。
    """
    by_site: dict[str, list] = {}
    for ex in build_examples(gas):
        by_site.setdefault(ex.code, []).append(ex)

    out = []
    for code, rows in by_site.items():
        stacked = [
            sum(r.values[i] for r in rows) / len(rows) for i in range(N_BINS)
        ]
        out.append(
            {
                "code": code,
                "latitude": rows[0].latitude,
                "values": tuple(stacked),
                "years": len(rows),
                "year_min": min(r.year for r in rows),
                "year_max": max(r.year for r in rows),
            }
        )
    return tuple(sorted(out, key=lambda r: -r["latitude"]))


def coverage_report(gas: str) -> dict:
    """採否の内訳。落とした数を分子と分母で出す(G-16)。"""
    raws = raw_bin_means(gas)
    groups = site_groups(gas)
    fixed = [r for r in raws if r.code in groups]
    accepted = [r for r in fixed if accept_bins(r.bins)]
    return {
        "site_years_with_min_samples": len(raws),
        "site_years_fixed_sites": len(fixed),
        "site_years_accepted": len(accepted),
        "site_years_dropped_incomplete": len(fixed) - len(accepted),
        "site_years_ship_bands_excluded": len(raws) - len(fixed),
        "sites_accepted": len({r.code for r in accepted}),
        "groups_accepted": len({groups[r.code] for r in accepted}),
    }


# ------------------------------------------------------------------ 分割

def leave_one_group_out(examples) -> list:
    """群ごとに一つ抜く。同じ群の地点年は必ず同じ側に来る。"""
    order = sorted({ex.group for ex in examples})
    folds = []
    for held in order:
        test = [i for i, ex in enumerate(examples) if ex.group == held]
        train = [i for i, ex in enumerate(examples) if ex.group != held]
        if test and train:
            folds.append((train, test))
    return folds


def split_is_clean(examples, folds) -> bool:
    for train_idx, test_idx in folds:
        train_groups = {examples[i].group for i in train_idx}
        test_groups = {examples[i].group for i in test_idx}
        if train_groups & test_groups:
            return False
    return True


# ------------------------------------------------------------------ 第一調和

def first_harmonic_least_squares(values) -> tuple:
    """`v ≈ a1·cos(2πt) + b1·sin(2πt)` を最小二乗で解く。

    **正規方程式を 2×2 のまま解く。** 直交性を仮定して閉じた式を書くと、
    もう一方の経路(`first_harmonic_dft`)と同じ式になり、照合が恒等式になる
    (loop_000 で船舶帯の検算に対して踏んだのと同じ罠 — HC-045)。
    交差項が実際に消えることは、ここで解く過程が示す。
    """
    n = len(values)
    cos_terms = [math.cos(2 * math.pi * (i + 0.5) / n) for i in range(n)]
    sin_terms = [math.sin(2 * math.pi * (i + 0.5) / n) for i in range(n)]

    scc = sum(c * c for c in cos_terms)
    sss = sum(s * s for s in sin_terms)
    scs = sum(c * s for c, s in zip(cos_terms, sin_terms))
    rc = sum(v * c for v, c in zip(values, cos_terms))
    rs = sum(v * s for v, s in zip(values, sin_terms))

    det = scc * sss - scs * scs
    if abs(det) < 1e-12:
        raise ValueError("設計行列が特異 —— ビン数が第一調和を分解できていない")
    return (sss * rc - scs * rs) / det, (scc * rs - scs * rc) / det


def first_harmonic_dft(values) -> tuple:
    """同じ係数を離散フーリエ和から出す独立経路(T-019 の照合相手)。"""
    n = len(values)
    real = sum(v * math.cos(2 * math.pi * (i + 0.5) / n) for i, v in enumerate(values))
    imag = sum(v * math.sin(2 * math.pi * (i + 0.5) / n) for i, v in enumerate(values))
    return 2.0 * real / n, 2.0 * imag / n


def amplitude_of(values) -> float:
    """年周振幅(第一調和の大きさの 2 倍 = 山から谷まで)。"""
    a1, b1 = first_harmonic_least_squares(values)
    return 2.0 * math.hypot(a1, b1)


def peak_to_trough(values) -> float:
    """実測の山と谷の差。第一調和に頼らない振幅の測り方。"""
    return max(values) - min(values)
