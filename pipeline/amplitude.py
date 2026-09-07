"""年周振幅の経年変化(G-12 / P-06)。DL は使わない純統計。

「高緯度の呼吸は深くなっている」は文献の主張である。ここでは自前で測り直し、
置換検定の対照を置く。単位(ppm)は地点ごとに桁が違うので、
**判定は傾きの絶対値でなく置換帰無分布に対する位置で行う**。
"""
from __future__ import annotations

import random
import zlib

from pipeline import curves

#: この年数以上の採用年を持つ地点だけを対象にする。短い記録では傾きが雑音に埋もれる。
MIN_YEARS_FOR_TREND = 20


def ols_slope(points) -> float:
    """(x, y) 列に対する最小二乗直線の傾き。"""
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx == 0:
        raise ValueError("x がすべて同じ値で傾きを定義できない")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return sxy / sxx


def trend_test(points, permutations: int = 5000, seed: int = 20260906) -> dict:
    """年と振幅の対応をシャッフルして帰無分布を作る。

    帰無仮説は「振幅は年に依存しない」。年のラベルを入れ替えるので、
    振幅の分布そのもの(単位・散らばり)は保存される —— だから単位に依存しない。
    """
    if len(points) < 3:
        raise ValueError("点が少なすぎて傾きを検定できない")

    observed = ols_slope(points)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    rng = random.Random(seed)
    null_slopes = []
    for _ in range(permutations):
        shuffled = ys[:]
        rng.shuffle(shuffled)
        null_slopes.append(ols_slope(list(zip(xs, shuffled))))

    extreme = sum(1 for s in null_slopes if abs(s) >= abs(observed))
    return {
        "n": len(points),
        "slope": observed,
        "p_value": (extreme + 1) / (permutations + 1),
        "null_mean_slope": sum(null_slopes) / len(null_slopes),
        "permutations": permutations,
    }


def slope_with_covariate(points, covariate) -> dict:
    """`振幅 ~ 年 + 標本数` を解き、年の係数を返す。

    採取密度は年周振幅を下向きに歪める(標本が少ないほど山谷が雑音で開く)。
    実測 2026-09-06 では 15 地点中 12 地点で r(標本数, 振幅) < 0 であり、
    しかも ALT/ZEP/BRW では標本数が年とともに**減って**いる —— つまり交絡は
    「振幅が増えた」という主張と**同じ向き**に効く。だから共変量を入れて測り直す。
    """
    n = len(points)
    xs = [[1.0, float(x), float(c)] for (x, _), c in zip(points, covariate)]
    ys = [y for _, y in points]

    matrix = [[sum(r[i] * r[j] for r in xs) for j in range(3)] for i in range(3)]
    rhs = [sum(r[i] * y for r, y in zip(xs, ys)) for i in range(3)]

    # 3x3 のガウス消去
    a = [matrix[i][:] + [rhs[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("特異な設計行列")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(3):
            if r == col:
                continue
            f = a[r][col] / a[col][col]
            for c in range(col, 4):
                a[r][c] -= f * a[col][c]
    coef = [a[i][3] / a[i][i] for i in range(3)]
    return {"n": n, "slope_year": coef[1], "slope_samples": coef[2]}


def site_amplitude_series(gas: str = "co2") -> dict:
    """地点ごとの (年, 年周振幅) 列。振幅は実測の山と谷の差で取る。

    第一調和の大きさではなく実測の差にするのは、調和で測ると
    「調和に近い形かどうか」が振幅に混ざるためである。
    """
    out: dict[str, list] = {}
    for ex in curves.build_examples(gas):
        out.setdefault(ex.code, []).append((ex.year, curves.peak_to_trough(ex.values)))
    return {code: sorted(rows) for code, rows in out.items()}


def site_sample_counts(gas: str = "co2") -> dict:
    """地点ごとの (年, 標本数)。`site_amplitude_series` と同じ順に並ぶ。"""
    out: dict[str, list] = {}
    for ex in curves.build_examples(gas):
        out.setdefault(ex.code, []).append((ex.year, ex.n_samples))
    return {code: [n for _, n in sorted(rows)] for code, rows in out.items()}


def evaluate(gas: str = "co2", permutations: int = 5000) -> dict:
    """G-12 の判定材料。地点ごとに検定し、緯度帯でまとめる。"""
    import statistics

    from pipeline import ingest

    sites = {s.code: s for s in ingest.build_sites(gas)}
    series = site_amplitude_series(gas)
    counts = site_sample_counts(gas)

    rows = []
    for code, points in sorted(series.items()):
        if len(points) < MIN_YEARS_FOR_TREND:
            continue
        # Python の文字列ハッシュはプロセスごとに変わる(PYTHONHASHSEED)。
        # 種に使うと結果が再現しないので、決定的なハッシュを使う。
        seed = zlib.crc32(code.encode("utf-8"))
        result = trend_test(points, permutations=permutations, seed=seed)
        adjusted = slope_with_covariate(points, counts[code])
        rows.append(
            {
                "code": code,
                "latitude": sites[code].latitude,
                "years": len(points),
                "mean_amplitude": statistics.mean(y for _, y in points),
                "slope_ppm_per_year": result["slope"],
                "p_value": result["p_value"],
                "slope_adjusted_for_samples": adjusted["slope_year"],
                "slope_on_samples": adjusted["slope_samples"],
            }
        )

    def band(row):
        lat = row["latitude"]
        if lat >= 45:
            return "北緯45度以北"
        if lat >= 0:
            return "北半球 0-45度"
        return "南半球"

    bands: dict[str, dict] = {}
    for row in rows:
        b = bands.setdefault(
            band(row),
            {"sites": 0, "positive": 0, "significant_positive": 0, "positive_adjusted": 0},
        )
        b["sites"] += 1
        if row["slope_ppm_per_year"] > 0:
            b["positive"] += 1
            if row["p_value"] < 0.05:
                b["significant_positive"] += 1
        if row["slope_adjusted_for_samples"] > 0:
            b["positive_adjusted"] += 1

    return {
        "min_years": MIN_YEARS_FOR_TREND,
        "permutations": permutations,
        "sites_tested": len(rows),
        "bands": bands,
        "sites": rows,
    }
