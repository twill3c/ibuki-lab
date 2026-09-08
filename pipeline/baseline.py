"""ベースライン測定(SPEC §7)。すべて地点群ばらし leave-one-group-out で測る。

ここで決まるのは二つ。

- **模型が上回るべき相手**(G-09)。2 特徴系のうち**強いほう**を相手にする。
  弱いほうを選んで勝つのは対照ではない(HC-064)
- **P-01 の閾値**。中央値予測器の MAE の半分。模型の成績を見る前に決め方を定めてある
"""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from pipeline import curves

REPORT = Path(__file__).resolve().parents[1] / "data" / "baseline.json"

#: kNN の k は学習側だけで選ぶ。この候補から内側の群ばらしで決める。
K_CANDIDATES = (1, 3, 5, 7, 9, 15)

#: 陰性対照の反復回数(地点への緯度の割り当てを入れ替える)。
SHUFFLE_REPEATS = 20
SHUFFLE_SEED = 20260906


# ------------------------------------------------------------------ 予測器

def _solve(matrix, rhs):
    """小さな正方系をガウス消去で解く。"""
    n = len(matrix)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
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


def _fit_linear(features, targets):
    """切片つき線形回帰。"""
    dim = len(features[0])
    design = [[1.0] + list(f) for f in features]
    matrix = [[sum(row[i] * row[j] for row in design) for j in range(dim + 1)] for i in range(dim + 1)]
    rhs = [sum(row[i] * t for row, t in zip(design, targets)) for i in range(dim + 1)]
    coef = _solve(matrix, rhs)
    return lambda f: coef[0] + sum(c * x for c, x in zip(coef[1:], f))


def _neighbour_order(features) -> list:
    """各例について、近い順に並べた相手の添字。距離は一度だけ計算する。

    群を外して k 近傍を取る操作は、この並びを前から歩いて許された相手を拾うだけになる。
    外す群は高々二つなので、歩きは先頭付近で終わる。
    """
    n = len(features)
    order = []
    for i in range(n):
        fi = features[i]
        distances = [
            (sum((a - b) ** 2 for a, b in zip(fi, features[j])), j) for j in range(n) if j != i
        ]
        distances.sort()
        order.append([j for _, j in distances])
    return order


def _knn_prefix_means(order_i, targets, allowed, max_k):
    """許された相手を近い順に max_k 件まで拾い、k ごとの平均を返す。"""
    picked = []
    for j in order_i:
        if allowed(j):
            picked.append(targets[j])
            if len(picked) == max_k:
                break
    if not picked:
        raise ValueError("近傍が一つも残らない")
    means = {}
    running = 0.0
    for idx, value in enumerate(picked, start=1):
        running += value
        means[idx] = running / idx
    return {k: means[min(k, len(picked))] for k in K_CANDIDATES}


def _choose_k(order, targets, groups, held_out, candidates):
    """k は学習側だけで決める。内側も群ばらしにする。

    内側の一件抜きは「外側で抜いた群」と「その例自身の群」を同時に外すことで作る。
    """
    max_k = max(K_CANDIDATES)
    totals = {k: 0.0 for k in K_CANDIDATES}
    count = 0
    for i in candidates:
        own = groups[i]

        def allowed(j, own=own):
            return groups[j] != held_out and groups[j] != own

        try:
            means = _knn_prefix_means(order[i], targets, allowed, max_k)
        except ValueError:
            continue
        for k, value in means.items():
            totals[k] += abs(value - targets[i])
        count += 1
    if count == 0:
        return K_CANDIDATES[len(K_CANDIDATES) // 2]
    return min(K_CANDIDATES, key=lambda k: totals[k] / count)


# ------------------------------------------------------------------ 評価

def _features_for(system: str, ex):
    if system in ("harmonic_linear", "harmonic_knn"):
        return curves.first_harmonic_least_squares(ex.values)
    if system == "curve_knn":
        return tuple(ex.values)
    return ()


def _predictions(system: str, examples, labels, order=None) -> list:
    """leave-one-group-out の予測を、例の順に並べて返す。

    `order` は近傍の並び(距離は入替対照でも変わらないので使い回す)。
    """
    groups = [ex.group for ex in examples]
    features = [_features_for(system, ex) for ex in examples]
    preds = [None] * len(examples)
    is_knn = system in ("harmonic_knn", "curve_knn")
    if is_knn and order is None:
        order = _neighbour_order(features)
    chosen_k = []

    for train_idx, test_idx in curves.leave_one_group_out(examples):
        train_targets = [labels[i] for i in train_idx]
        held = groups[test_idx[0]]

        if system == "mean_predictor":
            value = statistics.mean(train_targets)
            for i in test_idx:
                preds[i] = value
        elif system == "median_predictor":
            value = statistics.median(train_targets)
            for i in test_idx:
                preds[i] = value
        elif system == "harmonic_linear":
            model = _fit_linear([features[i] for i in train_idx], train_targets)
            for i in test_idx:
                preds[i] = model(features[i])
        else:
            k = _choose_k(order, labels, groups, held, train_idx)
            chosen_k.append(k)
            for i in test_idx:
                preds[i] = _knn_prefix_means(
                    order[i], labels, lambda j: groups[j] != held, max(K_CANDIDATES)
                )[k]

    assert all(p is not None for p in preds), "予測されていない例がある"
    return preds


def _score(examples, labels, preds) -> dict:
    """例あたりと地点あたりの両方で測る。

    地点ごとの年数は 1〜40 年と桁が違うので、例あたりだけを見ると
    よく採れた地点が結論を握る。**主指標は地点あたり**とする。
    """
    per_example = [abs(p - t) for p, t in zip(preds, labels)]

    by_site: dict[str, list] = {}
    for ex, p, t in zip(examples, preds, labels):
        by_site.setdefault(ex.code, []).append(abs(p - t))
    per_site = [statistics.mean(v) for v in by_site.values()]

    return {
        "mae": statistics.mean(per_site),
        "mae_per_example": statistics.mean(per_example),
        "median_ae_per_site": statistics.median(per_site),
        "worst_site_ae": max(per_site),
        "sites": len(per_site),
        "examples": len(per_example),
    }


SYSTEMS = (
    "mean_predictor",
    "median_predictor",
    "harmonic_linear",
    "harmonic_knn",
    "curve_knn",
)


def run(gas: str = "co2", examples=None) -> dict:
    """`examples` を渡すとその母集団で測る。

    G-10 の対照母集団(CO2 と CH4 の両方が揃う 739 地点年)に対しても、
    模型と**同じ土俵**でベースラインを出せるようにするための引数である。
    渡さなければ主母集団(805 地点年)。
    """
    if examples is None:
        examples = curves.build_examples(gas)
    labels = [ex.latitude for ex in examples]

    systems = {}
    for system in SYSTEMS:
        preds = _predictions(system, examples, labels)
        systems[system] = _score(examples, labels, preds)

    best_name = min(systems, key=lambda s: systems[s]["mae"])
    best_two_feature = min(
        ("harmonic_linear", "harmonic_knn"), key=lambda s: systems[s]["mae"]
    )

    # 陰性対照: 地点への緯度の割り当てを入れ替える。地点年ではなく**地点**単位で入れ替える。
    site_lat = {ex.code: ex.latitude for ex in examples}
    codes = sorted(site_lat)
    rng = random.Random(SHUFFLE_SEED)
    # 距離はラベルに依らないので、近傍の並びは一度作って使い回す。
    shuffle_order = (
        _neighbour_order([_features_for(best_two_feature, ex) for ex in examples])
        if best_two_feature in ("harmonic_knn", "curve_knn")
        else None
    )
    shuffle_maes = []
    for _ in range(SHUFFLE_REPEATS):
        pool = [site_lat[c] for c in codes]
        rng.shuffle(pool)
        mapping = dict(zip(codes, pool))
        shuffled_labels = [mapping[ex.code] for ex in examples]
        preds = _predictions(best_two_feature, examples, shuffled_labels, order=shuffle_order)
        shuffle_maes.append(_score(examples, shuffled_labels, preds)["mae"])

    threshold = systems["median_predictor"]["mae"] / 2.0

    return {
        "generated_from": "pipeline/baseline.py",
        "gas": gas,
        "n_examples": len(examples),
        "n_sites": len({ex.code for ex in examples}),
        "n_groups": len({ex.group for ex in examples}),
        "coverage": curves.coverage_report(gas),
        "systems": systems,
        "best_system": best_name,
        "best_two_feature_system": best_two_feature,
        "controls": {
            "label_shuffle_mae": statistics.mean(shuffle_maes),
            "label_shuffle_min": min(shuffle_maes),
            "label_shuffle_repeats": SHUFFLE_REPEATS,
            "label_shuffle_system": best_two_feature,
        },
        "p01_threshold_mae": threshold,
    }


def load_or_run(gas: str = "co2") -> dict:
    if REPORT.exists():
        return json.loads(REPORT.read_text(encoding="utf-8"))
    report = run(gas)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = run()
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"例 {report['n_examples']} / 地点 {report['n_sites']} / 群 {report['n_groups']}")
    print(f"{'系':>18} {'MAE(地点)':>10} {'MAE(例)':>10} {'最悪地点':>10}")
    for name, s in report["systems"].items():
        print(f"{name:>18} {s['mae']:>10.2f} {s['mae_per_example']:>10.2f} {s['worst_site_ae']:>10.2f}")
    c = report["controls"]
    print(f"\n陰性対照(ラベル入替 {c['label_shuffle_repeats']} 回・{c['label_shuffle_system']}): "
          f"MAE {c['label_shuffle_mae']:.2f}(最小 {c['label_shuffle_min']:.2f})")
    print(f"P-01 の閾値(中央値予測器の半分): {report['p01_threshold_mae']:.2f} 度")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
