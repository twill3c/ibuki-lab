"""ベースラインと対照に対する検査(TEST_SPEC T-017 / T-018)。"""
import math
import random

from pipeline import amplitude, baseline


# ---------------------------------------------------------------- T-017 (G-03)

def test_t017_label_shuffle_does_not_beat_the_trivial_predictor():
    """陰性対照: 地点単位で緯度を入れ替えると、MAE が中央値予測器を下回らない。

    片側の主張である。入替えたほうが良くなったら、緯度以外の何かを測っている。
    """
    report = baseline.load_or_run()
    shuffled = report["controls"]["label_shuffle_mae"]
    trivial = report["systems"]["median_predictor"]["mae"]

    assert shuffled >= trivial * 0.95, (
        f"ラベル入替が中央値予測器を上回った(入替 {shuffled:.2f} / 自明 {trivial:.2f})"
    )


def test_t017_positive_control_real_labels_beat_the_shuffle():
    """陽性対照: 素のラベルでは入替対照より良くなること。

    これが成り立たなければ、そもそも信号が無いか、評価系が壊れている。
    """
    report = baseline.load_or_run()
    best = min(s["mae"] for s in report["systems"].values())
    shuffled = report["controls"]["label_shuffle_mae"]

    assert best < shuffled, f"素のラベルが入替対照に勝てない(素 {best:.2f} / 入替 {shuffled:.2f})"


# ---------------------------------------------------------------- T-018 (G-12)

def _synthetic_series(slope_per_year: float, seed: int = 20260906):
    rng = random.Random(seed)
    return [
        (year, 10.0 + slope_per_year * (year - 1980) + rng.gauss(0.0, 0.4))
        for year in range(1980, 2026)
    ]


def test_t018_amplitude_trend_detector_fires_on_an_injected_trend():
    """陽性対照: 人工的に入れた増加傾向を検出器が捕まえる。"""
    injected = amplitude.trend_test(_synthetic_series(0.06), permutations=2000, seed=1)
    assert injected["slope"] > 0
    assert injected["p_value"] < 0.01, f"入れたトレンドを捕まえられない(p={injected['p_value']})"


def test_t018_negative_control_flat_series_is_not_flagged():
    """陰性対照: 傾きの無い系列では撃たない(HC-074 の誤検出 0 の確認)。"""
    flat = amplitude.trend_test(_synthetic_series(0.0), permutations=2000, seed=2)
    assert flat["p_value"] > 0.05, f"平坦な系列で撃った(p={flat['p_value']})"


def test_t018_permutation_null_is_centred_on_zero():
    """帰無分布が 0 の周りに来ること。来なければ置換の実装が壊れている。"""
    result = amplitude.trend_test(_synthetic_series(0.06), permutations=2000, seed=3)
    assert abs(result["null_mean_slope"]) < abs(result["slope"]) * 0.1, (
        f"帰無平均が 0 から離れている: {result['null_mean_slope']}"
    )


def test_t018_detector_is_scale_free_in_the_right_way():
    """同じ形の系列を一定倍しても p 値の判定が変わらないこと。

    振幅の単位(ppm)は地点ごとに大きく違うので、単位に引きずられる検出器は使えない。
    """
    base = _synthetic_series(0.06, seed=7)
    scaled = [(y, v * 3.0) for y, v in base]

    a = amplitude.trend_test(base, permutations=2000, seed=11)
    b = amplitude.trend_test(scaled, permutations=2000, seed=11)
    assert (a["p_value"] < 0.01) == (b["p_value"] < 0.01), "倍率で判定が変わった"
    assert math.isclose(b["slope"], a["slope"] * 3.0, rel_tol=1e-9), (
        "傾きが倍率どおりに動いていない —— 検出器が単位に依存している"
    )
