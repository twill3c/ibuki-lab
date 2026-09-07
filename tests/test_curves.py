"""年周曲線の構成と分割に対する検査(TEST_SPEC T-012..T-016 / T-019)。"""
import math

import pytest

from pipeline import curves


def _first_complete_year():
    """全ビンが埋まった実データの地点年を一つ取る。

    `raw_bin_means` の先頭は採否を通っていない —— 実測では 24 ビン中 13 しか
    埋まらない地点年が先頭に来る。対照の前提はフィクスチャから導く(HC-070)。
    """
    for raw in curves.raw_bin_means("co2"):
        if curves.accept_bins(raw.bins):
            return raw
    raise AssertionError("全ビンが埋まった地点年が一つも無い")


# ---------------------------------------------------------------- T-012 (G-02)

def test_t012_curve_has_no_level_and_no_slope():
    """直線除去の帰結として、平均も傾きも 0 になる(SPEC §2.3 手順 4)。

    絶対濃度は半球間の勾配そのもので、緯度の手がかりが最も強く漏れる経路である。
    """
    examples = curves.build_examples("co2")
    assert examples, "地点年が 0 件(走査対象が空)"

    for ex in examples:
        assert abs(sum(ex.values) / len(ex.values)) < 1e-9, f"{ex.key}: 平均が 0 でない"
        assert abs(curves.least_squares_slope(ex.values)) < 1e-9, f"{ex.key}: 傾きが 0 でない"


def test_t012_positive_control_detrend_check_fires():
    """陽性対照: 直線を除去していない曲線では、平均も傾きも 0 にならない(HC-041)。"""
    raw = [400.0 + 0.2 * i + 3.0 * math.sin(2 * math.pi * i / 24) for i in range(24)]

    assert abs(sum(raw) / len(raw)) > 1.0, "この対照の入力に下駄が乗っていない"
    assert abs(curves.least_squares_slope(raw)) > 0.01, "この対照の入力に傾きが無い"

    flat = curves.detrend(raw)
    assert abs(sum(flat) / len(flat)) < 1e-9
    assert abs(curves.least_squares_slope(flat)) < 1e-9


# ---------------------------------------------------------------- T-013 (G-02)

def test_t013_curve_is_invariant_to_an_added_offset():
    """同じ形に一定の下駄を履かせても、出来上がる曲線は変わらない。

    実データの一例から作る —— 自作の例文だけで済ませない(HC-068)。
    """
    sample = _first_complete_year()
    shifted = [v + 137.0 for v in sample.bins]

    a = curves.detrend(sample.bins)
    b = curves.detrend(shifted)
    for x, y in zip(a, b):
        assert x == pytest.approx(y, abs=1e-9)


# ---------------------------------------------------------------- T-014 (G-16)

def test_t014_every_example_is_fully_observed():
    """採った地点年は 24 ビンすべてが実測で埋まっている。補間は一切しない。"""
    examples = curves.build_examples("co2")
    for ex in examples:
        assert ex.filled_bins == curves.N_BINS, f"{ex.key}: 実測ビンが {ex.filled_bins}"
        assert ex.interpolated == 0, f"{ex.key}: 補間した点がある"


def test_t014_positive_control_incomplete_year_is_dropped():
    """陽性対照: 1 ビン欠けた地点年は採られない(HC-041)。

    実データの採用例から 1 ビンぶんの標本を抜いて渡す。
    """
    complete = _first_complete_year()
    assert complete.filled == curves.N_BINS, "この対照の前提(全ビン埋まり)が崩れている"

    holed = [None if i == 7 else v for i, v in enumerate(complete.bins)]
    assert curves.accept_bins(holed) is False, "欠けたビンを持つ地点年が採られてしまう"
    assert curves.accept_bins(complete.bins) is True, "正常な地点年を落としている(HC-074)"


# ---------------------------------------------------------------- T-015 (G-04)

def test_t015_split_never_shares_a_group():
    """群ばらし split で、同一群の地点年が train/test に跨らない。"""
    examples = curves.build_examples("co2")
    folds = curves.leave_one_group_out(examples)
    assert folds, "分割が 0 件"

    for train_idx, test_idx in folds:
        train_groups = {examples[i].group for i in train_idx}
        test_groups = {examples[i].group for i in test_idx}
        assert not (train_groups & test_groups), f"群が跨っている: {train_groups & test_groups}"
        assert test_idx, "test 側が空の分割がある"


def test_t015_positive_control_leaky_split_is_caught():
    """陽性対照: 同じ群を両側に置いた分割を検査が落とすこと(HC-041)。"""
    examples = curves.build_examples("co2")
    leaky = [(list(range(len(examples))), [0])]

    assert curves.split_is_clean(examples, curves.leave_one_group_out(examples)) is True
    assert curves.split_is_clean(examples, leaky) is False


# ---------------------------------------------------------------- T-016 (G-04)

def test_t016_nearby_sites_share_a_group():
    """大円距離 500 km 未満の地点対は同じ群に入る。

    陽性対照(HC-070): そのような対が実データ中に現に存在することを先に表明する。
    存在しなければこの検査は何も見ていない。
    """
    positions = curves.site_positions("co2")
    groups = curves.site_groups("co2")

    close_pairs = [
        (a, b)
        for i, a in enumerate(sorted(positions))
        for b in sorted(positions)[i + 1 :]
        if curves.great_circle_km(positions[a], positions[b]) < curves.GROUP_RADIUS_KM
    ]
    assert close_pairs, "500 km 未満の地点対が 1 組も無い —— この検査は何も見ていない"

    for a, b in close_pairs:
        assert groups[a] == groups[b], f"{a} と {b} が別群になっている"


# ---------------------------------------------------------------- T-019 (G-09)

def test_t019_first_harmonic_agrees_between_two_paths():
    """第一調和の係数を最小二乗と離散フーリエ和の二経路で出し、一致すること。

    等間隔・全ビン埋まりなので二経路は理論上一致する(HC-073 の言う「一致しうるか」の確認)。
    """
    examples = curves.build_examples("co2")
    assert examples, "走査対象が空"

    for ex in examples[:200]:
        a1, b1 = curves.first_harmonic_least_squares(ex.values)
        a2, b2 = curves.first_harmonic_dft(ex.values)
        assert a1 == pytest.approx(a2, abs=1e-9), f"{ex.key}: a1 が二経路で食い違う"
        assert b1 == pytest.approx(b2, abs=1e-9), f"{ex.key}: b1 が二経路で食い違う"


def test_t019_positive_control_shifted_path_is_caught():
    """陽性対照: 位相を 1 ビンずらした経路を照合が落とすこと(HC-065)。"""
    ex = curves.build_examples("co2")[0]
    rolled = ex.values[1:] + ex.values[:1]

    a1, b1 = curves.first_harmonic_least_squares(ex.values)
    a2, b2 = curves.first_harmonic_dft(rolled)
    assert abs(a1 - a2) > 1e-6 or abs(b1 - b2) > 1e-6, (
        "経路をずらしても同じ値が出る —— この照合は経路を見ていない"
    )
