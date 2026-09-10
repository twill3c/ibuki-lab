"""要約の切替に対する検査(TEST_SPEC T-056..T-058)。

L7 の実験は「要約の取り方**だけ**を変えた ablation」である。それが本当に
一点だけの違いであることと、切替が狙った性質を実際に変えることを確かめる。
"""
import numpy as np
import pytest

from pipeline import model


def _batch(n: int = 6, channels: int = 1, seed: int = 20260910):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, model.N_POINTS, channels)).astype(np.float32)


def _with_mode(mode: str, fn):
    original = model.POOL_MODE
    model.POOL_MODE = mode
    try:
        return fn()
    finally:
        model.POOL_MODE = original


# ---------------------------------------------------------------- T-056 (G-09)

def test_t056_invariant_pooling_cannot_see_phase():
    """位相なしの要約では、曲線を回しても出力が動かない。

    これは L6 で見つけた機構そのもの(SPEC §7.15)。**負けの理由**なので、
    直したあとも「直す前はこうだった」を検査として残す。
    """
    x = _batch()

    def run():
        params = model.init_params(0, channels=1)
        base = np.asarray(model.forward(params, x))
        worst = 0.0
        for shift in (1, 6, 12, 18):
            rolled = np.roll(x, shift, axis=1)
            worst = max(worst, float(np.max(np.abs(np.asarray(model.forward(params, rolled)) - base))))
        return worst

    assert _with_mode("invariant", run) < 1e-5


def test_t056_harmonic_pooling_sees_phase():
    """位相ありの要約では、回すと出力が動く。**切替が狙った性質を変えた証拠。**"""
    x = _batch()

    def run():
        params = model.init_params(0, channels=1)
        base = np.asarray(model.forward(params, x))
        rolled = np.roll(x, 6, axis=1)
        return float(np.max(np.abs(np.asarray(model.forward(params, rolled)) - base)))

    assert _with_mode("harmonic", run) > 1.0


# ---------------------------------------------------------------- T-057 (G-09)

def test_t057_only_the_summary_differs():
    """二つの系の違いが**要約の次元だけ**であること。

    層の数も畳み込みの形も変わっていない。変わったのは pooled の幅と、
    それを受ける dense1 の入力側だけである —— ここが増えていれば ablation が
    「一点だけの違い」でなくなる(HC-064)。
    """
    invariant = _with_mode("invariant", lambda: model.init_params(0, channels=1))
    harmonic = _with_mode("harmonic", lambda: model.init_params(0, channels=1))

    assert set(invariant) == set(harmonic), "重みの顔ぶれが変わっている"
    for name in invariant:
        if name == "dense1_w":
            continue
        assert np.asarray(invariant[name]).shape == np.asarray(harmonic[name]).shape, (
            f"{name} の形が変わっている —— 要約以外にも手が入っている"
        )

    assert np.asarray(invariant["dense1_w"]).shape[0] == model.WIDTH2 * 2
    assert np.asarray(harmonic["dense1_w"]).shape[0] == model.WIDTH2 * 4


def test_t057_parameter_count_grows_only_by_the_summary():
    """増えた重みが、要約の増分ぶんちょうどであること。"""
    invariant = _with_mode("invariant", lambda: model.count_params(model.init_params(0, 1)))
    harmonic = _with_mode("harmonic", lambda: model.count_params(model.init_params(0, 1)))
    assert harmonic - invariant == model.WIDTH2 * 2 * model.DENSE


# ---------------------------------------------------------------- T-058 (G-09)

def test_t058_harmonic_summary_rotates_with_the_input():
    """位相ありの要約が、入力を回すと**回る**(消えるのでも壊れるのでもない)。

    第一調和の (a, b) は、入力を θ 回すと (a, b) が θ 回転する。
    大きさ √(a²+b²) は保たれる —— それが「位相だけが変わった」ことの意味である。
    """

    def run():
        params = model.init_params(1, channels=1)
        x = _batch(n=1, seed=7)
        _, acts = model.forward_with_activations(params, x)
        pooled = np.asarray(acts["pooled"])[0]

        rolled = np.roll(x, 6, axis=1)
        _, acts_r = model.forward_with_activations(params, rolled)
        pooled_r = np.asarray(acts_r["pooled"])[0]
        return pooled, pooled_r

    pooled, pooled_r = _with_mode("harmonic", run)
    width = model.WIDTH2

    # 前半(平均・絶対値の最大)はずらしに不変なので変わらない
    assert np.allclose(pooled[: 2 * width], pooled_r[: 2 * width], atol=1e-5)

    # 後半(第一調和)は変わるが、大きさは保たれる
    a, b = pooled[2 * width : 3 * width], pooled[3 * width :]
    ar, br = pooled_r[2 * width : 3 * width], pooled_r[3 * width :]
    assert not np.allclose(a, ar, atol=1e-4), "位相の要約が動いていない"
    assert np.allclose(np.hypot(a, b), np.hypot(ar, br), atol=1e-5), (
        "大きさまで変わっている —— 位相だけが変わったとは言えない"
    )


def test_t058_negative_control_invariant_summary_has_no_harmonic_part():
    """陰性対照: 位相なしの要約には、そもそも回る部分が無い。"""

    def run():
        params = model.init_params(1, channels=1)
        _, acts = model.forward_with_activations(params, _batch(n=1, seed=7))
        return np.asarray(acts["pooled"]).shape[1]

    assert _with_mode("invariant", run) == model.WIDTH2 * 2


# ---------------------------------------------------------------- T-059 (G-10)

def test_t059_ch4_shuffle_control_actually_shuffles():
    """CH4 の入れ替え対照が、**全例で実際に入れ替わっている**こと。

    自分自身に当たった例が残っていれば、その分だけ対照が薄まる。
    入れ替えが効いていなければ、対照は何も言わない(HC-071)。
    """
    from pipeline import curves, experiment

    co2 = curves.build_examples("co2")
    ch4 = {(e.code, e.year): e for e in curves.build_examples("ch4")}
    paired = [e for e in co2 if (e.code, e.year) in ch4]
    second = [ch4[(e.code, e.year)] for e in paired]

    shuffled = experiment.shuffled_second_channel(paired, second)
    assert len(shuffled) == len(second)

    same = sum(1 for a, b in zip(second, shuffled) if a is b)
    assert same == 0, f"{same} 例が入れ替わっていない"

    # 値の集合は保たれる(分布も滑らかさも本物のまま、対応だけ壊れている)
    assert sorted(id(s) for s in shuffled) == sorted(id(s) for s in second)


def test_t059_shuffle_is_deterministic():
    """同じ種で二度呼べば同じ入れ替えになる(測定が再現する)。"""
    from pipeline import curves, experiment

    co2 = curves.build_examples("co2")
    ch4 = {(e.code, e.year): e for e in curves.build_examples("ch4")}
    paired = [e for e in co2 if (e.code, e.year) in ch4]
    second = [ch4[(e.code, e.year)] for e in paired]

    a = experiment.shuffled_second_channel(paired, second)
    b = experiment.shuffled_second_channel(paired, second)
    assert [x.key for x in a] == [x.key for x in b]
