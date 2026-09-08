"""模型そのものに対する検査(TEST_SPEC T-022 / T-024 / T-025 / T-026)。

学習の結果ではなく、**前向きと勾配が主張どおりの性質を持つか**を見る。
"""
import numpy as np
import pytest

from pipeline import model


def _tiny_batch(channels: int = 1, n: int = 5, seed: int = 20260906):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, model.N_POINTS, channels)).astype(np.float32)
    return x


# ---------------------------------------------------------------- T-022 (G-02)

def test_t022_input_tensor_carries_only_the_curve():
    """入力に入るのは曲線の値だけである。地点・年・経度・標高・絶対濃度は入らない。"""
    from pipeline import curves

    examples = curves.build_examples("co2")[:20]
    x, y = model.to_arrays(examples)

    assert x.shape == (20, model.N_POINTS, 1)
    for i, ex in enumerate(examples):
        assert np.allclose(x[i, :, 0], np.asarray(ex.values, dtype=np.float32), atol=1e-6), (
            f"{ex.key}: 入力が曲線そのものでない"
        )
    assert y.shape == (20,)


def test_t022_positive_control_leaked_feature_is_caught():
    """陽性対照: 緯度そのものを 1 点混ぜた入力を、漏れ検査が捕まえること(HC-041)。"""
    from pipeline import curves

    examples = curves.build_examples("co2")[:20]
    clean, y = model.to_arrays(examples)

    assert model.input_is_clean(clean, examples) is True, "正常な入力を落としている(HC-074)"

    leaked = clean.copy()
    leaked[:, 0, 0] = y  # 緯度を第 1 点に流し込む
    assert model.input_is_clean(leaked, examples) is False, "緯度の混入を見逃した"


# ---------------------------------------------------------------- T-024 (G-01)

def test_t024_forward_is_deterministic():
    """同じ重み・同じ入力なら二度呼んで完全一致する。"""
    params = model.init_params(0, channels=1)
    x = _tiny_batch()

    a = np.asarray(model.forward(params, x))
    b = np.asarray(model.forward(params, x))
    assert np.array_equal(a, b), "前向きが決定的でない"


def test_t024_positive_control_perturbed_weight_changes_output():
    """陽性対照: 重みを 1 要素動かせば出力が変わること。

    変わらなければ、その重みは前向きに関与していない —— 決定性の検査は
    「何も計算していない実装」でも緑になる。
    """
    params = model.init_params(0, channels=1)
    x = _tiny_batch()
    before = np.asarray(model.forward(params, x))

    unchanged = []
    for name in model.param_names(params):
        bumped = model.bump_param(params, name, 0.5)
        after = np.asarray(model.forward(bumped, x))
        if np.array_equal(before, after):
            unchanged.append(name)
    assert not unchanged, f"前向きに関与していない重みがある: {unchanged}"


# ---------------------------------------------------------------- T-025 (G-01)

def test_t025_autodiff_matches_finite_differences():
    """自動微分の勾配が有限差分と一致する(独立経路での検算)。

    **float64 で行う。** float32 のままだと中心差分の丸めが相対 1e-3 程度になり、
    二つの経路が理論上一致しえない閾値を掲げることになる(HC-073 / loop_002 VERIF-FALSE)。
    あわせて、MAE が微分できない残差 0 を跨いでいないことを先に確かめる ——
    跨いでいれば数値微分は自動微分と別のものを測っている。
    """
    params = model.init_params(1, channels=1, dtype=np.float64)
    x = _tiny_batch(n=4, seed=7).astype(np.float64)
    y = np.array([10.0, -30.0, 60.0, 0.0], dtype=np.float64)

    grads = model.loss_grad(params, x, y)
    eps = 1e-5

    checked = 0
    for name in model.param_names(params):
        plus_params = model.bump_param(params, name, eps)
        minus_params = model.bump_param(params, name, -eps)

        signs = [
            np.sign(model.residuals(p, x, y)) for p in (params, plus_params, minus_params)
        ]
        assert np.array_equal(signs[0], signs[1]) and np.array_equal(signs[0], signs[2]), (
            f"{name}: eps の両側で残差の符号が変わった —— この照合は成立しない"
        )

        analytic = model.pick_scalar(grads, name)
        numeric = (model.loss(plus_params, x, y) - model.loss(minus_params, x, y)) / (2 * eps)
        denom = max(1e-9, abs(analytic), abs(numeric))
        assert abs(analytic - numeric) / denom < 1e-6, (
            f"{name}: 自動微分 {analytic:.10g} 有限差分 {numeric:.10g}"
        )
        checked += 1
    assert checked == len(params), "確かめた重みが全部でない"


def test_t025_positive_control_wrong_gradient_is_caught():
    """陽性対照: 勾配を意図的にずらすと照合が撃つこと(HC-041)。

    照合が働いていなければ、間違った勾配も黙って通る。
    """
    params = model.init_params(1, channels=1, dtype=np.float64)
    x = _tiny_batch(n=4, seed=7).astype(np.float64)
    y = np.array([10.0, -30.0, 60.0, 0.0], dtype=np.float64)

    grads = model.loss_grad(params, x, y)
    eps = 1e-5
    name = "conv1_w"

    wrong = model.pick_scalar(grads, name) * 1.01
    numeric = (
        model.loss(model.bump_param(params, name, eps), x, y)
        - model.loss(model.bump_param(params, name, -eps), x, y)
    ) / (2 * eps)
    denom = max(1e-9, abs(wrong), abs(numeric))
    assert abs(wrong - numeric) / denom >= 1e-6, "1% ずらした勾配を照合が見逃す"


# ---------------------------------------------------------------- T-029 (G-01)

def test_t029_two_conv_paths_agree_on_real_data():
    """巡回畳み込みの二経路(XLA の畳み込み / 巡回の取り出し + 行列積)が一致する。

    速さのために経路を差し替えたので、**差し替えで何が変わっていないか**を
    実データに当てて確かめる(HC-070)。自作の例文だけで済ませない(HC-068)。
    """
    from pipeline import curves

    examples = curves.build_examples("co2")[:64]
    x, _ = model.to_arrays(examples)
    params = model.init_params(0, channels=1)

    a = np.asarray(model.circular_conv1d_lax(x / model.INPUT_SCALE, params["conv1_w"], params["conv1_b"]))
    b = np.asarray(model.circular_conv1d_matmul(x / model.INPUT_SCALE, params["conv1_w"], params["conv1_b"]))
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) < 1e-5, f"二経路の最大差 {np.max(np.abs(a - b)):.3g}"


def test_t029_positive_control_shifted_kernel_is_caught():
    """陽性対照: 片方の重みを 1 タップずらすと照合が撃つこと(HC-065)。

    ずらしても通るなら、この照合は位相を見ていない。
    """
    from pipeline import curves

    examples = curves.build_examples("co2")[:64]
    x, _ = model.to_arrays(examples)
    params = model.init_params(0, channels=1)
    rolled_w = np.roll(np.asarray(params["conv1_w"]), 1, axis=0)

    a = np.asarray(model.circular_conv1d_lax(x, params["conv1_w"], params["conv1_b"]))
    b = np.asarray(model.circular_conv1d_matmul(x, rolled_w, params["conv1_b"]))
    assert np.max(np.abs(a - b)) > 1e-5, "1 タップずらしても同じ値が出る"


# ---------------------------------------------------------------- T-028 (G-09)

def test_t028_first_layer_is_not_saturated_at_init():
    """第一層の tanh が初期時点で飽和していないこと。

    この課題では**振幅が信号**なので、tanh の平らな部分に落ちた入力は
    振幅の違いを失う。飽和したままでも成績は数字として出るし検査も緑になるので、
    内側を測る検査を別に置く(loop_002 GEN-LOGIC)。

    実測 2026-09-08: 生の ppm を入れると h1 の飽和率 51.3%、
    定数 10 ppm で割ると 1.2%。
    """
    from pipeline import curves, diagnose

    examples = curves.build_examples("co2")
    x, _ = model.to_arrays(examples)
    report = diagnose.activation_report(model.init_params(0, channels=1), x)

    assert report["h1"]["saturated_fraction"] < 0.10, (
        f"第一層の飽和率が {report['h1']['saturated_fraction']*100:.1f}% —— 振幅が潰れている"
    )
    assert report["h2"]["saturated_fraction"] < 0.10


def test_t028_positive_control_unscaled_input_saturates():
    """陽性対照: 入力を割らずに入れると、この検査が撃つこと(HC-041)。

    撃たなければ「飽和していない」は測っていないのと同じである。
    """
    from pipeline import curves, diagnose

    examples = curves.build_examples("co2")
    x, _ = model.to_arrays(examples)
    raw = x * model.INPUT_SCALE  # forward の中で割られるぶんを打ち消す
    report = diagnose.activation_report(model.init_params(0, channels=1), raw)

    assert report["h1"]["saturated_fraction"] > 0.10, (
        "生の ppm を入れても飽和が検出されない —— この検査は何も見ていない"
    )


# ---------------------------------------------------------------- T-026 (G-10)

def test_t026_second_channel_actually_reaches_the_output():
    """2 チャンネル系が CH4 を実際に読んでいることの証拠。

    潰しても出力が変わらないなら、その系は CH4 を測っていない —— 「足した」という
    宣言だけが残り、G-10 の判定が意味を失う(HC-071)。
    """
    params = model.init_params(2, channels=2)
    x = _tiny_batch(channels=2, n=6, seed=11)

    before = np.asarray(model.forward(params, x))

    flattened = x.copy()
    flattened[:, :, 1] = 0.0
    after = np.asarray(model.forward(params, flattened))

    assert not np.allclose(before, after), "CH4 チャンネルを潰しても出力が変わらない"


def test_t026_negative_control_flattening_an_absent_channel_changes_nothing():
    """陰性対照: 1 チャンネル系では潰す相手が無く、出力は当然変わらない。

    これが撃つようなら、上の対照は別の何かを見ている。
    """
    params = model.init_params(2, channels=1)
    x = _tiny_batch(channels=1, n=6, seed=11)

    before = np.asarray(model.forward(params, x))
    after = np.asarray(model.forward(params, x.copy()))
    assert np.array_equal(before, after)


def test_t026_channels_are_not_silently_symmetric():
    """CO2 と CH4 のチャンネルを入れ替えると出力が変わる。

    変わらないなら模型はチャンネルを区別しておらず、「CH4 を足した」と言えない。
    """
    params = model.init_params(3, channels=2)
    x = _tiny_batch(channels=2, n=6, seed=13)
    swapped = x[:, :, ::-1].copy()

    before = np.asarray(model.forward(params, x))
    after = np.asarray(model.forward(params, swapped))
    assert not np.allclose(before, after), "チャンネルの入れ替えが出力に効かない"
