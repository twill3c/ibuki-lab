"""緯度を当てる 1D CNN(JAX・素)。

**Flax を使わず前向きを自前で書く。** Windows の MAX_PATH で Flax が入らなかった
という事情が発端だが(loop_002 TOOL-ENV)、結果として G-01 の TS 前向きが
ここの式の直訳になるので、制約を利点として受ける。

設計で効いているのは二つ。

- **畳み込みは巡回**である。一年は環なので、12 月と 1 月は隣り合っている。
  端を 0 で埋めると「年の切れ目」という実在しない構造を模型に教えてしまう
- **小さい。** 独立標本は 57 群しかない。層と幅はそれに見合う大きさにする
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

#: 入力の点数(半月ビン)。curves.N_BINS と一致していなければならない。
N_POINTS = 24

#: 畳み込みの幅と出力チャンネル。
KERNEL = 5
WIDTH1 = 8
WIDTH2 = 8
DENSE = 8

#: 緯度をこの値で割ってから学習する(度のまま学ぶと初期損失が大きく暴れる)。
LATITUDE_SCALE = 90.0

#: 要約の取り方。**この切替だけが位相の見え方を変える。他は一切変えない。**
#:
#: - "invariant": 時間方向の平均と絶対値の最大。どちらもずらしに**不変**なので、
#:   前向き全体が位相を見ない(L6 で発見した機構 — SPEC §7.15)
#: - "harmonic": 上に加えて、学習した各チャンネルの**第一調和の係数** (a_c, b_c) を連結する。
#:   入力を回すと (a_c, b_c) も回るので、**位相が要約に残る**
#:
#: 生の入力の第一調和を渡せば相手(2 特徴 kNN)の特徴そのものになってしまうので、
#: **学習後のチャンネルに対して**取る。渡すのは「位相を見る道具」であって答えではない。
POOL_MODE = "invariant"

#: 入力をこの値(ppm)で割ってから第一層に入れる。
#:
#: **振幅を正規化するのではない。全例に同じ定数を掛けるので、地点間の振幅の比は保たれる。**
#: 生の ppm のまま入れると tanh が初期時点で 51.3% 飽和し(実測 2026-09-08、
#: |tanh| 中央値 0.958)、平らな部分で振幅 20 と 30 が区別できなくなる ——
#: この課題で信号そのものである量を第一層が捨てる。
#: 定数は年周振幅の桁(地点年ごとの |ppm| 最大の中央値 6.75)から桁だけを取り、
#: **試験側の成績は一切見ずに**決めた。
INPUT_SCALE = 10.0


# ------------------------------------------------------------------ 重み

def init_params(seed: int, channels: int = 1, dtype=np.float32) -> dict:
    """He 初期化に近いスケールの一様乱数。numpy で作るので JAX の版に依らない。

    `dtype` は既定で float32。float64 を渡せるのは有限差分照合(T-025)のためで、
    学習は常に float32 で回す。
    """
    rng = np.random.default_rng(20260906 + seed)

    def draw(shape, fan_in):
        limit = np.sqrt(6.0 / fan_in)
        return rng.uniform(-limit, limit, size=shape).astype(dtype)

    def zeros(shape):
        return jnp.asarray(np.zeros(shape, dtype=dtype))

    return {
        "conv1_w": jnp.asarray(draw((KERNEL, channels, WIDTH1), KERNEL * channels)),
        "conv1_b": zeros((WIDTH1,)),
        "conv2_w": jnp.asarray(draw((KERNEL, WIDTH1, WIDTH2), KERNEL * WIDTH1)),
        "conv2_b": zeros((WIDTH2,)),
        "dense1_w": jnp.asarray(draw((pooled_width(), DENSE), pooled_width())),
        "dense1_b": zeros((DENSE,)),
        "dense2_w": jnp.asarray(draw((DENSE, 1), DENSE)),
        "dense2_b": zeros((1,)),
    }


def pooled_width() -> int:
    """要約の次元。`POOL_MODE` で決まる。"""
    return WIDTH2 * (4 if POOL_MODE == "harmonic" else 2)


def param_names(params: dict) -> list:
    return sorted(params)


def count_params(params: dict) -> int:
    return int(sum(np.asarray(v).size for v in params.values()))


def bump_param(params: dict, name: str, delta: float) -> dict:
    """指定した重みの第 1 要素だけを動かした複製を返す(検査用)。"""
    out = dict(params)
    flat = jnp.reshape(params[name], (-1,))
    flat = flat.at[0].add(delta)
    out[name] = jnp.reshape(flat, params[name].shape)
    return out


def pick_scalar(tree: dict, name: str) -> float:
    return float(jnp.reshape(tree[name], (-1,))[0])


# ------------------------------------------------------------------ 前向き

def circular_conv1d_lax(x, w, b):
    """巡回畳み込み(XLA の畳み込み演算を使う経路)。

    `x` は (N, L, Cin)、`w` は (K, Cin, Cout)。一年は環なので端を巡回で埋める。
    0 埋めにすると年の切れ目という実在しない構造を教えることになる。
    """
    pad = (w.shape[0] - 1) // 2
    padded = jnp.concatenate([x[:, -pad:, :], x, x[:, :pad, :]], axis=1)
    out = jax.lax.conv_general_dilated(
        padded,
        w,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    return out + b


def circular_conv1d_matmul(x, w, b):
    """同じ巡回畳み込みを、巡回の取り出しと行列積で書いた経路。

    `jnp.roll` で K 本ずらした写しを作って並べ、(N, L, K*Cin) と (K*Cin, Cout) の
    行列積にする。**畳み込みではなく行列積**になるので、fold を `vmap` で束ねたときに
    XLA が素直に大きな行列積へまとめられる。TS への直訳もこちらのほうが易しい。

    数式としては `circular_conv1d_lax` と同一であり、T-029 が実データで照合する。
    """
    kernel = w.shape[0]
    pad = (kernel - 1) // 2
    shifted = jnp.concatenate(
        [jnp.roll(x, pad - k, axis=1) for k in range(kernel)], axis=2
    )
    flat_w = jnp.reshape(w, (kernel * w.shape[1], w.shape[2]))
    return shifted @ flat_w + b


#: 前向きが使う経路。二つは実データで照合してある(T-029)。
circular_conv1d = circular_conv1d_matmul


def forward_with_activations(params: dict, x):
    """緯度(度)と、途中の活性を返す。中間層は G-01 の照合に使う。"""
    scaled = x / INPUT_SCALE
    h1 = jnp.tanh(circular_conv1d(scaled, params["conv1_w"], params["conv1_b"]))
    h2 = jnp.tanh(circular_conv1d(h1, params["conv2_w"], params["conv2_b"]))

    # 位置に依らない要約を取る。平均だけだと「どれだけ振れたか」が消えるので、
    # 絶対値の最大を並べる。**どちらも巡回のずらしに対して不変**で、
    # invariant のままだと前向き全体が位相を見ない(SPEC §7.15)。
    parts = [jnp.mean(h2, axis=1), jnp.max(jnp.abs(h2), axis=1)]
    if POOL_MODE == "harmonic":
        # 学習した各チャンネルの第一調和。入力を回すと (a, b) も回るので位相が残る。
        length = h2.shape[1]
        angles = 2.0 * jnp.pi * (jnp.arange(length) + 0.5) / length
        cos = jnp.cos(angles)[None, :, None]
        sin = jnp.sin(angles)[None, :, None]
        parts.append(2.0 * jnp.mean(h2 * cos, axis=1))
        parts.append(2.0 * jnp.mean(h2 * sin, axis=1))
    pooled = jnp.concatenate(parts, axis=1)

    d1 = jnp.tanh(pooled @ params["dense1_w"] + params["dense1_b"])
    out = (d1 @ params["dense2_w"] + params["dense2_b"])[:, 0]
    return out * LATITUDE_SCALE, {"h1": h1, "h2": h2, "pooled": pooled, "d1": d1}


def forward(params: dict, x):
    return forward_with_activations(params, x)[0]


def loss(params: dict, x, y) -> float:
    """平均絶対誤差(度)。評価指標と同じものを最小化する。"""
    return float(jnp.mean(jnp.abs(forward(params, x) - y)))


def residuals(params: dict, x, y):
    """予測 − 正解。有限差分照合の前提(符号が変わっていないこと)を測るのに使う。

    MAE は残差 0 で微分できない。eps の両側で残差の符号が変われば、
    数値微分は自動微分と**別のもの**を測る —— 一致しえない照合になる(HC-073)。
    """
    return np.asarray(forward(params, x)) - np.asarray(y)


def _loss_jax(params, x, y):
    return jnp.mean(jnp.abs(forward(params, x) - y))


loss_grad = jax.grad(_loss_jax)


# ------------------------------------------------------------------ 入力の組み立て

def to_arrays(examples, second=None):
    """例の並びを (N, 24, C) の入力と (N,) の緯度に落とす。

    `second` を渡すと第 2 チャンネルになる(CH4)。**曲線の値以外は何も入れない。**
    """
    x = np.array([[list(ex.values)] for ex in examples], dtype=np.float32)
    x = np.transpose(x, (0, 2, 1))  # (N, 24, 1)
    if second is not None:
        extra = np.array([[list(s.values)] for s in second], dtype=np.float32)
        extra = np.transpose(extra, (0, 2, 1))
        x = np.concatenate([x, extra], axis=2)
    y = np.array([ex.latitude for ex in examples], dtype=np.float32)
    return x, y


def input_is_clean(x, examples) -> bool:
    """入力のどの点も、緯度そのものと一致していないこと(G-02 の漏れ検査)。

    曲線は ppm の偏差で平均 0、緯度は度で桁も分布も違う。**一致したら漏れている。**
    """
    y = np.array([ex.latitude for ex in examples], dtype=np.float32)
    for channel in range(x.shape[2]):
        for point in range(x.shape[1]):
            if np.allclose(x[:, point, channel], y, atol=1e-4):
                return False
    return True
