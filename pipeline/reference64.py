"""float64 で通した照合の基準を作る(L3)。

**このモジュールだけが x64 を有効にする。** `export_model.py` と同じところに置いたら、
optax の更新が float64 へ昇格して学習の軌跡まで変わった(loop_003 TOOL-ENV)。
数値の大域設定は、それを入れた理由と無関係な結果を静かに変えるので、
**学習には一切触れないプロセスに隔離する**。

ここが作る基準の役目は一つ。**「移植が厳密か」と「float32 の丸めか」を分けること。**

- TS(JS の数値 = float64)は、この float64 基準と **1e-9 で一致するはず**である。
  一致すれば、移植の式は厳密に同じだと言える
- 出荷経路(float32 で学習した重みを float32 で通した基準)とのずれは、
  **丸めの床**であって移植の誤りではない。±90 度の量に絶対 1e-6 を要求するのは
  原理的に不可能で、閾値は無次元化してから置く(HC-073)
"""
from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pipeline import curves, model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data" / "model_weights.json"
REFERENCE = ROOT / "data" / "reference_forward.json"

DIGITS = 15


def _round(value):
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return round(float(value), DIGITS)


def load_weights() -> dict:
    """書き出した JSON から重みを読み直す。

    **学習した重みをそのまま渡さない。** TS が読むのは JSON なので、
    JSON を経由した値で基準を作らなければ、桁落ちが照合に紛れ込む。
    """
    raw = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    return {
        name: jnp.asarray(np.asarray(raw[name], dtype=np.float64))
        for name in raw
        if name != "metadata"
    }


def main() -> int:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    params = load_weights()

    examples = curves.build_examples("co2")
    x, _ = model.to_arrays(examples)
    if len(examples) != reference["n_examples"]:
        raise SystemExit("基準の例数と現在の母集団が食い違う。export_model を先に走らせること")

    outputs, acts = model.forward_with_activations(
        params, jnp.asarray(np.asarray(x, dtype=np.float64))
    )
    outputs = np.asarray(outputs)

    float32_outputs = np.asarray(reference["outputs"], dtype=np.float64)
    gap = np.abs(outputs - float32_outputs)
    print(
        f"float32 基準との差: 中央値 {np.median(gap):.3e} / 最大 {gap.max():.3e} 度 "
        f"(無次元化すると最大 {gap.max() / model.LATITUDE_SCALE:.3e})"
    )

    # 顕著度(入力に対する勾配)の基準。TS 側は中心差分で近似するので、
    # **自動微分との照合**で近似が妥当な範囲にあることを示す(T-052)。
    def scalar_output(x_one):
        return model.forward(params, x_one[None, :, :])[0]

    grads = []
    for index in reference["sample_indices"]:
        x_one = jnp.asarray(np.asarray(x[index], dtype=np.float64))
        grads.append(np.asarray(jax.grad(scalar_output)(x_one))[:, 0].tolist())
    reference["input_gradients_float64"] = _round(grads)

    sample_indices = reference["sample_indices"]
    reference["outputs_float64"] = _round(outputs.tolist())
    reference["activations_float64"] = {
        key: _round(np.asarray(acts[key])[sample_indices].tolist())
        for key in ("h1", "h2", "pooled", "d1")
    }
    reference["float32_vs_float64"] = {
        "max_abs_deg": float(gap.max()),
        "median_abs_deg": float(np.median(gap)),
        "max_normalised": float(gap.max() / model.LATITUDE_SCALE),
        "note": (
            "学習側は float32、ブラウザ側は float64。この差は丸めの床であって移植の誤りではない。"
            "移植が厳密であることは outputs_float64 との 1e-9 照合が示す"
        ),
    }
    REFERENCE.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"基準 {REFERENCE.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
