"""模型の内側を測る。成績ではなく**機構**を見るための道具。

いちばん見たいのは第一層の飽和である。この課題では**振幅が信号**なので、
入力 20 ppm と 30 ppm が同じ tanh の平らな部分に落ちていたら、模型は
振幅を区別できない。成績が悪かったときに「模型が弱い」で片づけないために、
先に測れるようにしておく。
"""
from __future__ import annotations

import numpy as np

from pipeline import curves, model

#: |tanh| がこれを超えたら「飽和している」と数える。tanh(2.0) = 0.964。
SATURATION = 0.95


def activation_report(params, x) -> dict:
    _, acts = model.forward_with_activations(params, x)
    out = {}
    for name in ("h1", "h2"):
        values = np.abs(np.asarray(acts[name]))
        out[name] = {
            "saturated_fraction": float(np.mean(values > SATURATION)),
            "median_abs": float(np.median(values)),
            "p99_abs": float(np.percentile(values, 99)),
        }
    return out


def input_scale_report(gas: str = "co2") -> dict:
    """入力の大きさそのもの。飽和の話をする前に、何が入っているかを出す。"""
    examples = curves.build_examples(gas)
    x, _ = model.to_arrays(examples)
    peak = np.max(np.abs(x), axis=(1, 2))
    return {
        "abs_max_median": float(np.median(peak)),
        "abs_max_p95": float(np.percentile(peak, 95)),
        "abs_max_max": float(np.max(peak)),
    }


def main() -> int:
    examples = curves.build_examples("co2")
    x, _ = model.to_arrays(examples)

    print("入力の大きさ(|ppm| の地点年ごとの最大):")
    for key, value in input_scale_report().items():
        print(f"  {key}: {value:.2f}")

    print()
    print("初期重みでの活性(学習前):")
    report = activation_report(model.init_params(0, channels=1), x)
    for layer, stats in report.items():
        print(
            f"  {layer}: 飽和率 {stats['saturated_fraction']*100:5.1f}% / "
            f"中央値 {stats['median_abs']:.3f} / p99 {stats['p99_abs']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
