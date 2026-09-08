"""出荷する重みと、二実装照合の基準を書き出す(L3)。

**出荷する模型には保留した試験が無い。** 全 805 地点年で学習した一本だからである。
その一般化の見込みは L2 の leave-one-group-out 実測(地点あたり MAE 12.40 度)であり、
この模型自身の成績ではない。混同されないよう、メタデータにも SPEC にも書く(T-040)。

止め時は L2 と同じ作法で決める —— 群単位で内側の検証を切り出し、そこが最良の時点を採る。
試験側が無いので「最後まで回して最後を採る」ことも原理的には可能だが、
それは L2 の測定と違う手続きになり、出荷模型と測定値の対応が切れる。
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pipeline import curves, experiment, model

# **このモジュールは x64 を触らない。** float64 の基準は `pipeline/reference64.py` が
# 別プロセスで作る。同じところに置いて試したら、optax の更新が float64 へ昇格して
# **学習の軌跡まで変わった**(止め時 249 → 281 エポック、内側検証 7.89 → 7.79)。
# 数値の大域設定は、それを入れた理由と無関係な結果を静かに変える
# (loop_003 TOOL-ENV / HC-207)。

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data" / "model_weights.json"
REFERENCE = ROOT / "data" / "reference_forward.json"

#: 中間活性まで書き出す例の数。全例ぶん書くと数百 KB になるので、
#: 出力は全例・経路は抜き取りにする。抜き取りは緯度の広がりを保って選ぶ。
N_ACTIVATION_SAMPLES = 24

#: 書き出す小数の桁。float32 の有効桁(約 7 桁)より広く取り、
#: 丸めが 1e-6 の照合閾値に効かないようにする。
DIGITS = 12


def _round(value):
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return round(float(value), DIGITS)


def train_shipping_model(examples, x, y, seed: int = 0):
    """全例で学習する。止め時だけは群単位の内側検証で決める。"""
    groups = [ex.group for ex in examples]
    val_groups = experiment._inner_split(groups, held=None, seed=seed)

    train_w = np.zeros((1, len(examples)), dtype=np.float32)
    val_w = np.zeros((1, len(examples)), dtype=np.float32)
    for i, group in enumerate(groups):
        if group in val_groups:
            val_w[0, i] = 1.0
        else:
            train_w[0, i] = 1.0

    stacked = jax.tree.map(lambda leaf: jnp.stack([leaf]), model.init_params(seed, channels=1))
    trainer = experiment._make_trainer()
    best, best_val, best_epoch = trainer(
        stacked, jnp.asarray(x), jnp.asarray(y), jnp.asarray(train_w), jnp.asarray(val_w)
    )
    params = jax.tree.map(lambda leaf: leaf[0], best)
    return params, float(best_val[0]), int(best_epoch[0]), int(val_w.sum())


def weights_as_lists(params) -> dict:
    return {name: _round(np.asarray(params[name]).tolist()) for name in sorted(params)}


def pick_samples(examples, count: int) -> list:
    """緯度の広がりを保って抜き取る。片寄った抜き取りは照合の射程を狭める。"""
    order = sorted(range(len(examples)), key=lambda i: examples[i].latitude)
    step = max(1, len(order) // count)
    return sorted(order[::step][:count])


def main() -> int:
    examples = curves.build_examples("co2")
    x, y = model.to_arrays(examples)

    baseline = json.loads((ROOT / "data" / "model.json").read_text(encoding="utf-8"))
    logo_mae = baseline["systems"]["cnn_co2"]["mae"]

    params, val_mae, epoch, val_examples = train_shipping_model(examples, x, y)
    print(f"出荷模型: {len(examples)} 例で学習 / 止め時 {epoch} エポック "
          f"(内側検証 {val_examples} 例で MAE {val_mae:.2f})")

    outputs, acts = model.forward_with_activations(params, jnp.asarray(x))
    outputs = np.asarray(outputs)

    payload = weights_as_lists(params)
    payload["metadata"] = {
        "generated_from": "pipeline/export_model.py",
        "architecture": "circular 1D CNN (24 points, kernel 5, 8-8 channels, dense 8)",
        "params": model.count_params(params),
        "input_scale_ppm": model.INPUT_SCALE,
        "latitude_scale_deg": model.LATITUDE_SCALE,
        "n_points": model.N_POINTS,
        "trained_on_all_examples": True,
        "n_training_examples": len(examples),
        "stopping_epoch": epoch,
        "inner_val_mae": val_mae,
        "generalisation_estimate_mae": logo_mae,
        "generalisation_estimate_source": (
            "L2 の leave-one-group-out 実測(57 群・種 3 通りの平均)。"
            "**この模型自身の成績ではない** —— 出荷模型は全例で学習しており保留した試験が無い"
        ),
    }
    WEIGHTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sample_indices = pick_samples(examples, N_ACTIVATION_SAMPLES)
    reference = {
        "generated_from": "pipeline/export_model.py",
        "n_examples": len(examples),
        "tolerance": 1e-6,
        "inputs": _round(x[:, :, 0].tolist()),
        "outputs": _round(outputs.tolist()),
        "latitudes": _round([ex.latitude for ex in examples]),
        "codes": [ex.code for ex in examples],
        "years": [ex.year for ex in examples],
        "sample_indices": sample_indices,
        "activations": {
            key: _round(np.asarray(acts[key])[sample_indices].tolist())
            for key in ("h1", "h2", "pooled", "d1")
        },
    }
    REFERENCE.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"重み {WEIGHTS.stat().st_size / 1024:.1f} KB / "
          f"基準 {REFERENCE.stat().st_size / 1024:.1f} KB")
    print(f"予測の範囲: {outputs.min():.1f} 〜 {outputs.max():.1f} 度 "
          f"(正解は {min(e.latitude for e in examples):.1f} 〜 "
          f"{max(e.latitude for e in examples):.1f} 度)")
    print("次に `python -m pipeline.reference64` を走らせて float64 の基準を足すこと。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
