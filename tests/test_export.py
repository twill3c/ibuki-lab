"""出荷物と基準に対する検査(TEST_SPEC T-034 / T-039 / T-040)。

TS 側は `src/model/forward.test.ts` が同じ資産を別の言語から見る。
ここが守るのは Python 側の不変量と、**文書との突合**である。
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data" / "model_weights.json"
REFERENCE = ROOT / "data" / "reference_forward.json"
SPEC = ROOT / "SPEC.md"

#: N-02 が定める出荷物の上限。
MAX_WEIGHT_BYTES = 1_000_000


def _weights():
    if not WEIGHTS.exists():
        pytest.skip("data/model_weights.json が無い(python -m pipeline.export_model で作る)")
    return json.loads(WEIGHTS.read_text(encoding="utf-8"))


def _reference():
    if not REFERENCE.exists():
        pytest.skip("data/reference_forward.json が無い")
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- T-034 (G-01)

def test_t034_weights_match_the_training_side_shapes():
    """重みの形が学習側の定義と一致する。"""
    from pipeline import model

    weights = _weights()
    params = model.init_params(0, channels=1)

    for name, array in params.items():
        import numpy as np

        expected = np.asarray(array).shape
        got = np.asarray(weights[name]).shape
        assert got == expected, f"{name}: 形が {got} で、学習側の {expected} と違う"


def test_t034_weights_fit_the_shipping_budget():
    """出荷物が N-02 の上限を下回り、SPEC に書いたバイト数と一致する。"""
    size = WEIGHTS.stat().st_size
    assert size < MAX_WEIGHT_BYTES, f"重みが {size} バイトで上限 {MAX_WEIGHT_BYTES} を超える"

    written = f"重み {size / 1024:.1f} KB"
    assert written in SPEC.read_text(encoding="utf-8"), (
        f"SPEC §7.12 のバイト数が実体と食い違う(実体は {written})"
    )


# ---------------------------------------------------------------- T-039 (G-01)

def test_t039_reference_inputs_are_the_real_curves():
    """基準の入力が `curves.build_examples` の値そのものであること。

    基準の側が壊れていたら、照合は緑のまま何も言わない。
    """
    from pipeline import curves

    reference = _reference()
    examples = curves.build_examples("co2")

    assert reference["n_examples"] == len(examples)
    assert reference["codes"][:5] == [ex.code for ex in examples[:5]]

    # 基準が保存しているのは **float32 に落とした曲線を 12 桁で丸めた値**である。
    # float64 の元値と絶対 1e-9 で比べると、float32 の量子化(実測 6.5e-9)で落ちる ——
    # 単位も精度も違う量へ同じ閾値を置かない(HC-232)。
    import numpy as np

    for row, ex in zip(reference["inputs"][:40], examples[:40]):
        for got, expected in zip(row, ex.values):
            quantised = float(np.float32(expected))
            assert abs(got - quantised) < 1e-11, f"{ex.key}: 基準の入力が曲線と違う"


def test_t039_float64_reference_is_present_and_differs_from_float32():
    """float64 の基準があり、float32 の基準と**実際に違う**こと。

    同じなら、この二段構えは何も分けていない(HC-071 の「起きた証拠」)。
    """
    reference = _reference()
    assert "outputs_float64" in reference, "float64 の基準が無い(reference64 を走らせる)"

    gap = reference["float32_vs_float64"]
    assert gap["max_abs_deg"] > 1e-6, "float32 と float64 が同じ —— 二段構えが働いていない"
    assert gap["max_normalised"] < 1e-6, "無次元化しても差が大きい —— 丸めでは説明できない"


# ---------------------------------------------------------------- T-040 (G-14)

def test_t040_shipping_model_states_it_has_no_held_out_test():
    """出荷模型に保留した試験が無いことが、重みのメタデータと SPEC の両方に書いてある。"""
    meta = _weights()["metadata"]
    spec = SPEC.read_text(encoding="utf-8")

    assert meta["trained_on_all_examples"] is True
    assert "leave-one-group-out" in meta["generalisation_estimate_source"]
    assert "この模型自身の成績ではない" in meta["generalisation_estimate_source"]

    assert "出荷する模型には保留した試験が無い" in spec
    assert f"**{meta['generalisation_estimate_mae']:.2f} 度**" in spec, (
        "SPEC §7.12 の一般化の見込みがメタデータと食い違う"
    )
    assert f"**{meta['stopping_epoch']} エポック**" in spec


def test_t040_positive_control_metadata_mismatch_is_caught():
    """陽性対照: メタデータの数をずらすと突合が撃つこと(HC-041)。"""
    original = WEIGHTS.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["metadata"]["stopping_epoch"] += 1
    try:
        WEIGHTS.write_text(
            json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with pytest.raises(AssertionError):
            test_t040_shipping_model_states_it_has_no_held_out_test()
    finally:
        WEIGHTS.write_text(original, encoding="utf-8")
