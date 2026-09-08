"""学習と評価の運び方に対する検査(TEST_SPEC T-021 / T-023)。"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPORT = ROOT / "data" / "model.json"
BASELINE_REPORT = ROOT / "data" / "baseline.json"


def _reports():
    if not MODEL_REPORT.exists():
        pytest.skip("data/model.json が無い(python -m pipeline.experiment で作る)")
    return (
        json.loads(MODEL_REPORT.read_text(encoding="utf-8")),
        json.loads(BASELINE_REPORT.read_text(encoding="utf-8")),
    )


# ---------------------------------------------------------------- T-021 (G-09)

def test_t021_model_and_baseline_share_the_same_ground():
    """勝ち負けを言う前に、母集団・群・指標が同じであることを確かめる。

    土俵が違えば「上回った」は何も意味しない(HC-064)。
    """
    model_report, baseline_report = _reports()
    primary = model_report["systems"]["cnn_co2"]

    assert primary["n_examples"] == baseline_report["n_examples"]
    assert primary["n_groups"] == baseline_report["n_groups"]
    assert primary["n_sites"] == baseline_report["n_sites"]
    assert primary["metric"] == "mae_per_site"
    assert model_report["opponent"]["system"] == baseline_report["best_two_feature_system"]
    assert model_report["opponent"]["mae"] == pytest.approx(
        baseline_report["systems"][baseline_report["best_two_feature_system"]]["mae"]
    )


# ---------------------------------------------------------------- T-023 (G-04)

def test_t023_training_never_saw_the_held_out_group():
    """各 fold で学習に実際に使われた添字が、test 群と重ならない。

    宣言でなく**発火の記録**で見る(HC-071)。学習器が記録を残していなければ落とす。
    """
    model_report, _ = _reports()
    audit = model_report["fold_audit"]
    assert audit, "fold ごとの記録が無い —— 何も検査していない"

    for fold in audit:
        assert fold["train_groups"], "学習側の群が空"
        assert fold["test_groups"], "評価側の群が空"
        assert not set(fold["train_groups"]) & set(fold["test_groups"]), (
            f"群が跨っている: {fold['held_out']}"
        )
        assert fold["train_examples_seen"] > 0, "学習が一例も見ていない"
        assert fold["test_examples_scored"] > 0, "評価が一例も採点していない"


def test_t023_every_example_is_scored_exactly_once():
    """全例がちょうど一度ずつ評価側に回る(取りこぼしも二重計上も無い)。"""
    model_report, baseline_report = _reports()
    scored = sum(f["test_examples_scored"] for f in model_report["fold_audit"])
    assert scored == baseline_report["n_examples"], (
        f"採点された例が {scored} 件、母集団は {baseline_report['n_examples']} 件"
    )


# ---------------------------------------------------------------- T-033 (G-08 / G-09 / G-10)

def test_t033_verdicts_are_rederived_from_the_reports():
    """SPEC §7.9 の判定を、報告から**もう一度導いて**突き合わせる。

    判定は表に書いた瞬間に「決まったこと」に見えるが、表は誰も守らない(HC-152)。
    ここで導き直すことで、数が動いたのに判定が据え置かれる事故を止める。
    """
    model_report, baseline_report = _reports()
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")

    primary = model_report["systems"]["cnn_co2"]["mae"]
    threshold = model_report["p01_threshold_mae"]
    opponent = model_report["opponent"]["mae"]
    co2_only = model_report["systems"]["cnn_co2_paired"]["mae"]
    with_ch4 = model_report["systems"]["cnn_co2_ch4_paired"]["mae"]

    verdicts = {
        "G-08": primary <= threshold,
        "G-09": primary < opponent,
        "G-10": with_ch4 < co2_only,
    }
    expected = {"G-08": True, "G-09": False, "G-10": True}
    assert verdicts == expected, f"報告から導いた判定が SPEC と食い違う: {verdicts}"

    for gate, passed in verdicts.items():
        word = "**通過**" if passed else "**不通過**"
        assert f"| {gate} |" in spec, f"{gate} が SPEC §7.9 の表に無い"
        row = next(l for l in spec.splitlines() if l.startswith(f"| {gate} |") and "|" in l[6:] and word in l)
        assert row, f"{gate}: SPEC の判定が {word} になっていない"


def test_t033_opponent_is_the_stronger_two_feature_system():
    """相手が「2 特徴の系のうち強いほう」であること。弱いほうを選んで勝たない(HC-064)。"""
    model_report, baseline_report = _reports()
    two_feature = {
        k: v["mae"]
        for k, v in baseline_report["systems"].items()
        if k in ("harmonic_linear", "harmonic_knn")
    }
    assert model_report["opponent"]["system"] == min(two_feature, key=two_feature.get)


# ---------------------------------------------------------------- T-030 (G-09)

def test_t030_truncation_does_not_drive_the_conclusion():
    """予算を 3 倍にしても成績が閾値未満しか動かないこと。

    打ち切りは模型の成績を**下振れさせる**ので、「模型が 2 特徴に負けた」という
    結論に対しては都合の良い側に働く。閾値は測定前に決めてある(HC-218)。
    """
    model_report, _ = _reports()
    sensitivity = model_report["budget_sensitivity"]

    assert len(sensitivity["budgets"]) == 2, "予算を 2 通りで測っていない"
    assert abs(sensitivity["delta_mae"]) < sensitivity["margin"], (
        f"予算 3 倍で {sensitivity['delta_mae']:+.2f} 度動いた"
        f"(閾値 {sensitivity['margin']}) —— 打ち切りが結論を左右している"
    )
    assert sensitivity["truncation_drives_conclusion"] is False


def test_t030_budget_sticking_is_reported_not_hidden():
    """予算に張り付いた fold の件数が、隠されずに報告に出ていること。"""
    model_report, _ = _reports()
    for name, system in model_report["systems"].items():
        assert "folds_at_budget" in system, f"{name}: 張り付き件数が報告に無い"
        assert system["folds_at_budget"] <= system["folds_total"]


# ---------------------------------------------------------------- T-031 (G-10)

def test_t031_ch4_comparison_shares_one_ground():
    """CH4 の有無を比べる二系統が、同じ母集団・群・指標で測られていること。

    土俵が違えば「足したら下がった」は何も意味しない(HC-064)。
    """
    model_report, _ = _reports()
    a = model_report["systems"]["cnn_co2_paired"]
    b = model_report["systems"]["cnn_co2_ch4_paired"]

    for key in ("n_examples", "n_sites", "n_groups", "metric"):
        assert a[key] == b[key], f"{key} が食い違う: {a[key]} 対 {b[key]}"
    assert a["channels"] == 1 and b["channels"] == 2


def test_t031_paired_baseline_exists_on_the_same_ground():
    """対照母集団のベースラインが、同じ母集団で測られて存在すること。

    これが無いと、対照母集団の模型の成績を**主母集団のベースライン**と
    並べて読んでしまう。
    """
    paired = ROOT / "data" / "baseline_paired.json"
    if not paired.exists():
        pytest.skip("data/baseline_paired.json が無い")
    report = json.loads(paired.read_text(encoding="utf-8"))
    model_report, _ = _reports()

    assert report["n_examples"] == model_report["systems"]["cnn_co2_paired"]["n_examples"]
    assert report["n_groups"] == model_report["systems"]["cnn_co2_paired"]["n_groups"]


# ---------------------------------------------------------------- T-032 (G-14)

def test_t032_report_is_written_before_the_long_computation():
    """報告が `status` を持つこと。書き出し経路を計算の前に通した証拠である。"""
    model_report, _ = _reports()
    assert model_report.get("status") in ("running", "complete")


def test_t032_positive_control_numpy_types_are_converted():
    """陽性対照: numpy の型を含む辞書が `_jsonable` を通れば JSON にできること。

    通らない実装だと、書き出しは**計算がすべて終わってから**落ちる
    (loop_002 GEN-LOGIC で 14 走ぶんを失った)。
    """
    import numpy as np

    from pipeline import experiment

    raw = {"a": np.float64(1.5), "b": np.bool_(True), "c": np.int64(3), "d": [np.float32(0.5)]}
    with pytest.raises(TypeError):
        json.dumps(raw)
    assert json.loads(json.dumps(experiment._jsonable(raw))) == {
        "a": 1.5,
        "b": True,
        "c": 3,
        "d": [0.5],
    }


def test_t023_folds_cover_every_group():
    """すべての群が一度ずつ抜かれる。"""
    model_report, baseline_report = _reports()
    held = [f["held_out"] for f in model_report["fold_audit"]]
    assert len(held) == len(set(held)), "同じ群が二度抜かれている"
    assert len(held) == baseline_report["n_groups"]
