"""文書と実装の突合(TEST_SPEC T-007 / T-009 / T-010 / T-011)。

テストは実装を守るが、SPEC・README に書いた数を守るものは何も無い(HC-152)。
この節がその役目を負う。
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "SPEC.md"
CENSUS = ROOT / "data" / "census.json"

# SPEC §2 の素材表の行見出しと census のキーの対応。
# 表を増やしたらここも増やす —— 増やし忘れは T-009 の網羅検査が落とす。
SPEC_TABLE_KEYS = {
    "SRC-CO2 月次ファイル": "co2_month_files",
    "うち固定地点(event ヘッダに緯度あり)": "fixed_sites",
    "うち船舶帯(公称緯度なし・§2.1)": "ship_band_sites",
    "SRC-CO2 event 標本(全 event ファイル・棄却フラグ除去後)": "co2_event_samples_retained",
    "CO2 と CH4 の両方がある地点": "sites_with_both_gases",
}

# README にも同じ数を書いている。文書ごとに見出しが違うので対応表も分ける(HC-152)。
README = ROOT / "README.md"
README_TABLE_KEYS = {
    "NOAA GML 地上フラスコ網 CO2 月次ファイル": "co2_month_files",
    "うち固定地点(緯度がヘッダにある)": "fixed_sites",
    "うち船舶帯(緯度を観測から導いた)": "ship_band_sites",
    "CO2 event 標本(棄却フラグ除去後)": "co2_event_samples_retained",
}


def _table_numbers(path: Path, keys) -> dict:
    """マークダウンの 2 列表から「見出し → 数値の文字列」を読む。"""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2:
            continue
        label, value = cells
        if label in keys:
            out[label] = value
    return out


def _assert_table_matches_census(path: Path, keys: dict) -> None:
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    table = _table_numbers(path, keys)

    missing = set(keys) - set(table)
    assert not missing, f"{path.name} の表から行が消えている: {sorted(missing)}"

    for label, key in keys.items():
        written = table[label].replace(",", "")
        assert written.isdigit(), f"{path.name} / {label}: 数値が入っていない({table[label]})"
        assert int(written) == census[key], (
            f"{path.name} / {label}: 文書={written} census={census[key]}"
        )


# ---------------------------------------------------------------- T-007 (G-07)

def test_t007_text_hygiene_reports_no_violation():
    """字種検査が違反 0。

    パイプに通すと $? が別物になる(AGENTS.md 既知の罠)ので、直接 returncode を見る。
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "harness" / "text_hygiene.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_t007_positive_control_hygiene_detects_injected_char():
    """陽性対照: 検査器が実際に撃つこと(HC-041)。

    混入させる文字はソースに生で置かず、符号位置から組み立てる。
    U+0435 はキリル小文字イェーで、ラテン小文字 e と字形が区別できない。
    """
    sys.path.insert(0, str(ROOT))
    from harness import text_hygiene  # noqa: E402

    assert not text_hygiene.self_test(), "検査器の自己対照が落ちている"

    injected = chr(0x0435)
    assert text_hygiene.scan_line("これは" + injected + "です"), (
        "検査器が既知の混入を捕まえられない —— この検査は緑でも何も見ていない"
    )
    assert not text_hygiene.scan_line("これは正常な日本語の行です"), (
        "正常な行で誤検出している(HC-074)"
    )


# ---------------------------------------------------------------- T-009 (G-14)

def test_t009_spec_numbers_match_census():
    """SPEC §2 の素材表が census.json と一致する。"""
    _assert_table_matches_census(SPEC, SPEC_TABLE_KEYS)


def test_t009_readme_numbers_match_census():
    """README の素材表も同じ census と一致する。

    同じ数を二つの文書に書いているので、両方を突き合わせないと片方だけが嘘になる(HC-152)。
    """
    _assert_table_matches_census(README, README_TABLE_KEYS)


def test_t009_positive_control_shifted_number_is_caught():
    """陽性対照: 表の数を 1 ずらしたら突合が撃つこと(HC-041)。"""
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    table = _table_numbers(SPEC, SPEC_TABLE_KEYS)
    label = "SRC-CO2 月次ファイル"
    key = SPEC_TABLE_KEYS[label]

    assert int(table[label].replace(",", "")) == census[key], "前提が崩れている"

    tampered = ROOT / "data" / "census.json"
    original = json.loads(tampered.read_text(encoding="utf-8"))
    shifted = dict(original)
    shifted[key] = original[key] + 1
    try:
        tampered.write_text(json.dumps(shifted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(AssertionError):
            _assert_table_matches_census(SPEC, SPEC_TABLE_KEYS)
    finally:
        tampered.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- T-020 (G-14)

BASELINE = ROOT / "data" / "baseline.json"

# SPEC §7.3 の系の名前 → baseline.json のキー。行を増やしたらここも増やす。
SPEC_SYSTEM_ROWS = {
    "平均予測器": "mean_predictor",
    "中央値予測器": "median_predictor",
    "2 特徴・線形": "harmonic_linear",
    "**2 特徴・kNN**": "harmonic_knn",
    "24 次元・kNN": "curve_knn",
}


def _spec_system_table() -> dict:
    """SPEC §7.3 の 4 列表から「系の名前 → (MAE地点, MAE例, 最悪)」を読む。"""
    out = {}
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] in SPEC_SYSTEM_ROWS:
            out[cells[0]] = cells[1:]
    return out


def test_t020_spec_baseline_table_matches_report():
    """SPEC §7.3 の測定表が data/baseline.json と一致する(小数第 2 位まで)。

    SPEC の保証粒度は表に書いた桁数までであり、それ以上を要求しない(HC-016)。
    """
    report = json.loads(BASELINE.read_text(encoding="utf-8"))
    table = _spec_system_table()

    missing = set(SPEC_SYSTEM_ROWS) - set(table)
    assert not missing, f"SPEC §7.3 の表から行が消えている: {sorted(missing)}"

    for label, key in SPEC_SYSTEM_ROWS.items():
        written = [w.replace("**", "") for w in table[label]]
        measured = report["systems"][key]
        for value, field in zip(written, ("mae", "mae_per_example", "worst_site_ae")):
            assert float(value) == round(measured[field], 2), (
                f"§7.3 {label} / {field}: SPEC={value} baseline={measured[field]:.2f}"
            )


def test_t020_spec_control_and_threshold_match_report():
    """陰性対照の値と P-01 の閾値も文書と一致する。"""
    report = json.loads(BASELINE.read_text(encoding="utf-8"))
    spec = SPEC.read_text(encoding="utf-8")

    assert f"{report['controls']['label_shuffle_mae']:.2f} 度" in spec, "陰性対照の値が §7.3 と食い違う"
    assert f"閾値 {report['p01_threshold_mae']:.2f} 度" in spec, "P-01 の閾値が §7.4 と食い違う"
    assert str(report["n_examples"]) in spec and str(report["n_groups"]) in spec


def test_t020_positive_control_shifted_baseline_is_caught():
    """陽性対照: baseline.json の値をずらすと突合が撃つこと(HC-041)。"""
    original = json.loads(BASELINE.read_text(encoding="utf-8"))
    tampered = json.loads(BASELINE.read_text(encoding="utf-8"))
    tampered["systems"]["harmonic_knn"]["mae"] += 1.0
    try:
        BASELINE.write_text(json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(AssertionError):
            test_t020_spec_baseline_table_matches_report()
    finally:
        BASELINE.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- T-027 (G-14)

MODEL = ROOT / "data" / "model.json"

# SPEC §7.8 の系の名前 → model.json のキー。
SPEC_MODEL_ROWS = {
    "**1D CNN(CO2 のみ)**": "cnn_co2",
    "1D CNN(CO2 のみ・対照母集団)": "cnn_co2_paired",
    "1D CNN(CO2 + CH4)": "cnn_co2_ch4_paired",
}


def _spec_model_table() -> dict:
    out = {}
    for line in SPEC.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] in SPEC_MODEL_ROWS:
            out[cells[0]] = cells[1:]
    return out


def _require_model():
    if not MODEL.exists():
        pytest.skip("data/model.json が無い(.venv/Scripts/python.exe -m pipeline.experiment で作る)")
    return json.loads(MODEL.read_text(encoding="utf-8"))


def test_t027_spec_model_table_matches_report():
    """SPEC §7.8 の測定表が data/model.json と一致する(小数第 2 位まで)。"""
    report = _require_model()
    table = _spec_model_table()

    missing = set(SPEC_MODEL_ROWS) - set(table)
    assert not missing, f"SPEC §7.8 の表から行が消えている: {sorted(missing)}"

    for label, key in SPEC_MODEL_ROWS.items():
        written = [w.replace("**", "") for w in table[label]]
        measured = report["systems"][key]
        assert float(written[0]) == round(measured["mae"], 2), (
            f"§7.8 {label} / mae: SPEC={written[0]} model={measured['mae']:.2f}"
        )
        assert int(written[1].replace(",", "")) == measured["n_examples"], (
            f"§7.8 {label} / n_examples が食い違う"
        )
        assert int(written[2]) == measured["params"], f"§7.8 {label} / params が食い違う"


def test_t027_positive_control_shifted_model_number_is_caught():
    """陽性対照: model.json の値をずらすと突合が撃つこと(HC-041)。"""
    original = _require_model()
    tampered = json.loads(MODEL.read_text(encoding="utf-8"))
    tampered["systems"]["cnn_co2"]["mae"] += 1.0
    try:
        MODEL.write_text(json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(AssertionError):
            test_t027_spec_model_table_matches_report()
    finally:
        MODEL.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_t027_budget_sticking_is_small_and_disclosed():
    """予算に張り付いた fold が少数にとどまり、件数が文書に出ていること。

    **この検査は「張り付き 0 件」を要求しない。** 種が変われば内側分割も変わるので
    0 件は追い込めず、追い込む設計にすると探索そのものが止まる(HC-218)。
    打ち切りが結論を動かすかどうかは T-030(予算感度)が答える。
    ここで守るのは「隠していないこと」と「少数であること」だけである。
    """
    report = _require_model()
    spec = SPEC.read_text(encoding="utf-8")

    for name, system in report["systems"].items():
        share = system["folds_at_budget"] / system["folds_total"]
        assert share < 0.05, (
            f"{name}: {system['folds_at_budget']}/{system['folds_total']} が予算に張り付いた"
        )

    primary = report["systems"]["cnn_co2"]
    assert f"{primary['folds_total']} 走中 **{primary['folds_at_budget']} 件**" in spec, (
        "主系統の予算張り付き件数が SPEC §7.10 に書かれていない"
    )


# ---------------------------------------------------------------- T-010 (G-14)

def _parse_gates(spec_text: str) -> dict:
    gates = {}
    for line in spec_text.splitlines():
        m = re.match(r"\|\s*(G-\d+)\s*\|.*\|\s*(.*?)\s*\|\s*$", line)
        if m:
            gates[m.group(1)] = m.group(2)
    return gates


def _orphan_gates(gates: dict, referenced: set) -> list:
    return [
        gid
        for gid, verdict in gates.items()
        if gid not in referenced and "未実装" not in verdict
    ]


#: ゲートを守っている検査の在り処。**Python のテストだけではない** ——
#: 図の収まりや操作の到達は実ブラウザ検品が、前向きの照合は TS の検査が守る。
#: ここを狭く取ると、実際には守られているゲートが「参照なし」として落ちる。
CHECK_SOURCES = (
    ("tests", "test_*.py"),
    ("src", "**/*.test.ts"),
    ("scripts", "*.mjs"),
)


def _referenced_gate_ids() -> set:
    out = set()
    for folder, pattern in CHECK_SOURCES:
        for path in (ROOT / folder).glob(pattern):
            out |= set(re.findall(r"G-\d+", path.read_text(encoding="utf-8")))
    return out


def test_t010_browser_gates_are_declared_in_the_checker():
    """実ブラウザ検品が守るゲートが、検品器の対応表に**書かれている**こと。

    走査で `G-13` の文字が見つかるだけでは、コメントに書いただけかもしれない。
    検品器の GATES 表に現れることを要求すると、ケースとゲートの対応が実体になる(HC-157)。
    """
    checker = (ROOT / "scripts" / "browser_check.mjs").read_text(encoding="utf-8")
    table = re.search(r"const GATES = \{(.*?)\n\};", checker, re.S)
    assert table, "検品器に GATES の対応表が無い"

    declared = set(re.findall(r'"(G-\d+|[FN]-\d+)"', table.group(1)))
    for gate in ("G-13", "G-06", "G-16"):
        assert gate in declared, f"{gate} が検品器の対応表に無い"

    cases = set(re.findall(r'"(B-\d+c?)":', table.group(1)))
    reported = set(re.findall(r'report\("(B-\d+c?)"', checker))
    assert cases == reported, f"対応表とケースが食い違う: 表のみ {cases - reported} / 実行のみ {reported - cases}"


def test_t010_every_gate_is_referenced_or_declared_unimplemented():
    """SPEC の全 G-xx が、テストから参照されるか「未実装」と明記されている(HC-157)。"""
    gates = _parse_gates(SPEC.read_text(encoding="utf-8"))
    assert gates, "SPEC から品質ゲート表が読めない"

    referenced = _referenced_gate_ids()
    assert referenced, "テストから G-xx の参照が 1 件も読めない —— 走査が空振りしている"

    orphans = _orphan_gates(gates, referenced)
    assert not orphans, f"参照も未実装宣言も無いゲート: {orphans}"


def test_t010_positive_control_orphan_gate_is_caught():
    """陽性対照: 参照も未実装宣言も無いゲートを置いたら、検査が撃つこと(HC-041)。

    あわせて陰性対照 —— 正当な二形(参照あり / 未実装明記)は撃たないことを確かめる。
    """
    synthetic = "\n".join(
        [
            "| G-90 | 参照もされず未実装とも書かれていないゲート | 通過 |",
            "| G-91 | テストから参照されているゲート | 通過 |",
            "| G-92 | まだ書いていないゲート | 未実装(L9) |",
        ]
    )
    gates = _parse_gates(synthetic)
    assert set(gates) == {"G-90", "G-91", "G-92"}, "合成した表が読めていない"

    assert _orphan_gates(gates, {"G-91"}) == ["G-90"], "検査が撃つべきものを撃っていない"


# ---------------------------------------------------------------- T-011 (G-14)

def test_t011_recorded_hashes_match_files():
    """data/LICENSE-DATA.md に記録した SHA-256 が実ファイルと一致する。"""
    doc = (ROOT / "data" / "LICENSE-DATA.md").read_text(encoding="utf-8")
    pairs = re.findall(r"`([^`]+\.tar\.gz)`\s*\|\s*`([0-9a-f]{64})`", doc)
    assert pairs, "LICENSE-DATA.md にハッシュの記録が無い"

    for name, recorded in pairs:
        path = ROOT / "data" / "raw" / name
        if not path.exists():
            pytest.skip(f"{name} が手元に無い(配布物には含めない)")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == recorded, f"{name}: 記録 {recorded} 実体 {actual}"
