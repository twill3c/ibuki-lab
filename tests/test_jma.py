"""気象庁データと外部検証に対する検査(TEST_SPEC T-046..T-051)。

**気象庁は外部ホールドアウトである。** 学習にも、閾値の調整にも、学習率の選定にも
使っていない。T-048 がそれを走査で確かめる —— 宣言でなく、実際に読んでいないことを見る。
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "SPEC.md"
EXTERNAL = ROOT / "data" / "external.json"

#: 学習経路のモジュール。ここが気象庁を読んでいたら外部検証は成立しない。
TRAINING_MODULES = (
    "pipeline/curves.py",
    "pipeline/experiment.py",
    "pipeline/model.py",
    "pipeline/baseline.py",
    "pipeline/export_model.py",
)


def _report():
    if not EXTERNAL.exists():
        pytest.skip("data/external.json が無い(python -m pipeline.external で作る)")
    return json.loads(EXTERNAL.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- T-046 (G-11)

def test_t046_jma_csv_parses_and_drops_gaps():
    """気象庁 CSV が読め、欠測と註記行が落ちる。"""
    from pipeline import jma

    for code in jma.STATIONS:
        rows = jma.read_monthly(code)
        assert rows, f"{code}: 行が 0 件"
        for (year, month), value in rows.items():
            assert 1980 <= year <= 2030
            assert 1 <= month <= 12
            assert isinstance(value, float)


def test_t046_positive_control_year_with_a_gap_is_dropped():
    """陽性対照: 欠測のある年は採用されない。

    実データに欠測が現に存在することを先に表明する —— 存在しなければ
    この対照は何も検査していない(HC-070)。
    """
    from pipeline import jma

    raw = (ROOT / "data" / "raw" / "jma" / "co2_monthave_ryo.csv").read_bytes().decode("cp932")
    assert "--" in raw, "欠測の印が実データに無い —— この対照は何も検査していない"

    rows = jma.read_monthly("ryo")
    gap_years = {y for (y, _m) in rows} - {y for y, _ in jma.annual_curves("ryo")}
    assert gap_years, "欠測で落ちた年が 0 件 —— 落とす経路が働いたか区別できない"

    for year, _curve in jma.annual_curves("ryo"):
        assert all((year, m) in rows for m in range(1, 13)), f"{year}: 欠けた月がある"


# ---------------------------------------------------------------- T-047 (G-11)

def test_t047_coordinates_agree_between_dms_and_decimal():
    """座標が度分表記と小数の両方で保たれ、互いに一致する。

    書き写しの誤りは、片方だけを持っていると永久に見つからない。
    """
    from pipeline import jma

    for code, station in jma.STATIONS.items():
        degrees, minutes = station["latitude_dms"]
        expected = degrees + minutes / 60.0
        assert abs(station["latitude"] - expected) < 1e-3, (
            f"{code}: 小数 {station['latitude']} が度分 {degrees}°{minutes}' と食い違う"
        )
        assert station["source"].startswith("http"), f"{code}: 出所の URL が無い"


def test_t047_minamitorishima_and_yonaguni_share_a_latitude():
    """南鳥島と与那国島がほぼ同緯度で、経度が大きく離れていること。

    この二つは**内蔵の対照**である。模型が緯度でなく大陸の影響を読んでいれば、
    同じ緯度に対して割れた答えを出す。前提が崩れたらその読み方はできない。
    """
    from pipeline import jma

    mnm = jma.STATIONS["mnm"]
    yon = jma.STATIONS["yon"]
    assert abs(mnm["latitude"] - yon["latitude"]) < 0.5, "同緯度という前提が崩れている"
    assert abs(mnm["longitude"] - yon["longitude"]) > 25, "経度が離れているという前提が崩れている"


# ---------------------------------------------------------------- T-048 (G-11)

def test_t048_training_path_never_reads_jma():
    """学習経路が気象庁を読んでいない。**宣言でなく走査で見る。**"""
    offenders = []
    for name in TRAINING_MODULES:
        text = (ROOT / name).read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+.*\bjma\b", text, re.M):
            offenders.append(name)
        if re.search(r"raw[/\\]jma", text):
            offenders.append(f"{name}(パス参照)")
    assert offenders == [], f"学習経路が気象庁を読んでいる: {offenders}"


def test_t048_positive_control_scan_catches_an_import():
    """陽性対照: 走査の正規表現が実際に import を捕まえること(HC-041)。"""
    assert re.search(r"^\s*(from|import)\s+.*\bjma\b", "from pipeline import jma", re.M)
    assert re.search(r"^\s*(from|import)\s+.*\bjma\b", "import pipeline.jma", re.M)
    assert not re.search(r"^\s*(from|import)\s+.*\bjma\b", "from pipeline import curves", re.M)
    # 走査対象が実在すること
    for name in TRAINING_MODULES:
        assert (ROOT / name).exists(), f"{name} が無い —— 走査が空振りしている"


# ---------------------------------------------------------------- T-049 (G-06)

def test_t049_monthly_to_halfmonth_is_repetition_not_interpolation():
    """月平均 12 点を半月 24 点へ広げる操作が、反復であって補間でないこと。

    補間すればそれは平滑であり、§2.2 で避けたのと同じ問題が入る。
    """
    from pipeline import jma

    months = [float(i) for i in range(1, 13)]
    expanded = jma.months_to_bins(months)

    assert len(expanded) == 24
    for i, value in enumerate(months):
        assert expanded[2 * i] == value
        assert expanded[2 * i + 1] == value
    # 補間なら隣り合う 2 点が異なる値になるはずで、ここでは必ず同じ
    assert len({expanded[2 * i + 1] - expanded[2 * i] for i in range(12)}) == 1


# ---------------------------------------------------------------- T-050 (G-11)

def test_t050_data_product_control_is_present():
    """資料形の対照(NOAA の月次に同じ処理を当てたもの)が報告にあること。

    これが無いと、外れた理由が「網が違う」のか「資料形が違う」のか言えない(HC-064)。
    """
    report = _report()
    assert "noaa_monthly_control" in report["systems"], "資料形の対照が無い"
    assert "noaa_event_reference" in report["systems"], "同じ模型の event 由来の基準が無い"

    control = report["systems"]["noaa_monthly_control"]
    reference = report["systems"]["noaa_event_reference"]
    assert control["n_sites"] == reference["n_sites"], "対照と基準の母集団が違う"
    assert control["source"] == "month" and reference["source"] == "event"


def test_t050_jma_is_reported_separately_from_the_controls():
    """気象庁の結果が対照と混ぜずに出ていること。"""
    report = _report()
    jma_rows = report["systems"]["jma_holdout"]["sites"]
    assert len(jma_rows) == 3
    assert {r["code"] for r in jma_rows} == {"ryo", "mnm", "yon"}
    for row in jma_rows:
        assert row["years"] > 0
        assert "predicted_latitude" in row and "true_latitude" in row


# ---------------------------------------------------------------- T-051 (G-14)

def test_t051_spec_external_table_matches_report():
    """SPEC §7.14 の表が data/external.json と一致する。"""
    report = _report()
    spec = SPEC.read_text(encoding="utf-8")

    for key, label in (
        ("jma_holdout", "気象庁 3 地点"),
        ("noaa_monthly_control", "NOAA・月次(資料形の対照)"),
        ("noaa_event_reference", "NOAA・event(同じ模型の基準)"),
    ):
        mae = report["systems"][key]["mae"]
        assert f"| {label} | {mae:.2f} |" in spec, (
            f"§7.14 の {label} が報告と食い違う(報告は {mae:.2f})"
        )


def test_t051_positive_control_shifted_number_is_caught():
    """陽性対照: 報告の値をずらすと突合が撃つこと(HC-041)。"""
    original = EXTERNAL.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["systems"]["jma_holdout"]["mae"] += 1.0
    try:
        EXTERNAL.write_text(
            json.dumps(tampered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with pytest.raises(AssertionError):
            test_t051_spec_external_table_matches_report()
    finally:
        EXTERNAL.write_text(original, encoding="utf-8")
