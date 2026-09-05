"""L0 の取り込みに対する検査(TEST_SPEC T-001..T-008)。

期待値は定数で書かず、実データ全域から導ける不変量で書く(HC-016)。
検出系のケースには陽性対照を対で置く(HC-041)。
"""
import math

import pytest

from pipeline import ingest


# ---------------------------------------------------------------- T-001 (G-14)

def test_t001_site_set_matches_month_files():
    """取り込んだ地点集合が、tar 内の *_month.txt から導ける集合と厳密に一致する。

    期待値の出所: 実データの全域走査。件数ではなく集合の一致で書くので、
    データが更新されても壊れない(HC-016)。
    """
    from_files = ingest.discover_month_site_codes("co2")
    from_sites = {s.code for s in ingest.build_sites("co2")}

    assert from_files, "月次ファイルが 1 件も見つからない(走査対象が空)"
    assert from_sites == from_files, (
        f"取りこぼし {sorted(from_files - from_sites)} / 余り {sorted(from_sites - from_files)}"
    )


# ---------------------------------------------------------------- T-002 (G-02)

def test_t002_fixed_site_latitude_comes_from_header():
    """固定地点の緯度は event ヘッダの site_latitude そのものである。

    期待値の出所: 外部権威(NOAA GML の自己記述ヘッダ)。
    """
    fixed = [s for s in ingest.build_sites("co2") if s.lat_source == "header"]
    assert fixed, "ヘッダ由来の緯度を持つ地点が 0 件(走査対象が空)"

    for s in fixed:
        header = ingest.read_event_header(ingest.event_path("co2", s.code))
        assert s.latitude == pytest.approx(float(header["site_latitude"]))
        assert -90.0 <= s.latitude <= 90.0, f"{s.code}: 緯度が範囲外 {s.latitude}"


# ---------------------------------------------------------------- T-003 (G-05)

def test_t003_fill_value_latitude_is_never_adopted():
    """公称緯度が欠測値の系統を固定地点として採らない。

    陽性対照(HC-070): 「欠測値を持つ系統が実データ中に現に存在すること」を先に表明する。
    存在しなければこの対照は何も検査していないので、その場合は落とす。
    """
    fills = ingest.event_headers_with_fill_latitude("co2")
    assert fills, "公称緯度が欠測値の系統が 1 件も無い —— この対照は何も検査していない"

    for s in ingest.build_sites("co2"):
        if s.latitude is not None:
            assert abs(s.latitude) <= 90.0, f"{s.code}: 欠測値が緯度として採られた {s.latitude}"
        assert s.lat_source != "header" or s.code not in fills


# ---------------------------------------------------------------- T-004 (G-05)

def test_t004_ship_band_latitude_comes_from_observed_marks():
    """船舶帯の緯度は、符号名を見ずにデータから導いた目標点である。

    期待値の出所: 実データの緯度分布(`observed_marks`)。符号名との一致は
    「独立に決まった二つの量が合った」という主張であり、恒等式ではない。
    """
    bands = [s for s in ingest.build_sites("co2") if s.lat_source == "band"]
    assert bands, "船舶帯が 0 件(走査対象が空)"

    for s in bands:
        marks = ingest.observed_marks("co2", s.parent)
        assert s.latitude in marks, f"{s.code}: 緯度 {s.latitude} が観測された目標点に無い"
        nominal = ingest.band_nominal_latitude(s.code)
        assert s.latitude == pytest.approx(nominal, abs=ingest.BAND_NAME_CHECK_DEG), (
            f"{s.code}: 観測された目標点 {s.latitude} が符号名 {nominal} と食い違う"
        )


def test_t004_check_is_not_an_identity():
    """この照合が恒等式でないことの証拠(HC-045 / HC-071)。

    データから導いた目標点の集合が、符号名から導いた集合と**一致しない**ことを示す。
    一致していたら「名前で絞ってから名前と比べている」疑いが残る。

    実測 2026-09-05: POC の観測目標点は 35N / 40N / 45N を含むが、これらには月次
    ファイルが無い(README §7.7 —— 記録が疎な系統は月次平均が作られない)。
    """
    band_nominals = {
        ingest.band_nominal_latitude(s.code)
        for s in ingest.build_sites("co2")
        if s.lat_source == "band" and s.parent == "poc"
    }
    marks = set(ingest.observed_marks("co2", "poc"))

    assert band_nominals <= marks, f"符号名にあって観測に無い目標点: {sorted(band_nominals - marks)}"
    extra = marks - band_nominals
    assert extra, (
        "観測された目標点の集合が符号名の集合と完全に一致した —— "
        "この照合は独立性の証拠を残していない"
    )


def test_t004_positive_control_band_check_rejects_shifted_marks():
    """陽性対照: 目標点をずらして与えると例外で止まること(HC-041)。"""
    marks = ingest.observed_marks("co2", "poc")
    assert ingest.derive_band_latitude("pocn30", marks) == pytest.approx(30.0)

    shifted = tuple(m + 12.0 for m in marks)
    with pytest.raises(ingest.IngestError):
        ingest.derive_band_latitude("pocn30", shifted)


def test_t004_band_spacing_is_derived_from_data():
    """半値幅は観測された目標点の刻みから導く。README の POC ±2.5 と一致する。

    README が明示しているのは POC の ±2.5 のみである。SCS へ 2.5 を持ち込まない。
    """
    assert ingest.band_half_width("poc") == pytest.approx(2.5), "README §7.7 の POC 帯定義と一致しない"
    assert ingest.band_half_width("scs") == pytest.approx(1.5), "SCS の刻みは 3 度のはず"


# ---------------------------------------------------------------- T-005 (G-06)

def test_t005_rejected_samples_are_dropped_and_target_is_not_empty():
    """QC 棄却フラグ(第 1 列が '.' 以外)の標本が除去される。

    期待値の出所: README §7.5 のフラグ定義。
    「除去された件数 > 0」を別表明で確かめないと、除去器が働いていなくても緑になる。
    """
    counts = ingest.qc_counts("co2")
    assert counts["total"] > 0, "event 標本が 0 件(走査対象が空)"
    assert counts["rejected"] > 0, "棄却フラグの標本が 0 件 —— 除去器が働いたかどうか区別できない"
    assert counts["retained"] == counts["total"] - counts["rejected"]


# ---------------------------------------------------------------- T-006 (G-06)

def test_t006_training_input_is_event_only():
    """学習入力経路に渡るのは event 由来の系列だけである(SPEC §2.2 / G-06)。"""
    series = ingest.training_series("co2")
    assert series, "学習系列が 0 件(走査対象が空)"
    for s in series:
        assert s.source == "event"
        assert s.smoothed is False


def test_t006_positive_control_smoothed_series_is_refused():
    """陽性対照: 平滑済み(month 由来)の系列を学習経路へ渡すと落ちること。"""
    smoothed = ingest.Series(code="brw", source="month", smoothed=True, values=[])
    with pytest.raises(ingest.IngestError):
        ingest.assert_trainable([smoothed])


# ---------------------------------------------------------------- T-008 (G-14)

def test_t008_census_counts_agree_with_independent_path():
    """census の数が、tar 一覧と行数を数える別経路の値と一致する(二経路一致)。

    経路も比べる(HC-065): 片方はファイル名から、もう片方はヘッダ解析から数える。
    """
    census = ingest.build_census()
    independent = ingest.count_by_filename_path()

    assert census["co2_month_files"] == independent["co2_month_files"]
    assert census["ch4_month_files"] == independent["ch4_month_files"]
    assert census["fixed_sites"] + census["ship_band_sites"] + census["sites_without_latitude"] == (
        census["co2_month_files"]
    ), "地点の三分類が月次ファイル数を尽くしていない"
    assert not math.isnan(float(census["co2_event_samples_retained"]))
