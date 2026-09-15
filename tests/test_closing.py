"""未測定だった P-04 / P-05 の測定の部品(TEST_SPEC T-065..T-068)。"""
import re
from pathlib import Path

import pytest

from pipeline import closing_experiment as ce
from pipeline import curves

ROOT = Path(__file__).resolve().parents[1]


def _example(code, lat, year=2000, group=None, values=None):
    return curves.Example(
        code=code, year=year, group=group or code, latitude=lat,
        values=tuple(values if values is not None else [0.0] * curves.N_BINS),
        n_samples=24, filled_bins=curves.N_BINS, interpolated=0,
    )


# ---------------------------------------------------------------- T-065

def test_t065_bands_are_exclusive_and_boundaries_go_to_extratropics():
    assert ce.band_of(23.44) == "北の中高緯度"
    assert ce.band_of(23.43) == "熱帯"
    assert ce.band_of(0.0) == "熱帯"
    assert ce.band_of(-23.43) == "熱帯"
    assert ce.band_of(-23.44) == "南の中高緯度"
    assert ce.band_of(82.45) == "北の中高緯度"
    assert ce.band_of(-89.98) == "南の中高緯度"


def test_t065_site_error_averages_over_seeds_and_years():
    examples = [_example("a", 50.0, 2000), _example("a", 50.0, 2001), _example("b", -60.0)]
    labels = [50.0, 50.0, -60.0]
    seed0 = [52.0, 46.0, -50.0]   # a: 2, 4 / b: 10
    seed1 = [50.0, 54.0, -70.0]   # a: 0, 4 / b: 10
    errors = ce.site_errors(examples, [seed0, seed1], labels)
    assert errors["a"] == (50.0, pytest.approx(2.5))
    assert errors["b"] == (-60.0, pytest.approx(10.0))


def test_t065_mismatched_lengths_are_refused():
    with pytest.raises(AssertionError):
        ce.site_errors([_example("a", 10.0)], [[1.0, 2.0]], [10.0])


# ---------------------------------------------------------------- T-066

def _errors(north, tropics, south):
    out = {}
    for i, e in enumerate(north):
        out[f"n{i}"] = (50.0 + i, e)
    for i, e in enumerate(tropics):
        out[f"t{i}"] = (float(i), e)
    for i, e in enumerate(south):
        out[f"s{i}"] = (-50.0 - i, e)
    return out


def test_t066_positive_control_clear_separation_gives_small_p():
    errors = _errors([1.0] * 10, [10.0] * 8, [12.0] * 8)
    result = ce.permutation_test(errors, repeats=2000)
    assert result["statistic"] == pytest.approx(9.0)
    assert result["p"] < 0.01


def test_t066_negative_control_equal_errors_give_p_one():
    errors = _errors([5.0] * 10, [5.0] * 8, [5.0] * 8)
    result = ce.permutation_test(errors, repeats=500)
    assert result["statistic"] == pytest.approx(0.0)
    assert result["p"] == pytest.approx(1.0)


def test_t066_is_deterministic_for_a_seed():
    errors = _errors([1.0, 4.0, 2.0, 3.0], [5.0, 1.0, 6.0], [2.0, 7.0, 3.0])
    a = ce.permutation_test(errors, repeats=300, seed=1)
    b = ce.permutation_test(errors, repeats=300, seed=1)
    assert a == b


# ---------------------------------------------------------------- T-067

def test_t067_monthly_pairs_keep_order_identity_and_swap_only_values():
    events = [
        _example("brw", 71.3, 2001, group="g1", values=[1.0] * 24),
        _example("brw", 71.3, 2002, group="g1", values=[2.0] * 24),
        _example("mlo", 19.5, 2001, group="g2", values=[3.0] * 24),
    ]
    month = {"brw": [(2002, [9.0] * 24)], "mlo": [(2001, [8.0] * 24), (1999, [7.0] * 24)]}
    ev, mo = ce.monthly_pairs(events, month)
    assert [(e.code, e.year) for e in ev] == [("brw", 2002), ("mlo", 2001)]
    assert [(e.code, e.year) for e in mo] == [("brw", 2002), ("mlo", 2001)]
    for a, b in zip(ev, mo):
        assert (a.code, a.year, a.group, a.latitude) == (b.code, b.year, b.group, b.latitude)
        assert a.values != b.values
    assert mo[0].values == tuple([9.0] * 24)
    # event 側の例は書き換わらない
    assert events[1].values == tuple([2.0] * 24)


def test_t067_negative_control_no_month_record_means_no_pair():
    events = [_example("spo", -90.0, 2010)]
    ev, mo = ce.monthly_pairs(events, {"spo": [(2011, [0.0] * 24)]})
    assert ev == [] and mo == []


# ---------------------------------------------------------------- T-068 (G-06)

#: 出荷する模型を作る経路。月次(平滑済み)を読んではいけない。
SHIPPING_PATH = ("experiment.py", "export_model.py", "reference64.py", "phase_experiment.py", "nested_experiment.py", "model.py", "curves.py")
MONTHLY = re.compile(r"noaa_monthly_curves|month_path|_month\.txt|months_to_bins")


def monthly_references(source: str) -> list:
    return MONTHLY.findall(source)


def test_t068_shipping_path_never_reads_monthly_files():
    found = {
        name: monthly_references((ROOT / "pipeline" / name).read_text(encoding="utf-8"))
        for name in SHIPPING_PATH
    }
    assert {k: v for k, v in found.items() if v} == {}


def test_t068_only_closing_experiment_trains_on_monthly():
    trainers = [
        p.name for p in (ROOT / "pipeline").glob("*.py")
        if "run_logo" in p.read_text(encoding="utf-8") and monthly_references(p.read_text(encoding="utf-8"))
    ]
    assert trainers == ["closing_experiment.py"]


def test_t068_positive_control_scanner_catches_a_monthly_import():
    assert monthly_references("from pipeline.external import noaa_monthly_curves")
    assert monthly_references("path = ingest.month_path(gas, code)")
    assert not monthly_references("curves.build_examples('co2')")
