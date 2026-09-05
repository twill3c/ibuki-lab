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


def _referenced_gate_ids() -> set:
    out = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        out |= set(re.findall(r"G-\d+", path.read_text(encoding="utf-8")))
    return out


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
