"""画面④「解剖台」のデータを書き出す(L7)。

**通らなかったゲートと、測って捨てた記録を消さずに出す**(F-07)。
成績表だけを出すと、うまくいった部分しか読めない —— このプロジェクトで一番
価値のある記録は、外れた予測と、負けの機構が分かった経緯のほうである。

出典はすべて既存の正本を読むだけで、ここで新しく測らない。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "data" / "screen4.json"


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


CH4_CONTROL_LABELS = {
    "cnn_ch4_rotated_paired": "CH4 を巡回でずらした(CO2 との位相差を壊す)",
    "cnn_ch4_equalised_paired": "CH4 の振幅を揃えた(振幅の大小を消す)",
}


def browser_check_count() -> int:
    """実ブラウザ検品の件数は、検品器の GATES 表から数える。

    手で書いた 19 が、検品を 23 件に増やした後も古いまま画面に出ていた(L8 で発見)。
    """
    source = (ROOT / "scripts" / "browser_check.mjs").read_text(encoding="utf-8")
    return len(re.findall(r'^\s*"B-\d+[a-z]?":', source, flags=re.M))


def load_nested() -> dict | None:
    """L8 の測定。**走り終わっていない報告は出さない**(途中の数を判定として見せない)。"""
    path = ROOT / "data" / "nested.json"
    if not path.exists():
        return None
    nested = json.loads(path.read_text(encoding="utf-8"))
    return nested if nested.get("status") == "complete" else None


def nested_block(nested: dict) -> dict:
    harmonic = nested["nested"]["harmonic"]
    invariant = nested["nested"]["invariant"]
    v = nested["verdicts"]
    opponent = nested["opponent_mae"]
    gap = opponent - harmonic["mae"]
    spread = harmonic["mae_spread"]

    lines = []
    if v["p07_beats_beyond_spread"]:
        lines.append(f"P-07: 位相ありは相手の {opponent:.2f} 度を {gap:.2f} 度下回り、差が種の幅 {spread:.2f} 度を超えた。")
    elif v["p07_numerically"]:
        lines.append(
            f"P-07: 位相ありは相手の {opponent:.2f} 度を {gap:.2f} 度下回ったが、"
            f"種の幅 {spread:.2f} 度より小さい差なので「上回った」とは言わない。"
        )
    else:
        lines.append(f"P-07: 位相ありは {harmonic['mae']:.2f} 度で、相手の {opponent:.2f} 度を下回らなかった。")
    lines.append(
        f"P-08: 位相ありの種の幅は L7 の {nested['l7']['harmonic_spread']:.2f} 度から {spread:.2f} 度へ"
        + ("縮んだ。" if v["p08"] else "縮まなかった。")
    )
    reversed_seeds = [r["seed"] for r in harmonic["seeds"] if r["mae_stop_rule"] < r["mae"]]
    lines.append(
        f"P-09: 同じ重みの上で、選ぶ分割で選ぶと {harmonic['mae']:.2f} 度、"
        f"止め時の分割で選ぶと {harmonic['mae_stop_rule']:.2f} 度。"
        + ("平均では分けたほうが良かった。" if v["p09"] else "分けても良くならなかった。")
        + (f"ただし種 {'・'.join(str(s) for s in reversed_seeds)} では逆だった。" if reversed_seeds else "")
    )
    diff = invariant["mae"] - harmonic["mae"]
    lines.append(
        f"P-11: nested でも位相あり {harmonic['mae']:.2f} 度 対 位相なし {invariant['mae']:.2f} 度。"
        + ("位相の効果は向きが保たれた。" if v["p11"] else "位相の効果は残らなかった。")
        + (
            f"ただし差 {diff:.2f} 度は位相なしの種の幅 {invariant['mae_spread']:.2f} 度より小さく、大きさは確かでない。"
            if v["p11"] and diff < invariant["mae_spread"] else ""
        )
    )

    def row(entry: dict) -> dict:
        return {
            "mae": entry["mae"],
            "spread": entry["mae_spread"],
            "stop_rule_mae": entry["mae_stop_rule"],
            "seeds": entry["mae_per_seed"],
        }

    return {
        "train_fraction": 1 - nested["stop_fraction"] - nested["select_fraction"],
        "harmonic": row(harmonic),
        "invariant": row(invariant),
        "summary": lines,
        "verdicts": v,
    }


def ch4_components_block(nested: dict) -> dict:
    d = nested["ch4_decomposition"]
    rows = []
    for key, label in CH4_CONTROL_LABELS.items():
        c = d["controls"][key]
        sides = c["closer_to_real_per_seed"]
        if all(sides):
            side = "3 種とも本物寄り"
        elif not any(sides):
            side = "3 種とも基準寄り"
        else:
            side = f"揃わない(本物寄り {sum(sides)}/3)"
        rows.append({
            "label": label, "mae": c["mae"], "spread": c["spread"], "retained": c["retained"],
            "seeds_agree": c["all_seeds_agree"], "side": side,
        })
    r_eq = d["controls"]["cnn_ch4_equalised_paired"]["retained"]
    r_rot = d["controls"]["cnn_ch4_rotated_paired"]["retained"]
    tail = f"(振幅を揃えた r={r_eq:.2f}・巡回でずらした r={r_rot:.2f})"
    if d["p10_identified"]:
        summary = "P-10 は成立。振幅を揃えると効き目が消え、位相差を壊しても残り、3 種とも同じ側に落ちた" + tail + "。"
    elif d["p10_holds_numerically"]:
        summary = "P-10 は数の上では成立。r は予測の側に出たが種ごとの側が揃わず、要素を特定したとは言わない" + tail + "。"
    else:
        controls = d["controls"]
        direction = "予測と逆向きで、" if r_rot < r_eq else ""
        summary = (
            f"P-10 は外れた。{direction}振幅を揃えても r={r_eq:.2f}、CO2 との位相差を壊すと r={r_rot:.2f} だった。"
            + (
                "種ごとの側が揃わない対照があるので、何が効き目を運んでいるかは特定しない。"
                if not all(c["all_seeds_agree"] for c in controls.values())
                else ""
            )
        )
    return {
        "rows": rows,
        "summary": summary,
        "p10_holds_numerically": d["p10_holds_numerically"],
        "p10_identified": d["p10_identified"],
    }


def main() -> int:
    baseline = load("baseline.json")
    paired_baseline = load("baseline_paired.json")
    model_report = load("model.json")
    phase = load("phase.json")
    external = load("external.json")
    amplitude = load("amplitude.json")

    opponent = baseline["best_two_feature_system"]
    opponent_mae = baseline["systems"][opponent]["mae"]

    systems = [
        {"name": "常に中央値を答える(自明)", "mae": baseline["systems"]["median_predictor"]["mae"], "kind": "baseline"},
        {"name": "2 特徴・線形", "mae": baseline["systems"]["harmonic_linear"]["mae"], "kind": "baseline"},
        {"name": "2 特徴・kNN(相手)", "mae": opponent_mae, "kind": "opponent"},
        {"name": "曲線 24 点そのまま・kNN", "mae": baseline["systems"]["curve_knn"]["mae"], "kind": "baseline"},
        {"name": "1D CNN(位相なし)", "mae": model_report["systems"]["cnn_co2"]["mae"],
         "spread": model_report["systems"]["cnn_co2"]["mae_spread"], "kind": "model"},
        {"name": "1D CNN(位相あり)", "mae": phase["systems"]["cnn_harmonic"]["mae"],
         "spread": phase["systems"]["cnn_harmonic"]["mae_spread"], "kind": "model"},
        {"name": "ラベル入替(陰性対照)", "mae": baseline["controls"]["label_shuffle_mae"], "kind": "control"},
    ]

    gates = [
        {"id": "G-03", "title": "陰性対照", "verdict": "通過",
         "detail": f"地点単位で緯度を入れ替えると MAE {baseline['controls']['label_shuffle_mae']:.2f} 度。"
                   f"自明な中央値予測器の {baseline['systems']['median_predictor']['mae']:.2f} 度すら下回らない"},
        {"id": "G-08", "title": "目玉① 未知の地点の緯度を当てる", "verdict": "通過",
         "detail": f"閾値 {model_report['p01_threshold_mae']:.2f} 度に対し {model_report['systems']['cnn_co2']['mae']:.2f} 度。"
                   "ただし 2 特徴 kNN が既に満たす低い障壁だった"},
        {"id": "G-09", "title": "目玉② 2 特徴の回帰を上回る", "verdict": "不通過",
         "detail": f"位相なしの模型で {model_report['systems']['cnn_co2']['mae']:.2f} 度、相手は {opponent_mae:.2f} 度。"
                   "**負けの機構は後から分かった** —— 模型が位相を原理的に見ていなかった(§7.15)"},
        {"id": "G-10", "title": "目玉③ CH4 を足すと下がる", "verdict": "通過(留保つき)",
         "detail": "同一母集団で 11.69 → 10.76 度。ただし種の幅が 2.63 → 0.40 と激減しており、"
                   "情報が増えたのか学習が安定しただけかを L7 の入替対照で切り分けた"},
        {"id": "G-11", "title": "外部ホールドアウト(気象庁)", "verdict": "通過",
         "detail": f"気象庁 3 地点で {external['systems']['jma_holdout']['mae']:.2f} 度。"
                   f"資料形の効果 {external['decomposition']['product_effect_deg']:+.2f} 度に対し "
                   f"網の効果 {external['decomposition']['network_effect_deg']:+.2f} 度で、劣化はほぼ資料形の側"},
        {"id": "G-12", "title": "目玉④ 高緯度の振幅は増えている", "verdict": "通過",
         "detail": f"北緯 45 度以北の {amplitude['bands']['北緯45度以北']['sites']} 地点すべてで傾きが正。"
                   "採取密度の交絡は主張と同じ向きに効くので、標本数を共変量に入れて解き直しても残ることを確かめた"},
    ]

    discarded = [
        {"title": "船舶帯の検算が循環していた",
         "loop": "L0",
         "detail": "標本を符号名の緯度で絞り込んでから、その中央値を符号名と比べていた。"
                   "絞り込みが結論を保証しており原理的に落ちない照合だった。21 帯すべてで差が 0.000 度に"
                   "なったことで気づき、緯度分布のみから目標点を導く方式に置き換えた"},
        {"title": "第一層が初期時点で 51.3% 飽和していた",
         "loop": "L2",
         "detail": "生の ppm を入れると tanh が飽和し、**この課題で信号そのものである振幅**を"
                   "第一層が捨てていた。検査は全部緑で成績も数字としては出るので、内側を測るまで気づけない"},
        {"title": "模型は位相を原理的に見ていなかった",
         "loop": "L6",
         "detail": "曲線を半年ずらしても答えが 1.5e-14 度しか動かない。巡回畳み込みはずらしに同変で、"
                   "要約(平均と絶対値の最大)がずらしに不変なので前向き全体が位相を見ない。"
                   "**これが G-09 の負けの機構**で、相手の 2 特徴は振幅と位相の両方を持っていた"},
        {"title": "同じ緯度の二地点で答えが 14.6 度割れた",
         "loop": "L5",
         "detail": "南鳥島(24.28)に 15.9 度、与那国島(24.47)に 30.5 度。経度は 31 度離れ、"
                   "片方は太平洋の孤島、片方は大陸のすぐ手前。模型が読んでいるのは緯度でなく年周振幅で、"
                   "振幅は大陸への近さで決まる"},
        {"title": "内側検証で学習率を選ぶ規則が、試験と逆順に並んだ",
         "loop": "L7",
         "detail": "位相ありの掃引で内側検証 9.00/8.24/7.69 に対し試験 11.10/11.99/12.47。"
                   "内側検証は止め時を選ぶのにも使っているので偏った推定量になっており、"
                   "学習率という別の軸の比較には使えない。規則は事前に決めたので動かさず、限界を記録した"},
    ]

    predictions = [
        {"id": "P-01", "text": "年周曲線だけで未知地点の緯度を当てられる", "verdict": "当たった"},
        {"id": "P-02", "text": "ニューラルネットは 2 特徴の回帰を上回らない", "verdict": "当たった"},
        {"id": "P-03", "text": "CH4 を足しても緯度誤差は下がらない", "verdict": "外れた"},
        {"id": "P-06", "text": "高緯度の年周振幅は記録期間中に増大している", "verdict": "成立"},
        {"id": "P-07", "text": "位相を見える要約に替えると 2 特徴 kNN を下回る", "verdict": "後述"},
    ]

    closing_path = ROOT / "data" / "closing.json"
    closing = json.loads(closing_path.read_text(encoding="utf-8")) if closing_path.exists() else None
    if closing is not None and closing.get("status") == "complete":
        p04, p05 = closing["p04"], closing["p05"]
        predictions += [
            {"id": "P-04", "text": "誤差は南半球と熱帯で大きい",
             "verdict": "成立" if p04["sayable"] else "数の上では成立" if p04["holds_numerically"] else "外れた"},
            {"id": "P-05", "text": "平滑済みの月次で学習すると、2 特徴回帰との差はさらに縮む",
             "verdict": "成立" if p05["sayable"] else "数の上では成立" if p05["holds_numerically"] else "外れた"},
        ]
        predictions.sort(key=lambda p: p["id"])

    nested = load_nested()
    if nested is not None:
        v = nested["verdicts"]
        h = nested["nested"]["harmonic"]
        if not v["p07_numerically"]:
            discarded.append({
                "title": "L7 の「相手と並んだ」は、偏った選定規則の上の数だった",
                "loop": "L8",
                "detail": f"学習率を fold ごとに、止め時とは別の分割で選び直すと、位相ありは {h['mae']:.2f} 度で"
                          f"相手の {nested['opponent_mae']:.2f} 度を下回らなかった。種の幅は "
                          f"{nested['l7']['harmonic_spread']:.2f} → {h['mae_spread']:.2f} 度に縮んで数は安定したが、"
                          f"L7 の {nested['l7']['harmonic_mae']:.2f} 度は残らなかった",
            })
        for p in predictions:
            if p["id"] == "P-07":
                p["verdict"] = (
                    "上回った(nested)" if v["p07_beats_beyond_spread"]
                    else "数の上では成立(nested でも)" if v["p07_numerically"]
                    else "不成立(nested)"
                )
        predictions += [
            {"id": "P-08", "text": "学習率の選定を nested にすると、位相ありの種の幅が縮む",
             "verdict": "成立" if v["p08"] else "外れた"},
            {"id": "P-09", "text": "止め時と学習率選びの分割を分けると、選んだ学習率の試験誤差が良くなる",
             "verdict": "成立" if v["p09"] else "外れた"},
            {"id": "P-10", "text": "CH4 の効き目は振幅が運んでいる",
             "verdict": "成立" if v["p10_identified"] else "数の上では成立" if v["p10_numerically"] else "外れた"},
            {"id": "P-11", "text": "nested にしても、位相ありは位相なしを下回る",
             "verdict": "成立" if v["p11"] else "外れた"},
        ]

    payload = {
        "generated_from": "pipeline/build_screen4.py",
        "opponent": {"system": opponent, "mae": opponent_mae},
        "population": {
            "examples": baseline["n_examples"],
            "sites": baseline["n_sites"],
            "groups": baseline["n_groups"],
            "paired_examples": paired_baseline["n_examples"],
        },
        "systems": systems,
        "gates": gates,
        "discarded": discarded,
        "predictions": predictions,
        "phase": {
            "invariant_mae": model_report["systems"]["cnn_co2"]["mae"],
            "invariant_spread": model_report["systems"]["cnn_co2"]["mae_spread"],
            "harmonic_mae": phase["systems"]["cnn_harmonic"]["mae"],
            "harmonic_spread": phase["systems"]["cnn_harmonic"]["mae_spread"],
            "harmonic_seeds": phase["systems"]["cnn_harmonic"]["mae_per_seed"],
            "lr_sweep": phase["lr_sweep"]["rows"],
            "chosen_lr": phase["lr_sweep"]["chosen_lr"],
        },
        "ch4": phase.get("ch4_decomposition"),
        "ch4_systems": {
            name: {"mae": phase["systems"][name]["mae"], "spread": phase["systems"][name]["mae_spread"]}
            for name in ("cnn_invariant_paired", "cnn_ch4_paired", "cnn_ch4_shuffled_paired")
            if name in phase["systems"]
        },
        "nested": nested_block(nested) if nested is not None else None,
        "ch4_components": ch4_components_block(nested) if nested is not None else None,
        "verification": {
            "two_implementation": "float64 どうし 1e-9・出荷経路は無次元化した 1e-6",
            "browser_checks": browser_check_count(),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"系 {len(systems)} / ゲート {len(gates)} / 捨てた記録 {len(discarded)} / 予測 {len(predictions)}")
    print(f"{OUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
