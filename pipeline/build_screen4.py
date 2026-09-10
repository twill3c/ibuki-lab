"""画面④「解剖台」のデータを書き出す(L7)。

**通らなかったゲートと、測って捨てた記録を消さずに出す**(F-07)。
成績表だけを出すと、うまくいった部分しか読めない —— このプロジェクトで一番
価値のある記録は、外れた予測と、負けの機構が分かった経緯のほうである。

出典はすべて既存の正本を読むだけで、ここで新しく測らない。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "data" / "screen4.json"


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


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
        "verification": {
            "two_implementation": "float64 どうし 1e-9・出荷経路は無次元化した 1e-6",
            "browser_checks": 19,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"系 {len(systems)} / ゲート {len(gates)} / 捨てた記録 {len(discarded)} / 予測 {len(predictions)}")
    print(f"{OUT.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
