/**
 * 顕著度と曲線操作に対する検査(TEST_SPEC T-052..T-055)。
 *
 * 中心差分は近似なので、**自動微分と照合して近似が妥当な範囲にあることを示す**。
 * 照合せずに絵を塗ると、絵が何を表しているか誰にも言えない。
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import type { Weights } from "../model/forward";
import {
  EPSILON_PPM,
  amplitudeOf,
  gradientTimesInput,
  inputGradient,
  latitudeToUnit,
  morph,
  saliencyCap,
  saliencyColour,
} from "./saliency";
import { forward, forwardWithActivations } from "../model/forward";

const weights = JSON.parse(
  readFileSync(new URL("../../data/model_weights.json", import.meta.url), "utf8"),
) as Weights;

const reference = JSON.parse(
  readFileSync(new URL("../../data/reference_forward.json", import.meta.url), "utf8"),
) as {
  inputs: number[][];
  sample_indices: number[];
  input_gradients_float64: number[][];
};

// ---------------------------------------------------------------- T-052

describe("顕著度の照合", () => {
  /**
   * この前向きは **max(|h2|) を含むので滑らかでない**。最大を取る位置が
   * 差分の刻みの中で入れ替わると、中心差分と自動微分は**別のもの**を測る。
   * 一致を要求する前に「一致しうる点かどうか」を先に分ける(HC-232 / HC-073)。
   */
  function argmaxSignature(curve: readonly number[]): string {
    const { h2 } = forwardWithActivations(weights, curve);
    const channels = (h2[0] as number[]).length;
    const out: number[] = [];
    for (let c = 0; c < channels; c += 1) {
      let best = -1;
      let bestValue = -Infinity;
      for (let t = 0; t < h2.length; t += 1) {
        const magnitude = Math.abs((h2[t] as number[])[c] as number);
        if (magnitude > bestValue) {
          bestValue = magnitude;
          best = t;
        }
      }
      out.push(best);
    }
    return out.join(",");
  }

  it("最大を取る位置が動かない点では、中心差分が自動微分と一致する", () => {
    let worstRelative = 0;
    let checked = 0;
    let skippedKinks = 0;

    reference.sample_indices.forEach((index, position) => {
      const curve = reference.inputs[index] as number[];
      const ours = inputGradient(weights, curve);
      const theirs = reference.input_gradients_float64[position] as number[];
      expect(ours.length).toBe(theirs.length);

      const scale = Math.max(...theirs.map((v) => Math.abs(v)), 1e-12);
      const base = argmaxSignature(curve);

      for (let i = 0; i < ours.length; i += 1) {
        const plus = [...curve];
        const minus = [...curve];
        plus[i] = (curve[i] as number) + EPSILON_PPM;
        minus[i] = (curve[i] as number) - EPSILON_PPM;
        if (argmaxSignature(plus) !== base || argmaxSignature(minus) !== base) {
          skippedKinks += 1;
          continue;
        }
        const relative = Math.abs((ours[i] as number) - (theirs[i] as number)) / scale;
        worstRelative = Math.max(worstRelative, relative);
        checked += 1;
      }
    });

    // 走査対象が空でないこと、そして**折れ点が実在すること**の両方を記録に残す。
    // 折れ点が 0 件なら、この選り分けは何もしていない(HC-070)。
    expect(checked).toBeGreaterThan(400);
    expect(skippedKinks).toBeGreaterThan(0);
    expect(worstRelative).toBeLessThan(1e-4);
  });

  it("陽性対照: 刻みを桁で変えると照合が緩む(近似であることの裏づけ)", () => {
    const curve = reference.inputs[reference.sample_indices[0] as number] as number[];
    const theirs = reference.input_gradients_float64[0] as number[];
    const scale = Math.max(...theirs.map((v) => Math.abs(v)), 1e-12);

    // 刻みを 1000 倍にすると曲率を拾って差が広がる。広がらなければ、
    // この照合は刻みに依らない何かを見ている
    const coarse = coarseGradient(curve, EPSILON_PPM * 1000);
    const worst = Math.max(
      ...coarse.map((v, i) => Math.abs(v - (theirs[i] as number)) / scale),
    );
    expect(worst).toBeGreaterThan(1e-4);
  });
});

function coarseGradient(curve: readonly number[], eps: number): number[] {
  const work = [...curve];
  return curve.map((_v, i) => {
    const original = work[i] as number;
    work[i] = original + eps;
    const plus = forward(weights, work);
    work[i] = original - eps;
    const minus = forward(weights, work);
    work[i] = original;
    return (plus - minus) / (2 * eps);
  });
}

// ---------------------------------------------------------------- T-053

describe("勾配 × 入力", () => {
  it("値が 0 の点は寄与も 0 になる", () => {
    const curve = new Array(24).fill(0) as number[];
    for (const v of gradientTimesInput(weights, curve)) {
      expect(Math.abs(v)).toBeLessThan(1e-9);
    }
  });

  it("実データでは寄与が一様でない(絵にする値がある)", () => {
    const curve = reference.inputs[reference.sample_indices[0] as number] as number[];
    const contributions = gradientTimesInput(weights, curve);
    const cap = saliencyCap(contributions);
    expect(cap).toBeGreaterThan(1e-3);
    expect(new Set(contributions.map((v) => v.toFixed(4))).size).toBeGreaterThan(10);
  });
});

// ---------------------------------------------------------------- T-054

describe("曲線をいじる", () => {
  const curve = reference.inputs[reference.sample_indices[12] as number] as number[];

  it("振幅を倍にすると振幅が倍になり、平均は動かない", () => {
    const doubled = morph(curve, 2, 0);
    expect(amplitudeOf(doubled)).toBeCloseTo(amplitudeOf(curve) * 2, 6);
    expect(mean(doubled)).toBeCloseTo(mean(curve), 6);
  });

  it("位相を回しても値の集合は変わらない(環として回す)", () => {
    const rotated = morph(curve, 1, 6);
    expect([...rotated].sort()).toEqual([...curve].sort());
    expect(rotated[6]).toBeCloseTo(curve[0] as number, 9);
    // 一周回すと元に戻る
    expect(morph(curve, 1, 24).map((v) => v.toFixed(9))).toEqual(curve.map((v) => v.toFixed(9)));
  });

  it("**振幅を変えると模型の答えが動く** —— L5 の結論を触って確かめられる", () => {
    const small = forward(weights, morph(curve, 0.4, 0));
    const large = forward(weights, morph(curve, 2.0, 0));
    expect(Math.abs(large - small)).toBeGreaterThan(5);
  });

  it("**位相を回しても答えが動かない** —— 模型は位相を原理的に見ていない", () => {
    // 巡回畳み込みはずらしに同変、pooling(平均と絶対値の最大)はずらしに不変。
    // したがって前向き全体がずらしに不変になる。**これが G-09 の負けの機構である**
    // (相手の 2 特徴は振幅と位相の両方を持つ)。loop_006 GEN-LOGIC / SPEC §7.15。
    const base = forward(weights, curve);
    for (const shift of [1, 6, 12, 18]) {
      const shifted = forward(weights, morph(curve, 1, shift));
      expect(Math.abs(shifted - base)).toBeLessThan(1e-9);
    }
  });

  it("陰性対照: 位相でなく形を崩せば答えは動く(不変性が『何も効かない』ではない)", () => {
    const base = forward(weights, curve);
    const reversed = [...curve].reverse();
    expect(Math.abs(forward(weights, reversed) - base)).toBeGreaterThan(0.5);
  });
});

// ---------------------------------------------------------------- T-055

describe("表示の補助", () => {
  it("緯度が縦位置へ単調に写り、範囲外を丸める", () => {
    expect(latitudeToUnit(90)).toBeCloseTo(0, 9);
    expect(latitudeToUnit(-90)).toBeCloseTo(1, 9);
    expect(latitudeToUnit(0)).toBeCloseTo(0.5, 9);
    expect(latitudeToUnit(200)).toBeCloseTo(0, 9);
    expect(latitudeToUnit(-200)).toBeCloseTo(1, 9);
  });

  it("色が符号で変わり、0 で無色に寄る", () => {
    expect(saliencyColour(-1, 1)).toContain("hsl(205");
    expect(saliencyColour(1, 1)).toContain("hsl(15");
    expect(saliencyColour(0, 1)).toContain("96%");
  });

  it("頭打ちが 0 除算にならない", () => {
    expect(Number.isFinite(saliencyCap([0, 0, 0]))).toBe(true);
    expect(saliencyColour(0, 0)).toContain("hsl(");
  });
});

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}
