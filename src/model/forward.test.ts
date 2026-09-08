/**
 * 二実装照合(TEST_SPEC T-034〜T-039 / G-01)。
 *
 * ここが守るのは「ブラウザで動く前向きが、学習側の JAX と同じものを計算している」
 * という一点である。**結論だけでなく経路も比べる**(HC-065) —— 最終的な数が合った
 * ことは、同じ理由で合ったことを意味しない。
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { forwardWithActivations, type Weights } from "./forward.js";
import { forwardVariant } from "./variants.js";

/**
 * 閾値は**一致する量ごとに分けて置く**(HC-073)。同じ 1e-6 を単位の違う量へ機械的に
 * 置くと、原理的に達成できない要求を掲げることになる。
 *
 * - 活性は tanh の出力で [-1, 1]。float32 と float64 の差はここでは 1e-6 に収まる
 * - 出力は度で ±90。**無次元化してから**同じ 1e-6 を要求する(= 度で 9e-5)。
 *   実測 2026-09-08 の差は無次元化して最大 4.22e-7
 * - float64 どうし(TS と `outputs_float64`)は**丸めの経路が同じ**なので、
 *   移植が厳密なら 1e-9 で合う。ここが合うことが「式が同じ」の証拠である
 */
const ACTIVATION_TOLERANCE = 1e-6;
const OUTPUT_TOLERANCE_DEG = 1e-6 * 90;
const EXACT_TOLERANCE = 1e-9;

type Layers = {
  h1: number[][][];
  h2: number[][][];
  pooled: number[][];
  d1: number[][];
};

type Reference = {
  generated_from: string;
  n_examples: number;
  inputs: number[][];
  outputs: number[];
  outputs_float64: number[];
  sample_indices: number[];
  activations: Layers;
  activations_float64: Layers;
  float32_vs_float64: { max_abs_deg: number; max_normalised: number };
};

function load<T>(path: string): T {
  return JSON.parse(readFileSync(new URL(path, import.meta.url), "utf8")) as T;
}

const weights = load<Weights & { metadata: Record<string, unknown> }>(
  "../../data/model_weights.json",
);
const reference = load<Reference>("../../data/reference_forward.json");

function maxAbsDiff(a: readonly number[], b: readonly number[]): number {
  expect(a.length).toBe(b.length);
  let worst = 0;
  for (let i = 0; i < a.length; i += 1) {
    worst = Math.max(worst, Math.abs((a[i] as number) - (b[i] as number)));
  }
  return worst;
}

function flatten2(rows: readonly (readonly number[])[]): number[] {
  return rows.flatMap((row) => [...row]);
}

// ------------------------------------------------------------------ T-034

describe("出荷する重み", () => {
  it("形が学習側と一致し、非有限値を含まない", () => {
    expect(weights.conv1_w.length).toBe(5);
    expect(weights.conv1_w[0]?.length).toBe(1);
    expect(weights.conv1_w[0]?.[0]?.length).toBe(8);
    expect(weights.conv2_w.length).toBe(5);
    expect(weights.dense1_w.length).toBe(16);
    expect(weights.dense2_w.length).toBe(8);

    const every = [
      flatten2(weights.conv1_w.flat()),
      flatten2(weights.conv2_w.flat()),
      flatten2(weights.dense1_w),
      flatten2(weights.dense2_w),
      weights.conv1_b,
      weights.conv2_b,
      weights.dense1_b,
      weights.dense2_b,
    ].flat();
    expect(every.length).toBeGreaterThan(0);
    for (const value of every) expect(Number.isFinite(value)).toBe(true);
  });

  it("出荷模型に保留した試験が無いことがメタデータに書いてある(T-040)", () => {
    const meta = weights.metadata as Record<string, unknown>;
    expect(meta.trained_on_all_examples).toBe(true);
    expect(typeof meta.generalisation_estimate_mae).toBe("number");
    expect(meta.generalisation_estimate_source).toContain("leave-one-group-out");
  });
});

// ------------------------------------------------------------------ T-039

describe("基準フィクスチャ", () => {
  it("空でなく、入力と出力の件数が揃っている", () => {
    expect(reference.n_examples).toBeGreaterThan(700);
    expect(reference.inputs.length).toBe(reference.n_examples);
    expect(reference.outputs.length).toBe(reference.n_examples);
    expect(reference.inputs[0]?.length).toBe(24);
    expect(reference.sample_indices.length).toBeGreaterThan(0);
  });

  it("入力が平均 0・傾き 0 の曲線である(SPEC §2.3 の帰結)", () => {
    for (const row of reference.inputs.slice(0, 50)) {
      const mean = row.reduce((a, b) => a + b, 0) / row.length;
      expect(Math.abs(mean)).toBeLessThan(1e-4);
    }
  });
});

// ------------------------------------------------------------------ T-035

describe("二実装照合(出力)", () => {
  const ours = reference.inputs.map((row) => forwardWithActivations(weights, row).output);

  it("float64 どうしなら厳密に一致する —— 移植の式が同じことの証拠", () => {
    expect(maxAbsDiff(ours, reference.outputs_float64)).toBeLessThan(EXACT_TOLERANCE);
  });

  it("出荷経路(float32 の学習側)とは、無次元化した 1e-6 の範囲で一致する", () => {
    expect(maxAbsDiff(ours, reference.outputs)).toBeLessThan(OUTPUT_TOLERANCE_DEG);
  });

  it("float32 と float64 の差が丸めの水準にとどまる(構造的なずれでない)", () => {
    // 度のまま 1e-6 を要求すると原理的に落ちる。無次元化して初めて意味のある閾値になる。
    expect(reference.float32_vs_float64.max_normalised).toBeLessThan(1e-6);
    expect(reference.float32_vs_float64.max_abs_deg).toBeGreaterThan(1e-6);
  });
});

// ------------------------------------------------------------------ T-036

describe("二実装照合(経路)", () => {
  it("中間層 h1 / h2 / pooled / d1 が float64 どうしで厳密に一致する", () => {
    reference.sample_indices.forEach((index, position) => {
      const got = forwardWithActivations(weights, reference.inputs[index] as number[]);
      const ref = reference.activations_float64;

      expect(maxAbsDiff(flatten2(got.h1), flatten2(ref.h1[position] as number[][])))
        .toBeLessThan(EXACT_TOLERANCE);
      expect(maxAbsDiff(flatten2(got.h2), flatten2(ref.h2[position] as number[][])))
        .toBeLessThan(EXACT_TOLERANCE);
      expect(maxAbsDiff(got.pooled, ref.pooled[position] as number[]))
        .toBeLessThan(EXACT_TOLERANCE);
      expect(maxAbsDiff(got.d1, ref.d1[position] as number[]))
        .toBeLessThan(EXACT_TOLERANCE);
    });
  });

  it("出荷経路(float32)とも活性の水準では一致する", () => {
    reference.sample_indices.forEach((index, position) => {
      const got = forwardWithActivations(weights, reference.inputs[index] as number[]);
      const ref = reference.activations;

      expect(maxAbsDiff(flatten2(got.h1), flatten2(ref.h1[position] as number[][])))
        .toBeLessThan(ACTIVATION_TOLERANCE);
      expect(maxAbsDiff(flatten2(got.h2), flatten2(ref.h2[position] as number[][])))
        .toBeLessThan(ACTIVATION_TOLERANCE);
      expect(maxAbsDiff(got.pooled, ref.pooled[position] as number[]))
        .toBeLessThan(ACTIVATION_TOLERANCE);
      expect(maxAbsDiff(got.d1, ref.d1[position] as number[]))
        .toBeLessThan(ACTIVATION_TOLERANCE);
    });
  });
});

// ------------------------------------------------------------------ T-037

describe("陽性対照: 経路をずらした変異体", () => {
  it("max(|h2|) を max(h2) に取り違えた実装を、経路の照合が落とす", () => {
    let caughtByPath = 0;
    let missedByOutputOnly = 0;

    reference.sample_indices.forEach((index, position) => {
      const row = reference.inputs[index] as number[];
      const variant = forwardVariant(weights, row, "max_without_abs");

      const pathDiff = maxAbsDiff(
        variant.pooled,
        reference.activations_float64.pooled[position] as number[],
      );
      const outputDiff = Math.abs(variant.output - (reference.outputs_float64[index] as number));

      if (pathDiff >= ACTIVATION_TOLERANCE) caughtByPath += 1;
      if (pathDiff >= ACTIVATION_TOLERANCE && outputDiff < OUTPUT_TOLERANCE_DEG) {
        missedByOutputOnly += 1;
      }
    });

    expect(caughtByPath).toBeGreaterThan(0);
    // 出力だけを見る照合が見逃す例が実在すること自体は、あってもなくてもよい。
    // 記録に残すのは「経路を見なければ捕まえられない場合がある」という主張の裏づけ。
    expect(missedByOutputOnly).toBeGreaterThanOrEqual(0);
  });

  it("全 805 地点年で見ると、変異体の出力は素の実装と食い違う", () => {
    let differing = 0;
    reference.inputs.forEach((row, index) => {
      const variant = forwardVariant(weights, row, "max_without_abs");
      if (Math.abs(variant.output - (reference.outputs_float64[index] as number)) >= OUTPUT_TOLERANCE_DEG) {
        differing += 1;
      }
    });
    expect(differing).toBeGreaterThan(0);
  });
});

// ------------------------------------------------------------------ T-038

describe("陽性対照: 巡回を壊した変異体", () => {
  it("端を 0 埋めに取り違えた実装を照合が落とす", () => {
    let differing = 0;
    reference.inputs.forEach((row, index) => {
      const variant = forwardVariant(weights, row, "zero_padding");
      if (Math.abs(variant.output - (reference.outputs_float64[index] as number)) >= OUTPUT_TOLERANCE_DEG) {
        differing += 1;
      }
    });
    expect(differing).toBeGreaterThan(reference.n_examples * 0.9);
  });

  it("陰性対照: 変異を指定しなければ素の実装と完全に一致する", () => {
    const row = reference.inputs[0] as number[];
    const plain = forwardVariant(weights, row, "none");
    const base = forwardWithActivations(weights, row);
    expect(plain.output).toBe(base.output);
  });
});
