/**
 * わざと壊した前向き。**照合の陽性対照にだけ使う**(HC-041 / HC-065)。
 *
 * 二実装照合が緑であることは、照合が働いていることを意味しない。対象に違反が無いときと、
 * 照合そのものが何も見ていないときとで、まったく同じ緑が返る。ここに置いた変異体を
 * 照合が落とせなければ、その照合は何も検査していない。
 *
 * **出荷経路からは呼ばれない。** 画面側のコードがここを import していないことは
 * T-041 が走査で確かめる。
 */
import {
  INPUT_SCALE,
  KERNEL,
  LATITUDE_SCALE,
  N_POINTS,
  dense,
  poolMeanAndMaxAbs,
  tanhAll,
  type Activations,
  type Weights,
} from "./forward.js";

export type VariantKind =
  /** 変異なし(陰性対照)。 */
  | "none"
  /**
   * `max(|h2|)` を `max(h2)` に取り違える。
   *
   * **出力が変わらない例が存在する** —— どの通路でも最大値が正なら二つは一致する。
   * 出力だけを見る照合はその例を見逃す。経路(pooled)を見て初めて捕まる。
   */
  | "max_without_abs"
  /**
   * 端の巡回を 0 埋めに取り違える。
   *
   * 一年が環であるという前提が実装に効いていることの裏返し。
   */
  | "zero_padding";

function convWithPadding(
  x: readonly (readonly number[])[],
  w: readonly (readonly (readonly number[])[])[],
  b: readonly number[],
  zeroPad: boolean,
): number[][] {
  const length = x.length;
  const pad = (KERNEL - 1) >> 1;
  const inChannels = (w[0] as readonly (readonly number[])[]).length;
  const outChannels = (
    (w[0] as readonly (readonly number[])[])[0] as readonly number[]
  ).length;

  const out: number[][] = [];
  for (let i = 0; i < length; i += 1) {
    const row = new Array<number>(outChannels);
    for (let o = 0; o < outChannels; o += 1) row[o] = b[o] as number;

    for (let k = 0; k < KERNEL; k += 1) {
      const raw = i + k - pad;
      let source: number;
      if (zeroPad) {
        if (raw < 0 || raw >= length) continue; // ここが変異点
        source = raw;
      } else {
        source = ((raw % length) + length) % length;
      }
      const xs = x[source] as readonly number[];
      const wk = w[k] as readonly (readonly number[])[];
      for (let c = 0; c < inChannels; c += 1) {
        const value = xs[c] as number;
        const wkc = wk[c] as readonly number[];
        for (let o = 0; o < outChannels; o += 1) {
          row[o] = (row[o] as number) + value * (wkc[o] as number);
        }
      }
    }
    out.push(row);
  }
  return out;
}

function poolMeanAndMaxSigned(h: readonly (readonly number[])[]): number[] {
  const channels = (h[0] as readonly number[]).length;
  const means = new Array<number>(channels).fill(0);
  const maxes = new Array<number>(channels).fill(-Infinity);

  for (const row of h) {
    for (let c = 0; c < channels; c += 1) {
      const value = row[c] as number;
      means[c] = (means[c] as number) + value;
      if (value > (maxes[c] as number)) maxes[c] = value; // ここが変異点(絶対値を取らない)
    }
  }
  for (let c = 0; c < channels; c += 1) means[c] = (means[c] as number) / h.length;
  return [...means, ...maxes];
}

export function forwardVariant(
  weights: Weights,
  curve: readonly number[],
  kind: VariantKind,
): Activations {
  if (curve.length !== N_POINTS) {
    throw new Error(`入力の長さが ${curve.length} で、${N_POINTS} でない`);
  }
  const zeroPad = kind === "zero_padding";
  const scaled: number[][] = curve.map((v) => [v / INPUT_SCALE]);

  const h1 = tanhAll(convWithPadding(scaled, weights.conv1_w, weights.conv1_b, zeroPad));
  const h2 = tanhAll(convWithPadding(h1, weights.conv2_w, weights.conv2_b, zeroPad));
  const pooled =
    kind === "max_without_abs" ? poolMeanAndMaxSigned(h2) : poolMeanAndMaxAbs(h2);
  const d1 = dense(pooled, weights.dense1_w, weights.dense1_b).map((v) => Math.tanh(v));
  const output = (dense(d1, weights.dense2_w, weights.dense2_b)[0] as number) * LATITUDE_SCALE;

  return { h1, h2, pooled, d1, output };
}
