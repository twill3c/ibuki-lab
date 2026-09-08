/**
 * 年周曲線から緯度を当てる 1D CNN の前向き。**ランタイム依存ゼロ**。
 *
 * これは `pipeline/model.py` の `forward_with_activations` の直訳である。
 * 学習側の畳み込みを「巡回の取り出し + 行列積」で書いてあるので、ここも同じ形に書ける
 * (畳み込みライブラリを持ち込まずに済むのは、そのおかげである)。
 *
 * **式を変えたくなったら、必ず両方を同時に変えること。** 片方だけ直すと、
 * 二実装照合(G-01)はそれを捕まえるが、捕まえるのはテストであって利用者ではない。
 */

/** 入力の点数(半月ビン)。`pipeline/curves.N_BINS` と一致していなければならない。 */
export const N_POINTS = 24;

/** 畳み込みの幅。`pipeline/model.KERNEL`。 */
export const KERNEL = 5;

/** 入力をこの値(ppm)で割ってから第一層に入れる。`pipeline/model.INPUT_SCALE`。 */
export const INPUT_SCALE = 10;

/** 出力にこの値を掛けて度にする。`pipeline/model.LATITUDE_SCALE`。 */
export const LATITUDE_SCALE = 90;

export type Weights = {
  /** (K, Cin, Cout) */
  conv1_w: number[][][];
  conv1_b: number[];
  conv2_w: number[][][];
  conv2_b: number[];
  /** (2*Cout, Dense) */
  dense1_w: number[][];
  dense1_b: number[];
  dense2_w: number[][];
  dense2_b: number[];
};

export type Activations = {
  /** (L, Cout) */
  h1: number[][];
  h2: number[][];
  pooled: number[];
  d1: number[];
  output: number;
};

/**
 * 巡回畳み込み。`x` は (L, Cin)、`w` は (K, Cin, Cout)。
 *
 * 一年は環なので端を巡回で埋める。0 埋めにすると「年の切れ目」という
 * 実在しない構造を模型に教えることになる。
 */
export function circularConv1d(
  x: readonly (readonly number[])[],
  w: readonly (readonly (readonly number[])[])[],
  b: readonly number[],
): number[][] {
  const length = x.length;
  const kernel = w.length;
  const pad = (kernel - 1) >> 1;
  const inChannels = (w[0] as readonly (readonly number[])[]).length;
  const outChannels = (
    (w[0] as readonly (readonly number[])[])[0] as readonly number[]
  ).length;

  const out: number[][] = [];
  for (let i = 0; i < length; i += 1) {
    const row = new Array<number>(outChannels);
    for (let o = 0; o < outChannels; o += 1) row[o] = b[o] as number;

    for (let k = 0; k < kernel; k += 1) {
      // Python 側は roll(x, pad - k) を並べる。位置 i が読むのは x[(i + k - pad) mod L]。
      const source = (((i + k - pad) % length) + length) % length;
      const xs = x[source] as readonly number[];
      const wk = w[k] as readonly (readonly number[])[];
      for (let c = 0; c < inChannels; c += 1) {
        const value = xs[c] as number;
        if (value === 0) continue;
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

export function tanhAll(rows: readonly (readonly number[])[]): number[][] {
  return rows.map((row) => row.map((v) => Math.tanh(v)));
}

/**
 * 位置に依らない要約。平均だけだと「どれだけ振れたか」が消えるので、
 * **絶対値の**最大を並べる。どちらも巡回のずらしに対して不変である。
 */
export function poolMeanAndMaxAbs(h: readonly (readonly number[])[]): number[] {
  const channels = (h[0] as readonly number[]).length;
  const means = new Array<number>(channels).fill(0);
  const maxAbs = new Array<number>(channels).fill(0);

  for (const row of h) {
    for (let c = 0; c < channels; c += 1) {
      const value = row[c] as number;
      means[c] = (means[c] as number) + value;
      const magnitude = Math.abs(value);
      if (magnitude > (maxAbs[c] as number)) maxAbs[c] = magnitude;
    }
  }
  for (let c = 0; c < channels; c += 1) means[c] = (means[c] as number) / h.length;
  return [...means, ...maxAbs];
}

export function dense(
  x: readonly number[],
  w: readonly (readonly number[])[],
  b: readonly number[],
): number[] {
  const out = b.map((v) => v);
  for (let i = 0; i < x.length; i += 1) {
    const value = x[i] as number;
    if (value === 0) continue;
    const wi = w[i] as readonly number[];
    for (let o = 0; o < out.length; o += 1) {
      out[o] = (out[o] as number) + value * (wi[o] as number);
    }
  }
  return out;
}

/** 一本の曲線(長さ 24、ppm の偏差)を受けて、緯度(度)と途中の活性を返す。 */
export function forwardWithActivations(
  weights: Weights,
  curve: readonly number[],
  second?: readonly number[],
): Activations {
  if (curve.length !== N_POINTS) {
    throw new Error(`入力の長さが ${curve.length} で、${N_POINTS} でない`);
  }

  const scaled: number[][] = curve.map((v, i) =>
    second ? [v / INPUT_SCALE, (second[i] as number) / INPUT_SCALE] : [v / INPUT_SCALE],
  );

  const h1 = tanhAll(circularConv1d(scaled, weights.conv1_w, weights.conv1_b));
  const h2 = tanhAll(circularConv1d(h1, weights.conv2_w, weights.conv2_b));
  const pooled = poolMeanAndMaxAbs(h2);
  const d1 = dense(pooled, weights.dense1_w, weights.dense1_b).map((v) => Math.tanh(v));
  const output = (dense(d1, weights.dense2_w, weights.dense2_b)[0] as number) * LATITUDE_SCALE;

  return { h1, h2, pooled, d1, output };
}

export function forward(weights: Weights, curve: readonly number[]): number {
  return forwardWithActivations(weights, curve).output;
}
