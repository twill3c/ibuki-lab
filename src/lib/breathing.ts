/**
 * 画面①が使う計算。**描画から切り離してあるので単体で検査できる。**
 *
 * 図に添える文字(軸・凡例・注記)も、図を生成したのと同じ源から導く(HC-045)。
 * 座標を決め打ちしない。
 */

export type Curve = {
  code: string;
  latitude: number;
  values: number[];
  amplitude: number;
  years: number;
  trough_bin: number;
  peak_bin: number;
};

export type Site = Curve & { name: string; year_min: number; year_max: number };
export type Band = Curve & { parent: string; samples_per_bin_min: number; short_record: boolean };

export type Screen1 = {
  n_bins: number;
  bin_labels: string[];
  unit: string;
  sites: Site[];
  bands: Band[];
  coverage: Record<string, number>;
  provenance: { smoothed: boolean; note: string; site_curves: string; band_curves: string };
};

/**
 * 色の頭打ち。**外れ値に尺度を決めさせない。**
 *
 * 実測 2026-09-08: 地点の振幅は中央値 11.6 / p90 20.5 ppm だが、記録 1 年の
 * 大陸内の地点(Shangdianzi 55.3 ppm)が最大を握る。尺度をそこに合わせると
 * 大半の地点が無色になるので、頭打ちを置いて**超えた地点には印を付ける**。
 */
export const COLOUR_CAP_PPM = 8;

/** 色を決める。青が吸う(負)、赤が吐く(正)。0 は無色。 */
export function colourFor(value: number, cap: number = COLOUR_CAP_PPM): string {
  const t = Math.max(-1, Math.min(1, value / cap));
  const magnitude = Math.abs(t);
  // 明度で強さを出す。色相は二色だけにして、色覚の差で読めなくならないようにする。
  const light = 100 - 45 * magnitude;
  const hue = t < 0 ? 205 : 15;
  const sat = 12 + 68 * magnitude;
  return `hsl(${hue} ${sat.toFixed(0)}% ${light.toFixed(0)}%)`;
}

/** 帯が色の頭打ちを超えているか(印を付ける対象)。 */
export function exceedsCap(curve: Curve, cap: number = COLOUR_CAP_PPM): boolean {
  return curve.values.some((v) => Math.abs(v) > cap);
}

/** 緯度を 0..1 の縦位置へ。北を上にする。 */
export function latitudeToUnit(latitude: number): number {
  return (90 - latitude) / 180;
}

/** ビン番号を「何月ごろか」に直す。ラベルはこの関数からしか作らない。 */
export function binToMonthLabel(bin: number, nBins: number): string {
  const month = Math.floor((bin / nBins) * 12) + 1;
  const half = (bin / nBins) * 12 - Math.floor((bin / nBins) * 12) < 0.5 ? "前半" : "後半";
  return `${month}月${half}`;
}

/**
 * 緯度帯ごとに地点をまとめた行を作る。
 *
 * 地点をそのまま縦に並べると、北緯 40〜50 度に密集して南半球が潰れる。
 * **等幅の緯度帯に均してから並べる**ことで、緯度と振幅の対応が読めるようになる。
 * 空の帯は空のまま残す —— 詰めると緯度の目盛りが嘘になる。
 */
export type BandRow = {
  from: number;
  to: number;
  values: number[] | null;
  codes: string[];
  /** その行だけの振幅(山と谷の差)。正規化表示と注記に使う。 */
  amplitude: number;
};

export function latitudeBands(sites: Site[], step = 10): BandRow[] {
  const rows: BandRow[] = [];
  for (let top = 90; top > -90; top -= step) {
    const from = top;
    const to = top - step;
    const inside = sites.filter((s) => s.latitude <= from && s.latitude > to);
    if (inside.length === 0) {
      rows.push({ from, to, values: null, codes: [], amplitude: 0 });
      continue;
    }
    const nBins = (inside[0] as Site).values.length;
    const values = Array.from({ length: nBins }, (_, i) =>
      inside.reduce((sum, s) => sum + (s.values[i] as number), 0) / inside.length,
    );
    rows.push({
      from,
      to,
      values,
      codes: inside.map((s) => s.code),
      amplitude: Math.max(...values) - Math.min(...values),
    });
  }
  return rows;
}

/**
 * 行を自分の振幅で割る。**量は消えるが形が残る。**
 *
 * 絶対の目盛りでは南半球がほぼ無色になる(実測 2026-09-08: 南緯 10 度以南の行は
 * |値| が最大 1.17 ppm で、頭打ち 8 ppm に対して濃度 15% 未満)。それは事実だが、
 * 図としては「呼吸が浅い」ではなく「データが無い」と読める。
 *
 * **二つは用途が違う。** 絶対は量を、正規化は位相の反転を見せる。
 * どちらか一方に寄せず、切り替えで両方出して、何を見ているかを画面に書く。
 */
export function normaliseRow(values: number[]): number[] {
  const half = (Math.max(...values) - Math.min(...values)) / 2;
  if (half <= 0) return values.map(() => 0);
  const mid = (Math.max(...values) + Math.min(...values)) / 2;
  return values.map((v) => (v - mid) / half);
}

/**
 * 二つの曲線の位相差をビン単位で返す(環なので折り返す)。
 *
 * 「南では逆向き」を数で言うために使う。半年ずれていれば nBins/2 になる。
 */
export function phaseGap(a: number, b: number, nBins: number): number {
  const raw = Math.abs(a - b) % nBins;
  return Math.min(raw, nBins - raw);
}

/** 折れ線を SVG のパスにする。y は値、x はビン。**環なので端をつなぐ。** */
export function ringPath(
  values: number[],
  width: number,
  height: number,
  cap: number,
): string {
  const n = values.length;
  const points = values.map((v, i) => {
    const x = ((i + 0.5) / n) * width;
    const y = height / 2 - (Math.max(-cap, Math.min(cap, v)) / cap) * (height / 2);
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  return `M${points.join("L")}`;
}
