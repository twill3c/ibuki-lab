/**
 * 顕著度と、曲線をいじる操作(画面③)。
 *
 * **顕著度の意味をはっきりさせておく。** ここで出すのは
 * 「その半月の値を少し動かすと、模型の答えが何度動くか」であって、
 * 「模型がそこを見ている」でも「その季節が緯度を決めている」でもない。
 * 前者は測れる量、後者は測っていない主張である。
 */
import { forwardWithActivations, N_POINTS, type Weights } from "../model/forward";

/**
 * 中心差分の刻み(ppm)。
 *
 * 小さすぎると float64 の丸めが効き、大きすぎると曲率を拾う。曲線の値は
 * ±20 ppm 程度なので、その 1 万分の 1 を採る。**自動微分との照合(T-052)が
 * この選択の妥当性を測る** —— 合わなければ刻みが悪いか実装が違うかである。
 */
export const EPSILON_PPM = 2e-3;

/** 入力に対する勾配(度 / ppm)を中心差分で出す。 */
export function inputGradient(weights: Weights, curve: readonly number[]): number[] {
  const out: number[] = [];
  const work = [...curve];
  for (let i = 0; i < curve.length; i += 1) {
    const original = work[i] as number;
    work[i] = original + EPSILON_PPM;
    const plus = forwardWithActivations(weights, work).output;
    work[i] = original - EPSILON_PPM;
    const minus = forwardWithActivations(weights, work).output;
    work[i] = original;
    out.push((plus - minus) / (2 * EPSILON_PPM));
  }
  return out;
}

/**
 * 勾配 × 入力。**その点が実際に答えへ寄せている度数**になる。
 *
 * 勾配だけだと「値が 0 の点でも感度が高い」ことがあり、絵にすると
 * 何も起きていない季節が濃く塗られる。掛けると、実際に効いた分だけが残る。
 */
export function gradientTimesInput(weights: Weights, curve: readonly number[]): number[] {
  return inputGradient(weights, curve).map((g, i) => g * (curve[i] as number));
}

/** 曲線の振幅(山と谷の差)。 */
export function amplitudeOf(curve: readonly number[]): number {
  return Math.max(...curve) - Math.min(...curve);
}

/**
 * 振幅を `factor` 倍し、位相を `shiftBins` だけ回す。
 *
 * **これが画面③の主題である。** L5 の外部検証で、模型が読んでいるのは緯度でなく
 * 年周振幅だと分かった(同じ緯度の南鳥島と与那国島で答えが 14.6 度割れた)。
 * 振幅だけを動かして答えが動くなら、それは触って確かめられる。
 *
 * 位相は環として回す —— 一年は環なので、端で切らない。
 */
export function morph(
  curve: readonly number[],
  factor: number,
  shiftBins: number,
): number[] {
  const n = curve.length;
  const mean = curve.reduce((a, b) => a + b, 0) / n;
  const scaled = curve.map((v) => (v - mean) * factor + mean);
  const shift = ((Math.round(shiftBins) % n) + n) % n;
  return Array.from({ length: n }, (_, i) => scaled[(i - shift + n) % n] as number);
}

/** 緯度を縦帯の位置(0..1、北が上)へ。 */
export function latitudeToUnit(latitude: number): number {
  return (90 - Math.max(-90, Math.min(90, latitude))) / 180;
}

export type Choice = {
  code: string;
  label: string;
  group: string;
  latitude: number;
  values: number[];
  /** 学習に使ったか。**気象庁と船舶帯は使っていない。** */
  inTraining: boolean;
  note?: string;
};

/** 顕著度の塗り色。正負で色相を変え、強さで濃さを出す。 */
export function saliencyColour(value: number, cap: number): string {
  const t = Math.max(-1, Math.min(1, value / (cap || 1)));
  const magnitude = Math.abs(t);
  const hue = t < 0 ? 205 : 15;
  return `hsl(${hue} ${(20 + 60 * magnitude).toFixed(0)}% ${(96 - 42 * magnitude).toFixed(0)}%)`;
}

export function saliencyCap(values: readonly number[]): number {
  return Math.max(...values.map((v) => Math.abs(v)), 1e-6);
}

export { N_POINTS };
