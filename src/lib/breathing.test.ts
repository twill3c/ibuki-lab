/**
 * 画面①の計算に対する検査(TEST_SPEC T-042..T-045)。
 *
 * 描画から切り離した関数だけを見る。図が読めるかは実ブラウザ検品(B-01..B-11)が測る ——
 * **目で見なければ分からない性質をテストで代替しない**(HC-041)。
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  COLOUR_CAP_PPM,
  binToMonthLabel,
  colourFor,
  exceedsCap,
  latitudeBands,
  normaliseRow,
  phaseGap,
  ringPath,
  type Screen1,
} from "./breathing.js";

const data = JSON.parse(
  readFileSync(new URL("../../public/data/screen1.json", import.meta.url), "utf8"),
) as Screen1;

// ---------------------------------------------------------------- T-042

describe("緯度帯へ均す", () => {
  const rows = latitudeBands(data.sites, 10);

  it("18 行を返し、空の帯を詰めない", () => {
    expect(rows.length).toBe(18);
    for (let i = 0; i < rows.length; i += 1) {
      expect(rows[i]?.from).toBe(90 - i * 10);
      expect(rows[i]?.to).toBe(80 - i * 10);
    }
  });

  it("すべての地点がちょうど一つの帯に入る(取りこぼしも二重計上も無い)", () => {
    const assigned = rows.flatMap((r) => r.codes);
    expect(assigned.length).toBe(data.sites.length);
    expect(new Set(assigned).size).toBe(data.sites.length);
  });

  it("北ほど振幅が大きい —— 北緯 40 度以北の中央値が南緯 40 度以南の 5 倍を超える", () => {
    const north = rows.filter((r) => r.to >= 40 && r.values).map((r) => r.amplitude);
    const south = rows.filter((r) => r.from <= -40 && r.values).map((r) => r.amplitude);
    expect(north.length).toBeGreaterThan(0);
    expect(south.length).toBeGreaterThan(0);
    expect(median(north)).toBeGreaterThan(median(south) * 5);
  });
});

// ---------------------------------------------------------------- T-043

describe("正規化", () => {
  it("行を ±1 に収め、形を保つ", () => {
    const row = latitudeBands(data.sites, 10).find((r) => r.values)?.values as number[];
    const norm = normaliseRow(row);

    expect(Math.max(...norm)).toBeCloseTo(1, 9);
    expect(Math.min(...norm)).toBeCloseTo(-1, 9);
    // 山と谷の位置は動かない
    expect(norm.indexOf(Math.max(...norm))).toBe(row.indexOf(Math.max(...row)));
    expect(norm.indexOf(Math.min(...norm))).toBe(row.indexOf(Math.min(...row)));
  });

  it("**南半球の行を実際に見えるようにする** —— 正規化前は色が付かない", () => {
    const southern = latitudeBands(data.sites, 10).find((r) => r.from <= -80 && r.values);
    const raw = southern?.values as number[];

    // 絶対目盛りでは濃度が薄い(実測 2026-09-08: |値| は最大 0.5 ppm 程度)
    const rawIntensity = Math.max(...raw.map((v) => Math.abs(v) / COLOUR_CAP_PPM));
    expect(rawIntensity).toBeLessThan(0.2);

    const normIntensity = Math.max(...normaliseRow(raw).map((v) => Math.abs(v)));
    expect(normIntensity).toBeCloseTo(1, 9);
  });

  it("平坦な行は 0 のまま(0 で割らない)", () => {
    expect(normaliseRow([3, 3, 3, 3])).toEqual([0, 0, 0, 0]);
  });
});

// ---------------------------------------------------------------- T-044

describe("色と頭打ち", () => {
  it("符号で色相が変わり、0 は無色に寄る", () => {
    expect(colourFor(-COLOUR_CAP_PPM)).toContain("hsl(205");
    expect(colourFor(COLOUR_CAP_PPM)).toContain("hsl(15");
    expect(colourFor(0)).toContain("100%");
  });

  it("頭打ちを超えても色は飽和し、破綻しない", () => {
    expect(colourFor(1000)).toBe(colourFor(COLOUR_CAP_PPM));
    expect(colourFor(-1000)).toBe(colourFor(-COLOUR_CAP_PPM));
  });

  it("頭打ちを超える地点が実在する(印を付ける対象が空でない)", () => {
    const over = data.sites.filter((s) => exceedsCap(s));
    expect(over.length).toBeGreaterThan(0);
    expect(over.length).toBeLessThan(data.sites.length);
  });
});

// ---------------------------------------------------------------- T-045

describe("位相と経路", () => {
  it("位相差は環として測る(23 と 1 は 2 ビン差)", () => {
    expect(phaseGap(23, 1, 24)).toBe(2);
    expect(phaseGap(0, 12, 24)).toBe(12);
    expect(phaseGap(5, 5, 24)).toBe(0);
  });

  it("北と南で谷の時期がずれている", () => {
    const north = data.sites.filter((s) => s.latitude > 45).map((s) => s.trough_bin);
    const south = data.sites.filter((s) => s.latitude < -45).map((s) => s.trough_bin);
    expect(north.length).toBeGreaterThan(0);
    expect(south.length).toBeGreaterThan(0);
    expect(phaseGap(median(north), median(south), data.n_bins)).toBeGreaterThan(4);
  });

  it("折れ線の座標が器の中に収まる", () => {
    const values = data.bands[0]?.values as number[];
    const path = ringPath(values, 500, 30, COLOUR_CAP_PPM);
    const points = path
      .slice(1)
      .split("L")
      .map((p) => p.split(",").map(Number) as [number, number]);

    expect(points.length).toBe(values.length);
    for (const [x, y] of points) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(500);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(30);
    }
  });

  it("月のラベルはビン番号から導かれる(決め打ちしない)", () => {
    expect(binToMonthLabel(0, 24)).toBe("1月前半");
    expect(binToMonthLabel(1, 24)).toBe("1月後半");
    expect(binToMonthLabel(23, 24)).toBe("12月後半");
  });
});

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] as number;
}
