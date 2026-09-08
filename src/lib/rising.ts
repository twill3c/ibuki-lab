/**
 * 画面②が使う計算。描画から切り離してあるので単体で検査できる。
 */

export type Series = {
  code: string;
  name: string;
  latitude: number;
  times: number[];
  values: number[];
  trend: number[];
  n_samples: number;
  year_min: number;
  year_max: number;
  rise_ppm: number;
  amplitude_ppm: number;
};

export type ExternalSystem = {
  source: string;
  mae: number;
  median_ae: number;
  worst_site_ae: number;
  n_sites: number;
  n_examples: number;
};

export type Screen2 = {
  trend_degree: number;
  noaa: Series[];
  jma: Series[];
  external: {
    systems: Record<string, ExternalSystem>;
    jma_sites: {
      code: string;
      true_latitude: number;
      predicted_latitude: number;
      predicted_sd: number;
      mae: number;
      years: number;
    }[];
    decomposition: { product_effect_deg: number; network_effect_deg: number; note: string };
    model: { generalisation_estimate_mae: number; note: string };
    stations: Record<string, { name: string; latitude: number; longitude: number; note: string }>;
    coverage: Record<string, { years_accepted: number; years_dropped_incomplete: number }>;
  };
  provenance: { noaa: string; jma: string };
};

/** 長期の上がりを差し引いた残り。**引き算だけで、平滑を足さない。** */
export function residuals(series: Series): number[] {
  return series.values.map((v, i) => v - (series.trend[i] as number));
}

export type Extent = { xMin: number; xMax: number; yMin: number; yMax: number };

/** 複数の系列に共通の範囲。**系列ごとに軸を変えない** —— 比べる図で軸が動けば嘘になる。 */
export function sharedExtent(series: Series[], detrended: boolean): Extent {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const s of series) {
    const ys = detrended ? residuals(s) : s.values;
    for (const t of s.times) {
      if (t < xMin) xMin = t;
      if (t > xMax) xMax = t;
    }
    for (const y of ys) {
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
  }
  const pad = (yMax - yMin) * 0.04;
  return { xMin, xMax, yMin: yMin - pad, yMax: yMax + pad };
}

/** 系列を SVG のパスにする。器の外へ出さない。 */
export function seriesPath(
  series: Series,
  extent: Extent,
  width: number,
  height: number,
  detrended: boolean,
): string {
  const ys = detrended ? residuals(series) : series.values;
  const spanX = extent.xMax - extent.xMin || 1;
  const spanY = extent.yMax - extent.yMin || 1;
  const points = series.times.map((t, i) => {
    const x = ((t - extent.xMin) / spanX) * width;
    const y = height - (((ys[i] as number) - extent.yMin) / spanY) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `M${points.join("L")}`;
}

/** 軸の目盛りを**データの範囲から**作る。決め打ちしない(HC-045)。 */
export function yearTicks(extent: Extent, step = 10): number[] {
  const first = Math.ceil(extent.xMin / step) * step;
  const out: number[] = [];
  for (let year = first; year <= extent.xMax; year += step) out.push(year);
  return out;
}

export function valueTicks(extent: Extent, count = 5): number[] {
  const span = extent.yMax - extent.yMin;
  const raw = span / (count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude * 10;
  const first = Math.ceil(extent.yMin / step) * step;
  const out: number[] = [];
  for (let v = first; v <= extent.yMax; v += step) out.push(Number(v.toFixed(6)));
  return out;
}
