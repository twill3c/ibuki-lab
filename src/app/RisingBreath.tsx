"use client";

import { useMemo, useState } from "react";

import {
  residuals,
  seriesPath,
  sharedExtent,
  valueTicks,
  yearTicks,
  type Screen2,
} from "../lib/rising";

const WIDTH = 760;
const HEIGHT = 320;
const LEFT = 56;
const RIGHT = 96;
const TOP = 16;
const BOTTOM = 34;

const COLOURS: Record<string, string> = {
  brw: "hsl(205 60% 42%)",
  mlo: "hsl(35 70% 42%)",
  spo: "hsl(275 40% 45%)",
  ryo: "hsl(0 65% 45%)",
};

export default function RisingBreath({ data }: { data: Screen2 }) {
  const [detrended, setDetrended] = useState(false);
  const [withJma, setWithJma] = useState(false);

  const series = useMemo(
    () => (withJma ? [...data.noaa, ...data.jma] : data.noaa),
    [data.noaa, data.jma, withJma],
  );
  const extent = useMemo(() => sharedExtent(series, detrended), [series, detrended]);

  const innerWidth = WIDTH - LEFT - RIGHT;
  const innerHeight = HEIGHT - TOP - BOTTOM;

  const ext = data.external;
  const ref = ext.systems["noaa_event_reference"];
  const control = ext.systems["noaa_monthly_control"];
  const holdout = ext.systems["jma_holdout"];

  return (
    <section>
      <h2 id="rising">画面② 積み上がる息</h2>

      <p>
        毎年の脈は、上がり続ける台の上で起きている。ここに出しているのは
        <strong>平滑を経ていない生の測定結果</strong>で、点の一つ一つがフラスコ一本である。
      </p>

      <div className="controls">
        <label>
          <input
            type="checkbox"
            checked={detrended}
            onChange={(e) => setDetrended(e.target.checked)}
            data-testid="detrend-toggle"
          />
          長期の上がりを差し引く
        </label>
        <label>
          <input
            type="checkbox"
            checked={withJma}
            onChange={(e) => setWithJma(e.target.checked)}
            data-testid="jma-toggle"
          />
          気象庁 綾里を重ねる
        </label>
      </div>

      <div className="figure-scroll">
        <svg
          id="rising"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          width={WIDTH}
          height={HEIGHT}
          role="img"
          aria-label={detrended ? "長期の上がりを外した年周の残差" : "二酸化炭素濃度の推移"}
        >
          {valueTicks(extent).map((v) => {
            const y =
              TOP + innerHeight - ((v - extent.yMin) / (extent.yMax - extent.yMin)) * innerHeight;
            return (
              <g key={v}>
                <line x1={LEFT} x2={LEFT + innerWidth} y1={y} y2={y} stroke="var(--line)" />
                <text x={LEFT - 6} y={y + 4} fontSize={10} textAnchor="end" fill="var(--ink-faint)">
                  {v}
                </text>
              </g>
            );
          })}
          {yearTicks(extent).map((year) => {
            const x = LEFT + ((year - extent.xMin) / (extent.xMax - extent.xMin)) * innerWidth;
            return (
              <text
                key={year}
                x={x}
                y={HEIGHT - 12}
                fontSize={10}
                textAnchor="middle"
                fill="var(--ink-faint)"
              >
                {year}
              </text>
            );
          })}

          {series.map((s, i) => (
            <g key={s.code}>
              <path
                d={seriesPath(s, extent, innerWidth, innerHeight, detrended)}
                transform={`translate(${LEFT} ${TOP})`}
                fill="none"
                stroke={COLOURS[s.code] ?? "var(--ink-soft)"}
                strokeWidth={0.8}
                strokeOpacity={0.85}
                data-testid="rising-path"
                data-series={s.code}
              />
              <text
                x={LEFT + innerWidth + 8}
                y={TOP + 12 + i * 15}
                fontSize={10}
                fill={COLOURS[s.code] ?? "var(--ink-soft)"}
                data-testid="series-label"
              >
                {shortLabel(s.code, s.latitude)}
              </text>
            </g>
          ))}
          <text x={LEFT} y={TOP - 2} fontSize={10} fill="var(--ink-faint)">
            {detrended ? "ppm(長期の上がりを外した残り)" : "ppm"}
          </text>
        </svg>
      </div>

      <p className="note" data-testid="rising-note">
        {detrended
          ? `上がりを外すと、毎年ほぼ同じ形の脈が残る。振れ幅は ${data.noaa
              .map((s) => `${shortLabel(s.code, s.latitude)} ${s.amplitude_ppm.toFixed(1)}`)
              .join(" / ")} ppm で、緯度の順に並ぶ。`
          : `${data.noaa[0]?.year_min}年から${data.noaa[0]?.year_max}年で ${data.noaa
              .map((s) => `${shortLabel(s.code, s.latitude)} +${s.rise_ppm.toFixed(0)}`)
              .join(" / ")} ppm。どこで測っても同じだけ上がる —— 二酸化炭素はよく混ざる。`}
      </p>

      <h3 id="external">気象庁 3 地点での外部検証</h3>

      <p>
        ここまでに出てきた数はすべて NOAA の網の中のものである。
        <strong>
          気象庁の観測は別の機関が別の較正で運用しており、この模型は学習にも、閾値の決定にも、
          学習率の選定にも一度も使っていない。
        </strong>
      </p>

      <div className="figure-scroll">
      <table data-testid="external-table">
        <thead>
          <tr>
            <th>系</th>
            <th>網</th>
            <th>資料形</th>
            <th>地点</th>
            <th>MAE(度)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>NOAA・event(同じ模型の基準)</td>
            <td>NOAA</td>
            <td>生の測定結果</td>
            <td>{ref?.n_sites}</td>
            <td>{ref?.mae.toFixed(2)}</td>
          </tr>
          <tr>
            <td>NOAA・月次(資料形の対照)</td>
            <td>NOAA</td>
            <td>月平均</td>
            <td>{control?.n_sites}</td>
            <td>{control?.mae.toFixed(2)}</td>
          </tr>
          <tr>
            <td>
              <strong>気象庁 3 地点</strong>
            </td>
            <td>気象庁</td>
            <td>月平均</td>
            <td>{holdout?.n_sites}</td>
            <td>
              <strong>{holdout?.mae.toFixed(2)}</strong>
            </td>
          </tr>
        </tbody>
      </table>
      </div>

      <p className="note">
        上の二つは<strong>学習に使った例</strong>なので絶対値は楽観的である。
        比べてよいのは差のほうで、
        <strong>
          資料形の効果が {ext.decomposition.product_effect_deg >= 0 ? "+" : ""}
          {ext.decomposition.product_effect_deg.toFixed(2)} 度、網の効果が{" "}
          {ext.decomposition.network_effect_deg >= 0 ? "+" : ""}
          {ext.decomposition.network_effect_deg.toFixed(2)} 度
        </strong>
        。**劣化のほとんどは資料形の側**で、別の観測網へ移ること自体はほとんど効いていない。
        気象庁での {holdout?.mae.toFixed(2)} 度は、未知の地点に対する見込み
        {ext.model.generalisation_estimate_mae.toFixed(2)} 度と同程度である。
      </p>

      <h3>同じ緯度の二地点が割れた</h3>

      <div className="figure-scroll">
      <table data-testid="jma-table">
        <thead>
          <tr>
            <th>地点</th>
            <th>本当の緯度</th>
            <th>模型の答え</th>
            <th>誤差</th>
            <th>年数</th>
          </tr>
        </thead>
        <tbody>
          {ext.jma_sites.map((row) => (
            <tr key={row.code}>
              <td>{ext.stations[row.code]?.name}</td>
              <td>{row.true_latitude.toFixed(2)}°</td>
              <td>
                {row.predicted_latitude.toFixed(1)}° ±{row.predicted_sd.toFixed(1)}
              </td>
              <td>{row.mae.toFixed(1)}°</td>
              <td>{row.years}</td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>

      <div className="panel">
        <p style={{ margin: 0 }}>
          <strong>南鳥島と与那国島はほぼ同じ緯度にある</strong>
          (北緯 {ext.stations["mnm"]?.latitude.toFixed(2)}° と{" "}
          {ext.stations["yon"]?.latitude.toFixed(2)}°)。経度は 31 度離れていて、
          片方は太平洋の孤島、もう片方は大陸のすぐ手前にある。
          模型はこの二つに {jmaSpread(ext).toFixed(1)} 度違う答えを出した。
        </p>
        <p className="note" style={{ marginBottom: 0 }}>
          つまり<strong>模型が読んでいるのは緯度そのものではなく年周振幅</strong>で、
          振幅は大陸への近さで決まる。学習に使った地点では緯度と大陸への近さが絡んでいた
          (北の地点ほど陸に近い)ので、模型はその絡んだ手がかりを覚えた。
          綾里(内陸寄りの日本)に北緯 {predictedFor(ext, "ryo")?.toFixed(0)} 度と答えたのも同じ理由である。
          <strong>この割れ方は、同緯度の二地点を並べるまで見えなかった。</strong>
        </p>
      </div>
    </section>
  );
}

/**
 * 系列の短い名前。**正式名をそのまま置くと図からはみ出す** ——
 * Barrow の正式名は読点が無いので、読点で切る手当ては効かなかった(loop_005 GEN-LAYOUT)。
 * 符号と緯度だけにして、正式名は本文の表に置く。
 */
function shortLabel(code: string, latitude: number): string {
  const hemisphere = latitude >= 0 ? "N" : "S";
  return `${code.toUpperCase()} ${Math.abs(latitude).toFixed(0)}${hemisphere}`;
}

function jmaSpread(ext: Screen2["external"]): number {
  const mnm = ext.jma_sites.find((r) => r.code === "mnm")?.predicted_latitude ?? 0;
  const yon = ext.jma_sites.find((r) => r.code === "yon")?.predicted_latitude ?? 0;
  return Math.abs(mnm - yon);
}

function predictedFor(ext: Screen2["external"], code: string): number | undefined {
  return ext.jma_sites.find((r) => r.code === code)?.predicted_latitude;
}
