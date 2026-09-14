"use client";

import { useMemo, useState } from "react";

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
} from "../lib/breathing";

/**
 * 画面①「呼吸する地球」。
 *
 * 図の座標とラベルは**同じ源から導く**(HC-045)。軸の目盛りも凡例も、
 * データから計算した位置に置く —— 決め打ちしない。
 */

const HEAT_LEFT = 78;
const HEAT_RIGHT = 22;
const HEAT_TOP = 28;
const HEAT_BOTTOM = 34;
const HEAT_WIDTH = 720;
const ROW_HEIGHT = 26;

const LADDER_LEFT = 78;
const LADDER_RIGHT = 92;
const LADDER_TOP = 24;
const LADDER_ROW = 34;
const LADDER_WIDTH = 620;

export default function BreathingEarth({ data }: { data: Screen1 }) {
  const [bin, setBin] = useState(0);
  const [showSouth, setShowSouth] = useState(true);
  const [normalise, setNormalise] = useState(false);

  const rows = useMemo(() => latitudeBands(data.sites, 10), [data.sites]);
  const poc = useMemo(
    () =>
      data.bands
        .filter((b) => b.parent === "poc")
        .filter((b) => showSouth || b.latitude >= 0),
    [data.bands, showSouth],
  );

  const heatHeight = HEAT_TOP + rows.length * ROW_HEIGHT + HEAT_BOTTOM;
  const heatInnerWidth = HEAT_WIDTH - HEAT_LEFT - HEAT_RIGHT;
  const cellWidth = heatInnerWidth / data.n_bins;

  const ladderHeight = LADDER_TOP + poc.length * LADDER_ROW + 40;
  const ladderInner = LADDER_WIDTH - LADDER_LEFT - LADDER_RIGHT;

  const north = data.sites.filter((s) => s.latitude > 45);
  const south = data.sites.filter((s) => s.latitude < -45);
  const northTrough = median(north.map((s) => s.trough_bin));
  const southTrough = median(south.map((s) => s.trough_bin));
  const gap = phaseGap(northTrough, southTrough, data.n_bins);

  const capped = data.sites.filter((s) => exceedsCap(s)).length;
  // 正規化のときは行が ±1 に収まるので、頭打ちも 1 にする
  const cap = normalise ? 1 : COLOUR_CAP_PPM;

  return (
    <section>
      <h2 id="breathing">画面① 呼吸する地球</h2>

      <p>
        横は一年、縦は緯度。<strong>青は吸っている</strong>(その半月の濃度が年の平均より低い)、
        <strong>赤は吐いている</strong>。北ほど脈が深く、赤道で消え、南では向きが入れ替わる。
      </p>

      <div className="controls">
        <label>
          <input
            type="checkbox"
            checked={normalise}
            onChange={(e) => setNormalise(e.target.checked)}
            data-testid="normalise-toggle"
          />
          各行を自分の振幅で正規化する
        </label>
        <label>
          年内の位置
          <input
            type="range"
            min={0}
            max={data.n_bins - 1}
            value={bin}
            onChange={(e) => setBin(Number(e.target.value))}
            aria-label="年内の半月を選ぶ"
            data-testid="bin-slider"
          />
          <strong data-testid="bin-label">{binToMonthLabel(bin, data.n_bins)}</strong>
        </label>
      </div>

      <div className="legend" data-testid="legend">
        <span>
          <span className="swatch" style={{ background: colourFor(-cap, cap) }} />
          吸う{normalise ? "(その行の谷)" : ` −${COLOUR_CAP_PPM} ppm`}
        </span>
        <span>
          <span className="swatch" style={{ background: colourFor(0, cap) }} />0
        </span>
        <span>
          <span className="swatch" style={{ background: colourFor(cap, cap) }} />
          吐く{normalise ? "(その行の山)" : ` +${COLOUR_CAP_PPM} ppm`}
        </span>
        <span data-testid="scale-note">
          {normalise
            ? "行ごとに濃さを揃えた。量は消えて形だけが残る —— 南半球の脈が読めるようになるが、深さは比べられない"
            : `色は ±${COLOUR_CAP_PPM} ppm で頭打ち(帯行の |値| の 95 パーセンタイルが 7.5 ppm)。超える地点が ${capped} / ${data.sites.length}`}
        </span>
      </div>

      <div className="figure-scroll">
        <svg
          id="heatmap"
          viewBox={`0 0 ${HEAT_WIDTH} ${heatHeight}`}
          width={HEAT_WIDTH}
          height={heatHeight}
          role="img"
          aria-label="緯度と年内の位置ごとの二酸化炭素の偏差"
        >
          {rows.map((row, r) => {
            const y = HEAT_TOP + r * ROW_HEIGHT;
            return (
              <g key={`${row.from}`} data-row={row.from}>
                <text
                  x={HEAT_LEFT - 8}
                  y={y + ROW_HEIGHT / 2 + 4}
                  textAnchor="end"
                  fontSize={11}
                  fill="var(--ink-soft)"
                >
                  {labelForBand(row.from, row.to)}
                </text>
                {row.values === null ? (
                  <text
                    x={HEAT_LEFT + 6}
                    y={y + ROW_HEIGHT / 2 + 4}
                    fontSize={10}
                    fill="var(--ink-faint)"
                  >
                    観測地点なし
                  </text>
                ) : (
                  (normalise ? normaliseRow(row.values) : row.values).map((v, i) => (
                    <rect
                      key={i}
                      x={HEAT_LEFT + i * cellWidth}
                      y={y}
                      width={cellWidth + 0.5}
                      height={ROW_HEIGHT - 2}
                      fill={colourFor(v, cap)}
                      data-testid="heat-cell"
                    >
                      {/* title の子は文字列一本にする。JSX で並べると子のテキスト節点が分かれ、
                          React 19 がサーバの HTML と食い違って描画をやり直す(#418・loop_008 の本番検品で発見) */}
                      <title>
                        {`${labelForBand(row.from, row.to)} / ${binToMonthLabel(i, data.n_bins)} / ` +
                          `${(row.values as number[])[i]?.toFixed(2)} ppm / ${row.codes.length} 地点` +
                          (row.codes.length ? ` (${row.codes.join(", ")})` : "")}
                      </title>
                    </rect>
                  ))
                )}
              </g>
            );
          })}

          <rect
            x={HEAT_LEFT + bin * cellWidth}
            y={HEAT_TOP - 6}
            width={cellWidth}
            height={rows.length * ROW_HEIGHT + 8}
            fill="none"
            stroke="var(--ink)"
            strokeWidth={1.6}
            data-testid="bin-marker"
            pointerEvents="none"
          />

          {monthTicks(data.n_bins).map((tick) => (
            <text
              key={tick.month}
              x={HEAT_LEFT + tick.bin * cellWidth}
              y={heatHeight - 14}
              fontSize={11}
              fill="var(--ink-soft)"
              textAnchor="middle"
            >
              {tick.month}月
            </text>
          ))}
          <text x={HEAT_LEFT} y={18} fontSize={11} fill="var(--ink-faint)">
            一年(半月ごと {data.n_bins} 区切り)
          </text>
        </svg>
      </div>

      <p className="note">
        各行は緯度 10 度ぶんの地点を平均したもの。地点は北緯 40〜60 度に偏っているので、
        地点をそのまま縦に並べず<strong>等幅の緯度帯に均して</strong>いる。
        観測地点の無い帯は空のまま残した —— 詰めると緯度の目盛りが嘘になる。
      </p>

      <h3>太平洋の緯度梯子</h3>

      <p>
        同じ船・同じ海・同じ研究室で、緯度だけを 5 度おきに振った列。
        地点ごとの事情(内陸か島か、近くに都市があるか)が揃っているので、
        <strong>振幅が赤道で潰れて南で戻る形</strong>だけが残る。
      </p>

      <div className="controls">
        <label>
          <input
            type="checkbox"
            checked={showSouth}
            onChange={(e) => setShowSouth(e.target.checked)}
            data-testid="south-toggle"
          />
          南半球の帯も出す
        </label>
      </div>

      <div className="figure-scroll">
        <svg
          id="ladder"
          viewBox={`0 0 ${LADDER_WIDTH} ${ladderHeight}`}
          width={LADDER_WIDTH}
          height={ladderHeight}
          role="img"
          aria-label="太平洋周航の緯度帯ごとの年周曲線"
        >
          {poc.map((band, i) => {
            const y = LADDER_TOP + i * LADDER_ROW;
            const path = ringPath(band.values, ladderInner, LADDER_ROW - 8, COLOUR_CAP_PPM);
            return (
              <g key={band.code} transform={`translate(0 ${y})`} data-band={band.code}>
                <text x={LADDER_LEFT - 8} y={LADDER_ROW / 2} textAnchor="end" fontSize={11} fill="var(--ink-soft)">
                  {latitudeLabel(band.latitude)}
                </text>
                <line
                  x1={LADDER_LEFT}
                  x2={LADDER_LEFT + ladderInner}
                  y1={(LADDER_ROW - 8) / 2}
                  y2={(LADDER_ROW - 8) / 2}
                  stroke="var(--line)"
                  strokeWidth={1}
                />
                <path
                  d={path}
                  transform={`translate(${LADDER_LEFT} 0)`}
                  fill="none"
                  stroke={band.latitude >= 0 ? "var(--exhale)" : "var(--inhale)"}
                  strokeWidth={1.8}
                  data-testid="ladder-path"
                />
                <text
                  x={LADDER_LEFT + ladderInner + 8}
                  y={LADDER_ROW / 2}
                  fontSize={10}
                  fill="var(--ink-faint)"
                >
                  {band.amplitude.toFixed(1)} ppm
                </text>
              </g>
            );
          })}
          <text x={LADDER_LEFT} y={14} fontSize={11} fill="var(--ink-faint)">
            一年(縦は ±{COLOUR_CAP_PPM} ppm で頭打ち)
          </text>
        </svg>
      </div>

      <p className="note">
        右端は山と谷の差。<strong>
          30°N で {amplitudeAt(data, 30)} ppm、赤道で {amplitudeAt(data, 0)} ppm、
          35°S で {amplitudeAt(data, -35)} ppm
        </strong>
        。南半球の帯は北半球と山谷が入れ替わる —— 北緯 45 度以北の地点の谷は
        {binToMonthLabel(northTrough, data.n_bins)} 頃、南緯 45 度以南は
        {binToMonthLabel(southTrough, data.n_bins)} 頃で、
        {((gap / data.n_bins) * 12).toFixed(0)} か月ずれている。
      </p>

      <div className="panel">
        <p className="note" style={{ margin: 0 }}>
          <strong>この図が何から出ているか。</strong>
          {data.provenance.note}
          {" 地点の曲線は"}
          {data.provenance.site_curves}
          {"。帯の曲線は"}
          {data.provenance.band_curves}
          {"。"}
          採用した地点年は {data.coverage["site_years_accepted"]} 件で、
          欠けたビンがあるため落としたのが {data.coverage["site_years_dropped_incomplete"]} 件
          (固定地点の {data.coverage["site_years_fixed_sites"]} 件中)。
        </p>
      </div>
    </section>
  );
}

function labelForBand(from: number, to: number): string {
  const mid = (from + to) / 2;
  if (mid > 0) return `北緯 ${to}〜${from}`;
  if (mid < 0) return `南緯 ${Math.abs(from)}〜${Math.abs(to)}`;
  return `赤道 ±${Math.abs(to)}`;
}

function latitudeLabel(latitude: number): string {
  if (latitude > 0) return `北緯 ${latitude}°`;
  if (latitude < 0) return `南緯 ${Math.abs(latitude)}°`;
  return "赤道";
}

function monthTicks(nBins: number): { month: number; bin: number }[] {
  return Array.from({ length: 6 }, (_, k) => {
    const month = k * 2 + 1;
    return { month, bin: ((month - 1) / 12) * nBins + nBins / 24 };
  });
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] as number;
}

function amplitudeAt(data: Screen1, latitude: number): string {
  const band = data.bands.find((b) => b.parent === "poc" && b.latitude === latitude);
  return band ? band.amplitude.toFixed(1) : "—";
}
