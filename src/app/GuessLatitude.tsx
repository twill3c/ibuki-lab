"use client";

import { useMemo, useState } from "react";

import { forward, type Weights } from "../model/forward";
import {
  amplitudeOf,
  gradientTimesInput,
  latitudeToUnit,
  morph,
  saliencyCap,
  saliencyColour,
  type Choice,
} from "../lib/saliency";

/**
 * 画面③「当ててみる」。
 *
 * **主題は「模型に何が見えていて、何が見えていないか」である。** 振幅を動かすと
 * 答えが動き、位相を回しても答えが動かない —— 後者は構造から来る不変性で、
 * G-09 の負けの機構でもある(SPEC §7.15)。触って確かめられる形にした。
 */

export type Screen3 = {
  n_bins: number;
  choices: (Choice & { in_training: boolean; label: string; note: string })[];
  default_code: string;
};

const W = 720;
const H = 210;
const PAD_L = 44;
const PAD_R = 16;
const PAD_T = 18;
const PAD_B = 30;

const STRIP_W = 190;
const STRIP_H = 260;
const STRIP_X = 62;

export default function GuessLatitude({
  data,
  weights,
  generalisationMae,
}: {
  data: Screen3;
  weights: Weights;
  generalisationMae: number;
}) {
  const [code, setCode] = useState(data.default_code);
  const [factor, setFactor] = useState(1);
  const [shift, setShift] = useState(0);

  const choice = useMemo(
    () => data.choices.find((c) => c.code === code) ?? (data.choices[0] as Screen3["choices"][0]),
    [data.choices, code],
  );
  const curve = useMemo(() => morph(choice.values, factor, shift), [choice, factor, shift]);
  const answer = useMemo(() => forward(weights, curve), [weights, curve]);
  const contributions = useMemo(() => gradientTimesInput(weights, curve), [weights, curve]);
  const cap = saliencyCap(contributions);

  const groups = useMemo(() => {
    const out = new Map<string, typeof data.choices>();
    for (const c of data.choices) {
      const list = out.get(c.group) ?? [];
      list.push(c);
      out.set(c.group, list);
    }
    return [...out];
  }, [data.choices]);

  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const yScale = Math.max(...curve.map((v) => Math.abs(v)), 1) * 1.15;
  const cellW = innerW / curve.length;
  const error = Math.abs(answer - choice.latitude);

  return (
    <section>
      <h2 id="guess">画面③ 当ててみる</h2>

      <p>
        曲線を一本選ぶと、<strong>ブラウザの中の模型</strong>がその空気の緯度を答える。
        通信は起きない —— 重みは 9 KB の JSON で、前向きはこのページの中で走る。
      </p>

      <div className="controls">
        <label>
          曲線
          <select
            value={code}
            onChange={(e) => setCode(e.target.value)}
            data-testid="curve-select"
          >
            {groups.map(([group, list]) => (
              <optgroup key={group} label={group}>
                {list.map((c) => (
                  <option key={c.code} value={c.code}>
                    {c.label}({c.latitude.toFixed(1)}°)
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
      </div>

      <div className="controls">
        <label>
          振幅 ×{factor.toFixed(2)}
          <input
            type="range"
            min={0.2}
            max={2.5}
            step={0.05}
            value={factor}
            onChange={(e) => setFactor(Number(e.target.value))}
            data-testid="factor-slider"
          />
        </label>
        <label>
          位相 {shift === 0 ? "そのまま" : `${(shift / 2).toFixed(1)} か月ずらす`}
          <input
            type="range"
            min={0}
            max={data.n_bins - 1}
            step={1}
            value={shift}
            onChange={(e) => setShift(Number(e.target.value))}
            data-testid="shift-slider"
          />
        </label>
        <button
          type="button"
          onClick={() => {
            setFactor(1);
            setShift(0);
          }}
          data-testid="reset-morph"
        >
          もとに戻す
        </button>
      </div>

      <div className="two-up">
        <div className="figure-scroll">
          <svg
            id="guess-curve"
            viewBox={`0 0 ${W} ${H}`}
            width={W}
            height={H}
            role="img"
            aria-label="選んだ年周曲線と、各半月が答えへ寄せた度数"
          >
            {curve.map((v, i) => (
              <rect
                key={i}
                x={PAD_L + i * cellW}
                y={PAD_T}
                width={cellW + 0.5}
                height={innerH}
                fill={saliencyColour(contributions[i] as number, cap)}
                data-testid="saliency-cell"
              >
                {/* title の子は文字列一本にする(React #418・svg-title.test.ts) */}
                <title>
                  {`${monthLabel(i, curve.length)} / 値 ${v.toFixed(2)} ppm / 寄与 ` +
                    `${(contributions[i] as number).toFixed(2)} 度`}
                </title>
              </rect>
            ))}

            <line
              x1={PAD_L}
              x2={PAD_L + innerW}
              y1={PAD_T + innerH / 2}
              y2={PAD_T + innerH / 2}
              stroke="var(--line)"
            />
            <path
              d={curvePath(curve, innerW, innerH, yScale)}
              transform={`translate(${PAD_L} ${PAD_T})`}
              fill="none"
              stroke="var(--ink)"
              strokeWidth={2}
              data-testid="guess-path"
            />

            {[yScale, 0, -yScale].map((v) => (
              <text
                key={v}
                x={PAD_L - 6}
                y={PAD_T + innerH / 2 - (v / yScale) * (innerH / 2) + 4}
                fontSize={10}
                textAnchor="end"
                fill="var(--ink-faint)"
              >
                {v.toFixed(0)}
              </text>
            ))}
            {[1, 4, 7, 10].map((month) => (
              <text
                key={month}
                x={PAD_L + ((month - 1) / 12) * innerW + cellW}
                y={H - 10}
                fontSize={10}
                textAnchor="middle"
                fill="var(--ink-faint)"
              >
                {month}月
              </text>
            ))}
            <text x={PAD_L} y={12} fontSize={10} fill="var(--ink-faint)">
              ppm(年内の直線を外した偏差)。背景の濃さは、その半月が答えへ寄せた度数
            </text>
          </svg>
        </div>

        <div className="figure-scroll">
          <svg
            id="guess-strip"
            viewBox={`0 0 ${STRIP_W} ${STRIP_H}`}
            width={STRIP_W}
            height={STRIP_H}
            role="img"
            aria-label="模型の答えと本当の緯度"
          >
            <rect x={STRIP_X} y={14} width={40} height={STRIP_H - 44} fill="hsl(40 20% 94%)" />
            {[90, 60, 30, 0, -30, -60, -90].map((lat) => {
              const y = 14 + latitudeToUnit(lat) * (STRIP_H - 44);
              return (
                <g key={lat}>
                  <line x1={STRIP_X} x2={STRIP_X + 40} y1={y} y2={y} stroke="var(--line)" />
                  <text x={STRIP_X - 6} y={y + 4} fontSize={10} textAnchor="end" fill="var(--ink-faint)">
                    {lat === 0 ? "赤道" : `${Math.abs(lat)}°${lat > 0 ? "N" : "S"}`}
                  </text>
                </g>
              );
            })}

            <g data-testid="truth-marker" transform={`translate(0 ${14 + latitudeToUnit(choice.latitude) * (STRIP_H - 44)})`}>
              <line x1={STRIP_X - 2} x2={STRIP_X + 42} y1={0} y2={0} stroke="var(--ink)" strokeWidth={2} />
              <text x={STRIP_X + 46} y={4} fontSize={10} fill="var(--ink)">
                本当 {choice.latitude.toFixed(1)}°
              </text>
            </g>

            <g data-testid="answer-marker" transform={`translate(0 ${14 + latitudeToUnit(answer) * (STRIP_H - 44)})`}>
              <rect
                x={STRIP_X - 4}
                y={-((generalisationMae / 180) * (STRIP_H - 44))}
                width={48}
                height={((2 * generalisationMae) / 180) * (STRIP_H - 44)}
                fill="var(--accent)"
                fillOpacity={0.18}
              />
              <line x1={STRIP_X - 4} x2={STRIP_X + 44} y1={0} y2={0} stroke="var(--accent)" strokeWidth={2.5} />
              <text x={STRIP_X + 46} y={-4} fontSize={11} fill="var(--accent)" fontWeight={600}>
                {answer.toFixed(1)}°
              </text>
            </g>
            <text x={4} y={STRIP_H - 8} fontSize={9} fill="var(--ink-faint)">
              帯は未知の地点での見込み ±{generalisationMae.toFixed(1)}°
            </text>
          </svg>
        </div>
      </div>

      <p className="note" data-testid="guess-readout">
        {choice.label}({choice.note})。
        {choice.in_training ? (
          <strong>この地点は学習に使っている</strong>
        ) : (
          <strong>この曲線は学習に使っていない</strong>
        )}
        。振幅 {amplitudeOf(curve).toFixed(1)} ppm に対して模型の答えは{" "}
        {answer.toFixed(1)}°、本当の緯度は {choice.latitude.toFixed(1)}° で、
        差は {error.toFixed(1)}° ある。
      </p>

      <div className="panel">
        <h3 style={{ marginTop: 0 }}>位相を回しても答えは動かない</h3>
        <p>
          上の「位相」を動かしてみてほしい。曲線は横にずれるのに、
          <strong>答えは 1 度も動かない</strong>。これは学習の失敗ではなく、
          この模型の**構造から来る性質**である —— 巡回畳み込みはずらしに対して同変で、
          そのあとの要約(平均と絶対値の最大)がずらしに対して不変なので、
          前向き全体がずれを見ない。
        </p>
        <p className="note" style={{ marginBottom: 0 }}>
          <strong>これが G-09(目玉②)の負けの機構である。</strong>
          相手の「2 特徴」は第一調和の係数 a₁ と b₁、つまり<strong>振幅と位相の両方</strong>を持つ。
          こちらは振幅の大きさしか持たない。**半分の手がかりで戦って負けた**のであって、
          「ディープラーニングがこの問題に向かない」を測ったわけではない。
          位相を見える形にした模型で測り直すことを L7 に事前登録した(§6 P-07)。
        </p>
      </div>

      <p className="note">
        背景の濃さは「その半月の値を少し動かすと答えが何度動くか」に、その点の値を掛けたもの。
        <strong>「模型がそこを見ている」とは書かない</strong> —— 測ったのは前者だけである。
        中心差分で出しており、JAX の自動微分と照合してある(T-052)。
      </p>
    </section>
  );
}

function curvePath(values: number[], width: number, height: number, scale: number): string {
  const n = values.length;
  const points = values.map((v, i) => {
    const x = ((i + 0.5) / n) * width;
    const y = height / 2 - (v / scale) * (height / 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `M${points.join("L")}`;
}

function monthLabel(bin: number, nBins: number): string {
  const month = Math.floor((bin / nBins) * 12) + 1;
  const half = (bin / nBins) * 12 - Math.floor((bin / nBins) * 12) < 0.5 ? "前半" : "後半";
  return `${month}月${half}`;
}
