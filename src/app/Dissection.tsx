"use client";

/**
 * 画面④「解剖台」。
 *
 * **通らなかったゲートと、測って捨てた記録を消さずに出す**(F-07)。
 * 成績表だけを出すと、うまくいった部分しか読めない —— このプロジェクトで
 * いちばん価値のある記録は、外れた予測と負けの機構が分かった経緯のほうである。
 */

export type Screen4 = {
  opponent: { system: string; mae: number };
  population: { examples: number; sites: number; groups: number; paired_examples: number };
  systems: { name: string; mae: number; spread?: number; kind: string }[];
  gates: { id: string; title: string; verdict: string; detail: string }[];
  discarded: { title: string; loop: string; detail: string }[];
  predictions: { id: string; text: string; verdict: string }[];
  phase: {
    invariant_mae: number;
    invariant_spread: number;
    harmonic_mae: number;
    harmonic_spread: number;
    harmonic_seeds: number[];
    lr_sweep: Record<string, { val_mae: number; test_mae_per_site: number }>;
    chosen_lr: number;
  };
  ch4: { real_minus_base: number; shuffled_minus_base: number; note: string } | null;
  ch4_systems: Record<string, { mae: number; spread: number }>;
  nested: {
    train_fraction: number;
    harmonic: NestedRow;
    invariant: NestedRow;
    summary: string[];
  } | null;
  ch4_components: {
    rows: { label: string; mae: number; spread: number; retained: number; seeds_agree: boolean; side: string }[];
    summary: string;
  } | null;
  verification: { two_implementation: string; browser_checks: number };
};

type NestedRow = { mae: number; spread: number; stop_rule_mae: number; seeds: number[] };

const BAR_W = 660;
const ROW_H = 30;
const LABEL_W = 210;

const KIND_COLOUR: Record<string, string> = {
  baseline: "hsl(40 12% 62%)",
  opponent: "hsl(15 65% 46%)",
  model: "hsl(205 60% 42%)",
  control: "hsl(0 0% 78%)",
};

export default function Dissection({ data }: { data: Screen4 }) {
  const maxMae = Math.max(...data.systems.map((s) => s.mae)) * 1.08;
  const height = data.systems.length * ROW_H + 34;
  const sweep = Object.entries(data.phase.lr_sweep);

  return (
    <section>
      <h2 id="dissection">画面④ 解剖台</h2>

      <p>
        ここには<strong>通らなかったゲートと、測って捨てた記録</strong>も出す。
        成績だけを並べると、うまくいった部分しか読めない。
      </p>

      <h3>誰がどれだけ当てたか</h3>

      <p className="note">
        {data.population.examples} 地点年 / {data.population.sites} 地点 /{" "}
        {data.population.groups} 群。**地点群ばらしの leave-one-group-out**で、
        同じ地点の年が学習側と試験側に跨らない。棒が短いほど当たっている。
      </p>

      <div className="figure-scroll">
        <svg
          id="scoreboard"
          viewBox={`0 0 ${BAR_W} ${height}`}
          width={BAR_W}
          height={height}
          role="img"
          aria-label="系ごとの平均絶対誤差"
        >
          {data.systems.map((s, i) => {
            const y = i * ROW_H + 8;
            const w = ((BAR_W - LABEL_W - 64) * s.mae) / maxMae;
            return (
              <g key={s.name} data-system={s.name}>
                <text x={LABEL_W - 8} y={y + 14} fontSize={11} textAnchor="end" fill="var(--ink-soft)">
                  {s.name}
                </text>
                <rect
                  x={LABEL_W}
                  y={y + 3}
                  width={w}
                  height={ROW_H - 12}
                  fill={KIND_COLOUR[s.kind] ?? "var(--ink-faint)"}
                  data-testid="score-bar"
                />
                {s.spread !== undefined && (
                  <line
                    x1={LABEL_W + w - ((BAR_W - LABEL_W - 64) * s.spread) / (2 * maxMae)}
                    x2={LABEL_W + w + ((BAR_W - LABEL_W - 64) * s.spread) / (2 * maxMae)}
                    y1={y + ROW_H / 2 - 3}
                    y2={y + ROW_H / 2 - 3}
                    stroke="var(--ink)"
                    strokeWidth={1.4}
                    data-testid="spread-bar"
                  />
                )}
                <text x={LABEL_W + w + 6} y={y + 14} fontSize={11} fill="var(--ink)">
                  {s.mae.toFixed(2)}
                  {s.spread !== undefined ? ` (幅 ${s.spread.toFixed(2)})` : ""}
                </text>
              </g>
            );
          })}
          <text x={LABEL_W} y={height - 8} fontSize={10} fill="var(--ink-faint)">
            地点あたりの平均絶対誤差(度)。黒い横線は種を変えたときの幅
          </text>
        </svg>
      </div>

      <h3>位相を足したらどうなったか</h3>

      <div className="panel">
        <p style={{ marginTop: 0 }}>
          位相を見える要約に替えると、平均は{" "}
          <strong>
            {data.phase.invariant_mae.toFixed(2)} → {data.phase.harmonic_mae.toFixed(2)} 度
          </strong>
          に下がった(
          {(data.phase.invariant_mae - data.phase.harmonic_mae).toFixed(2)} 度の改善)。
          相手の {data.opponent.mae.toFixed(2)} 度も{" "}
          {(data.opponent.mae - data.phase.harmonic_mae).toFixed(2)} 度下回っている。
        </p>
        <p className="note" style={{ marginBottom: 0 }}>
          <strong>ただし勝ったとは言わない。</strong>種ごとの値は{" "}
          {data.phase.harmonic_seeds.map((m) => m.toFixed(2)).join(" / ")} で、
          幅が {data.phase.harmonic_spread.toFixed(2)} 度ある —— 相手との差
          {(data.opponent.mae - data.phase.harmonic_mae).toFixed(2)} 度の{" "}
          {((data.phase.harmonic_spread / (data.opponent.mae - data.phase.harmonic_mae)) || 0).toFixed(0)}{" "}
          倍である。**意味があるのは相手との勝ち負けでなく、位相を足して縮んだ
          {(data.phase.invariant_mae - data.phase.harmonic_mae).toFixed(2)} 度のほう。**
        </p>
      </div>

      <h4>学習率の掃引 —— 選ぶ規則が試験と逆順に並んだ</h4>

      <div className="figure-scroll">
        <table data-testid="sweep-table">
          <thead>
            <tr>
              <th>学習率</th>
              <th>内側検証</th>
              <th>試験</th>
            </tr>
          </thead>
          <tbody>
            {sweep.map(([lr, row]) => (
              <tr key={lr}>
                <td>
                  {lr}
                  {Number(lr) === data.phase.chosen_lr ? "(採用)" : ""}
                </td>
                <td>{row.val_mae.toFixed(2)}</td>
                <td>{row.test_mae_per_site.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="note">
        学習率は<strong>内側検証だけ</strong>で選ぶと事前に決めてある(試験を見て選べば
        それはもう成績ではない)。だがこの系では<strong>内側検証が良い候補ほど試験が悪い</strong>。
        内側検証は止め時を選ぶのにも使っているので偏った推定量になっており、
        **学習率という別の軸の比較には使えない**。規則は動かさず、限界を記録する。
      </p>

      {data.nested && (
        <>
          <h4>選び方を nested にして測り直した</h4>
          <p className="note">
            止め時を選ぶ分割と学習率を選ぶ分割を分け、fold ごとに学習率を選んだ。
            訓練に回る群は {Math.round(data.nested.train_fraction * 100)}% に減るので、
            全部で学ぶ相手に対しては<strong>模型に不利な向き</strong>の比較である。
            判定の線は測る前に書いた。
          </p>
          <div className="figure-scroll">
            <table data-testid="nested-table">
              <thead>
                <tr>
                  <th>要約</th>
                  <th>nested で選んだ MAE</th>
                  <th>種の幅</th>
                  <th>同じ重みを止め時の分割で選んだ MAE</th>
                </tr>
              </thead>
              <tbody>
                {(["harmonic", "invariant"] as const).map((key) => {
                  const row = data.nested![key];
                  return (
                    <tr key={key}>
                      <td>{key === "harmonic" ? "位相あり" : "位相なし"}</td>
                      <td>{row.mae.toFixed(2)}</td>
                      <td>{row.spread.toFixed(2)}</td>
                      <td>{row.stop_rule_mae.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {data.nested.summary.map((line) => (
            <p className="note" key={line}>
              {line}
            </p>
          ))}
        </>
      )}

      {data.ch4 && (
        <>
          <h3>CH4 は情報を足したのか、チャンネルが増えただけか</h3>
          <div className="figure-scroll">
            <table data-testid="ch4-table">
              <thead>
                <tr>
                  <th>系</th>
                  <th>MAE(度)</th>
                  <th>基準との差</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>CO2 のみ(基準)</td>
                  <td>{data.ch4_systems["cnn_invariant_paired"]?.mae.toFixed(2)}</td>
                  <td>—</td>
                </tr>
                <tr>
                  <td>CO2 + 本物の CH4</td>
                  <td>{data.ch4_systems["cnn_ch4_paired"]?.mae.toFixed(2)}</td>
                  <td>{data.ch4.real_minus_base >= 0 ? "+" : ""}{data.ch4.real_minus_base.toFixed(2)}</td>
                </tr>
                <tr>
                  <td>
                    CO2 + <strong>入れ替えた CH4</strong>(対照)
                  </td>
                  <td>{data.ch4_systems["cnn_ch4_shuffled_paired"]?.mae.toFixed(2)}</td>
                  <td>
                    {data.ch4.shuffled_minus_base >= 0 ? "+" : ""}
                    {data.ch4.shuffled_minus_base.toFixed(2)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="note">{data.ch4.note}</p>
        </>
      )}

      {data.ch4_components && (
        <>
          <h4>CH4 の何が効いたのか —— 要素を一つずつ潰した</h4>
          <div className="figure-scroll">
            <table data-testid="ch4-components-table">
              <thead>
                <tr>
                  <th>対照</th>
                  <th>MAE(度)</th>
                  <th>種の幅</th>
                  <th>効き目の残り r</th>
                  <th>3 種の側</th>
                </tr>
              </thead>
              <tbody>
                {data.ch4_components.rows.map((r) => (
                  <tr key={r.label}>
                    <td>{r.label}</td>
                    <td>{r.mae.toFixed(2)}</td>
                    <td>{r.spread.toFixed(2)}</td>
                    <td>{r.retained.toFixed(2)}</td>
                    <td>{r.side}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            r = (基準 − 対照) ÷ (基準 − 本物)。1 なら本物の CH4 と同じだけ効き、0 なら CH4 が無いのと同じ。
            {" "}
            {data.ch4_components.summary}
          </p>
        </>
      )}

      <h3>品質ゲートの判定</h3>

      <div className="figure-scroll">
        <table data-testid="gate-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>内容</th>
              <th>判定</th>
            </tr>
          </thead>
          <tbody>
            {data.gates.map((g) => (
              <tr key={g.id}>
                <td>{g.id}</td>
                <td style={{ whiteSpace: "normal", maxWidth: "34rem", textAlign: "left" }}>
                  <strong>{g.title}</strong>
                  <br />
                  <span className="note">{g.detail}</span>
                </td>
                <td>
                  <strong>{g.verdict}</strong>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>測って捨てた記録</h3>

      <p className="note">
        いずれも<strong>検査が全部緑のまま通っていた</strong>ものである。
        成績を見ているだけでは出てこない。
      </p>

      {data.discarded.map((d) => (
        <div className="panel" key={d.title} data-testid="discarded-item">
          <p style={{ margin: 0 }}>
            <strong>{d.title}</strong>
            <span className="note">({d.loop})</span>
          </p>
          <p className="note" style={{ marginBottom: 0 }}>
            {d.detail}
          </p>
        </div>
      ))}

      <h3>事前に書いた予測と、その後</h3>

      <div className="figure-scroll">
        <table data-testid="prediction-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>予測</th>
              <th>判定</th>
            </tr>
          </thead>
          <tbody>
            {data.predictions.map((p) => (
              <tr key={p.id}>
                <td>{p.id}</td>
                <td style={{ whiteSpace: "normal", maxWidth: "30rem", textAlign: "left" }}>{p.text}</td>
                <td>
                  <strong>{p.verdict}</strong>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="note">
        二実装照合は {data.verification.two_implementation}。
        実ブラウザ検品 {data.verification.browser_checks} 件(陽性対照つき)。
      </p>
    </section>
  );
}
