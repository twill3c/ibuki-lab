import { readFileSync } from "node:fs";
import { join } from "node:path";

import BreathingEarth from "./BreathingEarth";
import RisingBreath from "./RisingBreath";
import GuessLatitude, { type Screen3 } from "./GuessLatitude";
import Dissection, { type Screen4 } from "./Dissection";
import type { Screen1 } from "../lib/breathing";
import type { Screen2 } from "../lib/rising";
import type { Weights } from "../model/forward";

// 静的書き出しなので、データはビルド時に読んで埋め込む。
// 実行時に外へ取りに行かない(N-01)。
function load<T>(name: string): T {
  const path = join(process.cwd(), "public", "data", name);
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

export default function Page() {
  const screen1 = load<Screen1>("screen1.json");
  const screen2 = load<Screen2>("screen2.json");
  const screen3 = load<Screen3>("screen3.json");
  const screen4 = load<Screen4>("screen4.json");
  const model = JSON.parse(
    readFileSync(join(process.cwd(), "data", "model_weights.json"), "utf8"),
  ) as Weights & { metadata: { generalisation_estimate_mae: number } };

  return (
    <main>
      <h1>息吹ラボ</h1>
      <p className="lede">
        地球は年に一度、息をしている。北ほど深く、南では逆向きに。
      </p>
      <p>
        大気中の二酸化炭素は毎年きまった形で上下する。北半球の陸上植生が夏に吸い、冬に吐くからで、
        その振れ幅は北極圏で十数 ppm、赤道でわずか、南極ではほとんど消える。
        ここでは <strong>NOAA の地上フラスコ網が 1968 年から集めてきた生の測定結果</strong>だけを使って、
        その脈を絵にする。
      </p>

      <BreathingEarth data={screen1} />

      <RisingBreath data={screen2} />

      <GuessLatitude
        data={screen3}
        weights={model}
        generalisationMae={model.metadata.generalisation_estimate_mae}
      />

      <Dissection data={screen4} />

      <footer className="fleet">
        <p>
          出典: NOAA Global Monitoring Laboratory, Carbon Cycle Cooperative Global Air Sampling
          Network(CC0 1.0 /{" "}
          <a href="https://doi.org/10.15138/wkgj-f215">doi:10.15138/wkgj-f215</a>)。
          データは 2026-07-17 版を 2026-09-05 に取得したもの。
        </p>
        <p>
          非商用・広告なし。ディープラーニングの実装訓練として作っている。
          解析と検査の記録は SPEC.md にある。
        </p>
      </footer>

      {/* フリート共通フッタ(koho-lens 準拠 5 項目・この並び・下部固定) */}
      <nav className="fleet-footer" aria-label="フリート共通フッタ">
        <a href="https://github.com/twill3c/ibuki-lab/blob/main/LICENSE">MIT License</a> © 2026 坂田哲朗
        <span className="sep" aria-hidden="true">・</span>
        <a href="https://github.com/twill3c/ibuki-lab">GitHub</a>
        <span className="sep" aria-hidden="true">・</span>
        <a href="https://claude.ai/code/artifact/f2d07c3f-9cef-4095-a7cd-8f45abb2caa3">息吹ラボの歩き方</a>
        <span className="sep" aria-hidden="true">・</span>
        <a href="https://claude.ai/code/artifact/377ec55b-4315-4421-ba4a-a603400142ab">設計図</a>
        <span className="sep" aria-hidden="true">・</span>
        <a href="https://app-menu-amber.vercel.app/">App Menu</a>
      </nav>
    </main>
  );
}
