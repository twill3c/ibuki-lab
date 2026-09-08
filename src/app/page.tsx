import { readFileSync } from "node:fs";
import { join } from "node:path";

import BreathingEarth from "./BreathingEarth";
import RisingBreath from "./RisingBreath";
import type { Screen1 } from "../lib/breathing";
import type { Screen2 } from "../lib/rising";

// 静的書き出しなので、データはビルド時に読んで埋め込む。
// 実行時に外へ取りに行かない(N-01)。
function load<T>(name: string): T {
  const path = join(process.cwd(), "public", "data", name);
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

export default function Page() {
  const screen1 = load<Screen1>("screen1.json");
  const screen2 = load<Screen2>("screen2.json");

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
    </main>
  );
}
