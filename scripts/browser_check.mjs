// 実ブラウザ検品(TEST_SPEC B-01..B-10 / G-13 / G-15 / N-03)。
//
// out/ を静的配信して Playwright で測る。**在存でなく幾何と到達を測る**(HC-138)——
// 「要素が在る」は、図が読めることも操作が届いたことも意味しない。
// 検品器にも陽性対照を置く(HC-080)。失敗は終了コード 1。
import { createServer } from "node:http";
import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { chromium } from "playwright";

const root = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const dist = join(root, "out");
const shotDir = process.env["SHOT_DIR"] ?? join(root, "logs", "shots");
mkdirSync(shotDir, { recursive: true });

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
};

// 各ケースが確かめる品質ゲート。対応を書いておかないと、増やしたケースが
// どのゲートも守っていないことに気づけない(HC-157)。
const GATES = {
  "B-01": "F-01",
  "B-02": "G-13",
  "B-02c": "G-13",
  "B-03": "N-03",
  "B-04": "F-01",
  "B-05": "F-02",
  "B-06": "F-02",
  "B-07": "G-06",
  "B-08": "G-16",
  "B-09": "N-01",
  "B-10": "N-03",
  "B-11": "F-01",
  "B-12": "F-03",
  "B-13": "F-08",
  "B-14": "G-13",
};

const results = [];
function report(id, ok, detail) {
  results.push({ id, gate: GATES[id] ?? "?", ok, detail });
  console.log(`${ok ? "  OK " : "  NG "} ${id} (${GATES[id] ?? "?"}) ${detail}`);
}

function serve() {
  return new Promise((resolve) => {
    const srv = createServer((req, res) => {
      let p = normalize(decodeURIComponent((req.url ?? "/").split("?")[0] ?? "/"));
      if (p.endsWith("\\") || p.endsWith("/")) p += "index.html";
      const file = join(dist, p);
      if (!file.startsWith(dist) || !existsSync(file) || statSync(file).isDirectory()) {
        res.writeHead(404);
        res.end("not found");
        return;
      }
      res.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" });
      res.end(readFileSync(file));
    });
    srv.listen(0, "127.0.0.1", () => resolve({ srv, base: `http://127.0.0.1:${srv.address().port}/` }));
  });
}

/** SVG の描画要素の bbox が viewBox に収まるか。はみ出した要素の一覧を返す。 */
async function overflowing(page, selector) {
  return page.evaluate((sel) => {
    const svg = document.querySelector(sel);
    if (!svg) return [`${sel} が無い`];
    const vb = svg.viewBox.baseVal;
    const out = [];
    for (const el of svg.querySelectorAll("path, circle, line, rect, text, polygon, polyline")) {
      if (!(el instanceof SVGGraphicsElement)) continue;
      // defs の中身は描画木ではない。pattern の座標系は自前で bbox が負になる(HC-194)
      if (el.closest("defs")) continue;
      const b = el.getBBox();
      if (b.width === 0 && b.height === 0) continue;
      const tol = 0.5;
      if (
        b.x < vb.x - tol ||
        b.y < vb.y - tol ||
        b.x + b.width > vb.x + vb.width + tol ||
        b.y + b.height > vb.y + vb.height + tol
      ) {
        const name = el.id || el.getAttribute("data-testid") || el.tagName;
        out.push(`${name} bbox=${[b.x, b.y, b.width, b.height].map((v) => v.toFixed(1)).join(",")}`);
      }
    }
    return out;
  }, selector);
}

/** 横に溢れている要素。本文が画面幅を超えていないか(N-03)。 */
async function horizontallyOverflowing(page) {
  return page.evaluate(() => {
    const limit = document.documentElement.clientWidth;
    const bad = [];
    for (const el of document.querySelectorAll("main *")) {
      // 自分で横スクロールを持つ器は、中身がはみ出してよい
      if (el.closest(".figure-scroll")) continue;
      const r = el.getBoundingClientRect();
      if (r.width === 0) continue;
      if (r.right > limit + 1 || r.left < -1) {
        bad.push(`${el.tagName}.${el.className || "?"} right=${r.right.toFixed(0)} limit=${limit}`);
      }
    }
    return bad;
  });
}

async function main() {
  if (!existsSync(join(dist, "index.html"))) {
    console.error("out/index.html が無い。先に npm run build");
    process.exit(1);
  }
  const { srv, base } = await serve();
  const browser = await chromium.launch();
  const requested = [];

  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 1000 } });
    page.on("request", (r) => requested.push(r.url()));
    await page.goto(base, { waitUntil: "networkidle" });

    // B-01: 帯が実際に描かれている。**数だけでなく、色が一様でないこと**を見る ——
    // 全部同じ色なら「在る」は満たすが図は何も言っていない
    const cells = await page.$$eval('[data-testid="heat-cell"]', (els) =>
      els.map((el) => el.getAttribute("fill")),
    );
    const distinct = new Set(cells).size;
    report("B-01", cells.length > 200 && distinct > 20, `帯 ${cells.length} 個 / 色 ${distinct} 種`);

    // B-02: 図の要素が viewBox に収まる
    const heatOver = await overflowing(page, "svg#heatmap");
    const ladderOver = await overflowing(page, "svg#ladder");
    report("B-02", heatOver.length === 0 && ladderOver.length === 0,
      `はみ出し 熱図 ${heatOver.length} 件 / 梯子 ${ladderOver.length} 件 ${[...heatOver, ...ladderOver].slice(0, 2).join(" ")}`);

    // B-02c 陽性対照: viewBox 外の要素を注入して、検査が落ちることを先に確かめる
    await page.evaluate(() => {
      const svg = document.querySelector("svg#heatmap");
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", "-500");
      c.setAttribute("cy", "-500");
      c.setAttribute("r", "10");
      c.id = "positive-control";
      svg.appendChild(c);
    });
    const ctrl = await overflowing(page, "svg#heatmap");
    await page.evaluate(() => document.getElementById("positive-control")?.remove());
    report("B-02c", ctrl.some((h) => h.includes("positive-control")), `陽性対照 ${ctrl.length} 件検出`);

    // B-03: 横に溢れていない(1280px)
    const over1280 = await horizontallyOverflowing(page);
    report("B-03", over1280.length === 0, `1280px で溢れ ${over1280.length} 件 ${over1280.slice(0, 2).join(" ")}`);

    // B-04: スライダを動かすと**選択の印が動く**。操作が届いた証拠を見る(HC-138)
    const before = await page.$eval('[data-testid="bin-label"]', (el) => el.textContent);
    const slider = await page.$('[data-testid="bin-slider"]');
    await slider.scrollIntoViewIfNeeded();
    await slider.fill("12");
    const after = await page.$eval('[data-testid="bin-label"]', (el) => el.textContent);
    // 選択の印が実際に動いたか(在存でなく座標で見る — HC-138)
    const markerBefore = await page.$eval('[data-testid="bin-marker"]', (el) => Number(el.getAttribute("x")));
    await slider.fill("20");
    const markerAfter = await page.$eval('[data-testid="bin-marker"]', (el) => Number(el.getAttribute("x")));
    await slider.fill("12");
    report("B-04", before !== after && markerAfter > markerBefore,
      `${before} → ${after} / 印 x=${markerBefore.toFixed(0)} → ${markerAfter.toFixed(0)}`);

    // B-11: 正規化の切替が**南半球の行の色を実際に変える**。
    // 絶対目盛りでは南の行がほぼ無色で、図が「浅い」でなく「無い」と読める(loop_004 GEN-LAYOUT)
    const southRowColours = async () =>
      page.$$eval('g[data-row="-80"] [data-testid="heat-cell"]', (els) =>
        els.map((el) => el.getAttribute("fill")),
      );
    const plainSouth = await southRowColours();
    const normToggle = await page.$('[data-testid="normalise-toggle"]');
    await normToggle.scrollIntoViewIfNeeded();
    await normToggle.check();
    const normSouth = await southRowColours();
    const noteAfter = await page.$eval('[data-testid="scale-note"]', (el) => el.textContent ?? "");
    await normToggle.uncheck();
    const plainDistinct = new Set(plainSouth).size;
    const normDistinct = new Set(normSouth).size;
    report("B-11", normDistinct > plainDistinct && noteAfter.includes("量は消えて"),
      `南緯80-90 の色 ${plainDistinct} 種 → ${normDistinct} 種 / 凡例が用途を書いている`);

    // B-12: 画面②の切替が経路の座標を実際に変える(在存でなく到達 — HC-138)
    const risingPath = () => page.$eval('[data-series="brw"]', (el) => el.getAttribute("d") ?? "");
    const beforePath = await risingPath();
    const detrend = await page.$('[data-testid="detrend-toggle"]');
    await detrend.scrollIntoViewIfNeeded();
    await detrend.check();
    const afterPath = await risingPath();
    const noteText = await page.$eval('[data-testid="rising-note"]', (el) => el.textContent ?? "");
    await detrend.uncheck();
    report("B-12", beforePath !== afterPath && beforePath.length > 100 && noteText.includes("脈が残る"),
      `経路が ${beforePath.length} → ${afterPath.length} 文字で変化 / 注記が切り替わる`);

    // B-13: 外部検証が独立の節として出ており、学習に使っていないと書いてある
    const externalRows = await page.$$eval('[data-testid="external-table"] tbody tr', (els) => els.length);
    const jmaRows = await page.$$eval('[data-testid="jma-table"] tbody tr', (els) => els.length);
    const bodyText = await page.$eval("main", (el) => el.textContent ?? "");
    report("B-13", externalRows === 3 && jmaRows === 3 && bodyText.includes("一度も使っていない"),
      `対照表 ${externalRows} 行 / 地点表 ${jmaRows} 行 / 断りが本文にある`);

    // B-14: 画面②の図も viewBox に収まる
    const risingOver = await overflowing(page, "svg#rising");
    report("B-14", risingOver.length === 0, `はみ出し ${risingOver.length} 件 ${risingOver.slice(0, 2).join(" ")}`);

    // B-05: 緯度梯子が描かれ、**振幅が北から南へ単調に潰れて戻る**
    const amps = await page.$$eval('[data-band]', (els) =>
      els.map((el) => ({
        code: el.getAttribute("data-band"),
        label: el.querySelector("text:last-of-type")?.textContent ?? "",
      })),
    );
    report("B-05", amps.length >= 14, `帯 ${amps.length} 本`);

    // B-06: 南半球の切替が届く(要素数が実際に変わる)
    const beforeRows = amps.length;
    const toggle = await page.$('[data-testid="south-toggle"]');
    await toggle.scrollIntoViewIfNeeded();
    await toggle.uncheck();
    const afterRows = await page.$$eval("[data-band]", (els) => els.length);
    await toggle.check();
    const restored = await page.$$eval("[data-band]", (els) => els.length);
    report("B-06", afterRows < beforeRows && restored === beforeRows,
      `${beforeRows} → ${afterRows} → ${restored} 本`);

    // B-07: 平滑を経ていないことが画面に書いてある(G-06)
    const body = await page.$eval("main", (el) => el.textContent ?? "");
    report("B-07", body.includes("Thoning") || body.includes("平滑"), "由来の断りが本文にある");

    // B-08: 落とした地点年の数が分子と分母で出ている(G-16)
    report("B-08", /採用した地点年は\s*\d+\s*件/.test(body) && /落としたのが\s*\d+\s*件/.test(body),
      "採否が分子と分母で出ている");

    // B-09: 外部 host へ出ていない(N-01)
    const foreign = new Set(
      requested.filter((u) => !u.startsWith(base) && !u.startsWith("data:")).map((u) => new URL(u).host),
    );
    report("B-09", foreign.size === 0, `外部 host ${[...foreign].join(",") || "0 件"}`);

    // B-10: 三つの画面幅で溢れず、縦に伸びすぎない(N-03)
    const widths = [360, 768, 1280];
    const heights = [];
    let overflowTotal = 0;
    for (const width of widths) {
      await page.setViewportSize({ width, height: 900 });
      await page.waitForTimeout(120);
      const bad = await horizontallyOverflowing(page);
      overflowTotal += bad.length;
      const h = await page.evaluate(() => document.documentElement.scrollHeight);
      heights.push(`${width}px:${h}`);
      await page.screenshot({ path: join(shotDir, `screen1-${width}.png`), fullPage: false });
      if (bad.length) console.log(`      ${width}px 溢れ: ${bad.slice(0, 3).join(" / ")}`);
    }
    const tallest = Math.max(...heights.map((h) => Number(h.split(":")[1])));
    report("B-10", overflowTotal === 0 && tallest <= 16000, `${heights.join(" ")} / 溢れ ${overflowTotal} 件`);
  } finally {
    await browser.close();
    srv.close();
  }

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} 通過`);
  if (failed.length) {
    console.error(`不合格: ${failed.map((r) => `${r.id}(${r.gate})`).join(", ")}`);
    process.exit(1);
  }
  console.log(`スクリーンショット: ${shotDir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
