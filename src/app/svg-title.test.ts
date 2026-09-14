// SVG の <title> の子は式一つ(文字列一本)であること。
// JSX で文言と式を並べると子のテキスト節点が分かれ、React 19 がサーバの HTML と食い違って
// 描画をやり直す(React #418)。loop_008 で本番検品が初めて見つけた。手元の実ブラウザ検品は
// 幾何と到達しか見ておらず、25 件とも緑のままだった。
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

/** <title>…</title> の中身のうち、式一つでないものを返す。 */
export function splitTitles(source: string): string[] {
  const bad: string[] = [];
  for (const m of source.matchAll(/<title>([\s\S]*?)<\/title>/g)) {
    const inner = (m[1] ?? "").trim();
    // 式一つ: 先頭が { で末尾が } で、その外に文字が無い。中の { } の釣り合いで一本かを見る
    let depth = 0;
    let closedAt = -1;
    for (let i = 0; i < inner.length; i++) {
      if (inner[i] === "{") depth++;
      if (inner[i] === "}") {
        depth--;
        if (depth === 0) {
          closedAt = i;
          break;
        }
      }
    }
    const single = inner.startsWith("{") && closedAt === inner.length - 1;
    if (!single) bad.push(inner.slice(0, 80));
  }
  return bad;
}

describe("SVG の title は文字列一本", () => {
  const dir = __dirname;
  const files = readdirSync(dir).filter((f) => f.endsWith(".tsx"));

  it("画面のどこにも、子が分かれた title が無い", () => {
    const found = files.flatMap((f) =>
      splitTitles(readFileSync(join(dir, f), "utf-8")).map((s) => `${f}: ${s}`),
    );
    expect(found).toEqual([]);
  });

  it("title を実際に持つ画面を走査している(空振りでない)", () => {
    const withTitle = files.filter((f) => readFileSync(join(dir, f), "utf-8").includes("<title>"));
    expect(withTitle.length).toBeGreaterThanOrEqual(2);
  });

  it("陽性対照: 文言と式を並べた title を落とす", () => {
    expect(splitTitles("<title>\n  {a} / 値 {b} ppm\n</title>")).toHaveLength(1);
    expect(splitTitles("<title>地点 {code}</title>")).toHaveLength(1);
    expect(splitTitles("<title>{`${a} / ${b}`}</title>")).toHaveLength(0);
    expect(splitTitles("<title>\n  {`${a} / ` + (c ? ` (${d})` : \"\")}\n</title>")).toHaveLength(0);
  });
});
