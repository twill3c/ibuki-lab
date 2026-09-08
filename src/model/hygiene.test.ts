/**
 * 出荷経路の衛生(TEST_SPEC T-041 / G-01)。
 *
 * わざと壊した実装(`variants.ts`)は照合の陽性対照にだけ使う。出荷される画面が
 * うっかりこれを読んでいたら、利用者に壊れた前向きが出る。**検査だけが緑でも意味が無い。**
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const SRC = fileURLToPath(new URL("../", import.meta.url));

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}

describe("出荷経路", () => {
  const files = walk(SRC).filter((p) => p.endsWith(".ts"));
  const shipping = files.filter((p) => !p.endsWith(".test.ts"));

  it("走査対象が空でない", () => {
    expect(shipping.length).toBeGreaterThan(0);
    expect(files.length).toBeGreaterThan(shipping.length);
  });

  it("出荷されるコードが variants.ts を読んでいない", () => {
    const offenders = shipping.filter(
      (path) =>
        !path.endsWith("variants.ts") &&
        /from\s+["'][^"']*variants(\.js)?["']/.test(readFileSync(path, "utf8")),
    );
    expect(offenders).toEqual([]);
  });

  it("陽性対照: 検査の正規表現が実際に import を捕まえる", () => {
    const sample = 'import { forwardVariant } from "./variants.js";';
    expect(/from\s+["'][^"']*variants(\.js)?["']/.test(sample)).toBe(true);
    expect(/from\s+["'][^"']*variants(\.js)?["']/.test('import { x } from "./forward.js";')).toBe(
      false,
    );
  });

  it("前向きが外部ライブラリを読んでいない(N-02: ランタイム依存ゼロ)", () => {
    const forward = readFileSync(join(SRC, "model", "forward.ts"), "utf8");
    const imports = [...forward.matchAll(/from\s+["']([^"']+)["']/g)].map((m) => m[1] as string);
    expect(imports.filter((s) => !s.startsWith("."))).toEqual([]);
  });
});
