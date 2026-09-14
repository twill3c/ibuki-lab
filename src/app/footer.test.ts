// フリート共通フッタの行き先(TEST_SPEC B-24 の静的な半分)。
// 「どれかのリンクが github.com を向いている」では足りない(HC-098)。**項目ごとの行き先**を見る。
// 仮の https://github.com/ や app-menu.vercel.app(他者のサイト)を置いたまま公開した前例がある(HC-271)。
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const page = readFileSync(join(__dirname, "page.tsx"), "utf-8");
const REPO = "https://github.com/twill3c/ibuki-lab";
const ARTIFACT = /^https:\/\/claude\.ai\/code\/artifact\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

function hrefOf(label: string): string | null {
  const m = page.match(new RegExp(`<a href="([^"]+)"[^>]*>${label}</a>`));
  return m?.[1] ?? null;
}

describe("フリート共通フッタ", () => {
  it("MIT License はこのリポジトリの LICENSE を指す", () => {
    expect(hrefOf("MIT License")).toBe(`${REPO}/blob/main/LICENSE`);
  });
  it("GitHub はこのリポジトリを指す(ドメインだけの仮置きではない)", () => {
    expect(hrefOf("GitHub")).toBe(REPO);
  });
  it("歩き方と設計図は実在のアーティファクトを指す", () => {
    expect(hrefOf("息吹ラボの歩き方")).toMatch(ARTIFACT);
    expect(hrefOf("設計図")).toMatch(ARTIFACT);
    expect(hrefOf("息吹ラボの歩き方")).not.toBe(hrefOf("設計図"));
  });
  it("App Menu は app-menu-amber を指す(app-menu.vercel.app は他者のサイト)", () => {
    expect(hrefOf("App Menu")).toBe("https://app-menu-amber.vercel.app/");
    expect(page).not.toContain("app-menu.vercel.app");
  });
  it("5 項目がこの並びで出る", () => {
    const order = [">MIT License<", "© 2026 坂田哲朗", ">GitHub<", ">息吹ラボの歩き方<", ">設計図<", ">App Menu<"];
    const at = order.map((s) => page.indexOf(s));
    expect(at.every((i) => i >= 0)).toBe(true);
    expect([...at].sort((a, b) => a - b)).toEqual(at);
  });
});
