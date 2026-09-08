import type { NextConfig } from "next";

// 静的書き出しのみ。サーバ関数を一つも持たない(SPEC N-01)。
// データは public/data/ から同一オリジンで配る —— ビルド時にも実行時にも外へ取りに行かない。
// 出て行くのは集計と重みだけで、生データは配らない(data/raw は .gitignore)。
const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
};

export default nextConfig;
