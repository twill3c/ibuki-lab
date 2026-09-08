import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "息吹ラボ — 地球の呼吸を測る",
  description:
    "大気中の二酸化炭素の年周曲線から、その空気が地球のどの緯度で採られたかを当てる。" +
    "NOAA GML 地上フラスコ網(CC0 1.0)の生の測定結果から組んだ、非商用の実装訓練。",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
