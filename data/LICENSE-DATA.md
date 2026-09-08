# データの出所と権利 — ibuki-lab

## SRC-CO2 / SRC-CH4 — NOAA GML 地上フラスコ網

**ライセンス: CC0 1.0 Universal Public Domain Dedication。**
README(`README_co2_surface-flask_ccgg.html` §3 LICENSE)の記載:

> These data were produced by NOAA and are not subject to copyright protection in the
> United States. NOAA waives any potential copyright and related rights in these data
> worldwide through the Creative Commons Zero 1.0 Universal Public Domain Dedication (CC0 1.0)

CC0 は帰属を要求しないが、README §2.1 が引用を求めているので出典として掲示する。

### 引用(README §2.1 の指定どおり)

> Lan, X., G. Petron, K. Baugh, A.M. Crotwell, M.J. Crotwell, S. DeVogel, M. Madronich,
> J. Mauss, T. Mefford, E. Moglia, S. Morris, J.W. Mund, A. Searle and J. Miller (2026),
> Atmospheric Carbon Dioxide Dry Air Mole Fractions from the NOAA GML Global Greenhouse Gas
> Reference Network, Carbon Cycle Cooperative Global Air Sampling Network: 1967 - Present,
> Version: 2026-07-17, https://doi.org/10.15138/wkgj-f215

### 取得したファイル(2026-09-05 取得 / サーバ側 Last-Modified 2026-07-17)

| ファイル | SHA-256 |
|---|---|
| `co2_surface-flask_ccgg_text.tar.gz` | `f2b1f2c9b453f82641f222ac9f00c6d52449e7730551226322c805877a00c594` |
| `ch4_surface-flask_ccgg_text.tar.gz` | `834ec7bd7354fc74c55565d68893b5bd1328d50c1478223360d0a3148f26ddb3` |

取得元:

- https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/co2_surface-flask_ccgg_text.tar.gz
- https://gml.noaa.gov/aftp/data/trace_gases/ch4/flask/surface/ch4_surface-flask_ccgg_text.tar.gz
- README: https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/README_co2_surface-flask_ccgg.html

NOAA の警告(README §4)もそのまま引き受ける —— 標準ガスの再校正等により
データは改訂されうる。本プロジェクトは**上表の版に対する測定**であり、
版が動けば測り直す。

## SRC-JMA — 気象庁 温室効果ガス月平均値(L5 で取得予定)

- 綾里 / 南鳥島 / 与那国島 の CO2 月平均値 CSV
- 実在を確認済み(2026-09-05・HTTP 200):
  https://www.data.jma.go.jp/ghg/kanshi/obs/co2_monthave_ryo.csv
- 気象庁ホームページの利用規約(政府標準利用規約準拠)に従い、出典を明記して利用する
- **この系統は外部ホールドアウトである**(G-11)。学習・検証・閾値調整のどの段階でも見ない

## 配布物に何を含めるか

`data/raw/` の tar と展開物は**リポジトリに含めない**(`.gitignore`)。
再現手順は README に置き、上表の SHA-256 で同一性を確かめる。
公開するのは、そこから導いた集計と模型の重みだけである。

### 気象庁 CSV の取得(2026-09-09 実施)

```bash
mkdir -p data/raw/jma
for s in ryo mnm yon; do
  curl -o "data/raw/jma/co2_monthave_$s.csv" \
    "https://www.data.jma.go.jp/ghg/kanshi/obs/co2_monthave_$s.csv"
done
```

文字コードは cp932。座標は気象庁の観測地点一覧
(<https://www.data.jma.go.jp/env/ghg_obs/station/>)の度分表記から取り、
`pipeline/jma.py` が度分と小数の両方を持って突合する。

| 地点 | 緯度 | 経度 | 観測期間 |
|---|---|---|---|
| 綾里 | 39°02'N | 141°49'E | 1987- |
| 南鳥島 | 24°17'N | 153°59'E | 1993- |
| 与那国島 | 24°28'N | 123°01'E | 1997-2024/3(終了) |
