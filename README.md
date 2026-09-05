# 息吹ラボ(ibuki-lab)

**一本の年周曲線を見せて、その空気が地球のどの緯度で採られたかを当てる。**

大気中の二酸化炭素は年に一度大きく上下する。北半球の陸上植生が夏に吸い、冬に吐くからで、
その振れ幅は北緯 70 度帯で十数 ppm、赤道付近で数 ppm、南極ではほとんど消えて位相が逆になる。
**地球は年に一度、息をしている。北ほど深く、南では逆向きに。**

本プロジェクトはこの年周曲線を素材にニューラルネットを組み、**モデルが一年のどこを見て
答えを出したか**を曲線の上に塗って見せる。ディープラーニングの実装訓練であり、
非商用・広告なしで公開する。

正解ラベル(緯度)は観測所の所在地という物理的事実で、NOAA のデータファイルのヘッダに
書かれている。**オラクルが自作でない**という条件が素材の側で解けている。

詳細は [SPEC.md](SPEC.md)、検査の方針は [TEST_SPEC.md](TEST_SPEC.md)、
データの出所と権利は [data/LICENSE-DATA.md](data/LICENSE-DATA.md) にある。

## 素材(実測 2026-09-05)

| 区分 | 件数 |
|---|---|
| NOAA GML 地上フラスコ網 CO2 月次ファイル | 95 |
| うち固定地点(緯度がヘッダにある) | 74 |
| うち船舶帯(緯度を観測から導いた) | 21 |
| CO2 event 標本(棄却フラグ除去後) | 217,355 |

緯度の範囲は **−89.98 度(南極)〜 +82.45 度(Alert)**。極から極まで並ぶ。
正本は [data/census.json](data/census.json) で、SPEC の数値とは検査 T-009 で突き合わせる。

## 手元で回す

生データはリポジトリに含めない。次の手順で取得する。

```bash
mkdir -p data/raw/ext
curl -o data/raw/co2_surface-flask_ccgg_text.tar.gz \
  https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/co2_surface-flask_ccgg_text.tar.gz
curl -o data/raw/ch4_surface-flask_ccgg_text.tar.gz \
  https://gml.noaa.gov/aftp/data/trace_gases/ch4/flask/surface/ch4_surface-flask_ccgg_text.tar.gz
sha256sum data/raw/*.tar.gz          # data/LICENSE-DATA.md の記録と突き合わせる
tar -xzf data/raw/co2_surface-flask_ccgg_text.tar.gz -C data/raw/ext
tar -xzf data/raw/ch4_surface-flask_ccgg_text.tar.gz -C data/raw/ext

python pipeline/ingest.py            # data/census.json を作り直す
python -m pipeline.report_bands      # 船舶帯の検算の内訳を出す
python -m pytest                     # 検査
```

NOAA のデータは標準ガスの再校正等により改訂されうる。上表は
`data/LICENSE-DATA.md` に記録した版(サーバ側 Last-Modified 2026-07-17)に対する実測である。

## 進み方

7 段階ループプロトコルで進める。各ループの記録は
[logs/loops/](logs/loops/) に追記専用で残る。計画は SPEC §9。

## 出典

- NOAA Global Monitoring Laboratory, Carbon Cycle Cooperative Global Air Sampling Network
  (CC0 1.0 / <https://doi.org/10.15138/wkgj-f215>)
- 気象庁 温室効果ガス観測(L5 で取り込み予定。**外部ホールドアウトとして学習には使わない**)

## ライセンス

コードは [LICENSE](LICENSE) による。データの権利は
[data/LICENSE-DATA.md](data/LICENSE-DATA.md) に別記する。
