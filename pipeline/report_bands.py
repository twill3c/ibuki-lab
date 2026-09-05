"""船舶帯の検算が実際に何を突き合わせたかを表に出す(HC-071 の「起きた証拠」)。

`derive_band_latitude` は食い違えば例外で止まるので、通ったこと自体が証拠ではあるが、
どれだけの標本で・どれだけの差だったかを残さないと、後から緩んでも気づけない。
"""
from __future__ import annotations

from pipeline import ingest


def rows(gas: str = "co2"):
    out = []
    for site in ingest.build_sites(gas):
        if site.lat_source != "band":
            continue
        nominal = ingest.band_nominal_latitude(site.code)
        half = ingest.band_half_width(site.parent, gas)
        samples = [
            s
            for s in ingest.read_event_samples(ingest.event_path(gas, site.parent))
            if ingest.is_retained(s) and abs(s.latitude - site.latitude) <= half
        ]
        out.append(
            {
                "code": site.code,
                "parent": site.parent,
                "half_width": half,
                "nominal": nominal,
                "observed": site.latitude,
                "delta": abs(site.latitude - nominal),
                "samples": len(samples),
            }
        )
    return sorted(out, key=lambda r: (r["parent"], -r["nominal"]))


def main() -> int:
    data = rows()
    print(f"{'帯':>8} {'親':>4} {'半値幅':>7} {'符号名':>7} {'標本中央値':>10} {'差':>7} {'標本数':>7}")
    for r in data:
        print(
            f"{r['code']:>8} {r['parent']:>4} {r['half_width']:>7.2f} {r['nominal']:>7.1f} "
            f"{r['observed']:>10.3f} {r['delta']:>7.3f} {r['samples']:>7d}"
        )
    print()
    print(f"帯 {len(data)} 本 / 最大差 {max(r['delta'] for r in data):.3f} 度 / "
          f"標本合計 {sum(r['samples'] for r in data)} / "
          f"最小標本数 {min(r['samples'] for r in data)}")

    print()
    print("照合が恒等式でないことの証拠 —— データから導いた目標点と、符号名から導いた緯度の差集合:")
    for parent in sorted({r["parent"] for r in data}):
        marks = set(ingest.observed_marks("co2", parent))
        nominals = {r["nominal"] for r in data if r["parent"] == parent}
        print(f"  {parent}: 観測 {len(marks)} 点 / 符号名 {len(nominals)} 点 / "
              f"月次ファイルの無い目標点 {sorted(marks - nominals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
