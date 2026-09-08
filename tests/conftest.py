"""検査全体の前提。

JAX の 64 bit を有効にする。**学習の精度は変わらない** —— `model.init_params` と
`model.to_arrays` は float32 を明示して作るので、既定 dtype が動いても中身は動かない。
有効にするのは、勾配の有限差分照合を float64 で行うためである(T-025)。
float32 のままでは中心差分の丸めが相対 1e-3 程度になり、
**二つの経路が理論上一致しえない**閾値を掲げることになる(HC-073)。
"""
import jax

jax.config.update("jax_enable_x64", True)
