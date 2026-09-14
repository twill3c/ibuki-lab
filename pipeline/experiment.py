"""L2 の学習と評価。地点群ばらし leave-one-group-out。

**土俵は L1 と同じ**にする —— 同じ 805 地点年、同じ 57 群、同じ「地点あたり MAE」。
違う土俵で出した数を並べて勝ち負けを言わない(HC-064)。

学習の運びで効いている点:

- **エポック数も学習率も学習側だけで決める。** 外側で抜いた群を除いた中から、さらに
  群単位で内側の検証を切り出し、そこで最良の点を選ぶ。試験側を見て決めれば、
  その数字はもう成績ではない。**学習率の掃引は結果ごと報告に残す** —— 隠れた選択を作らない
- **fold は形でなく重みで分け、`vmap` で束ねて `lax.scan` の中で回す。** 実測 2026-09-06:
  Python で 1 エポックずつ呼ぶと 26 ms の呼び出し費用が支配し、fold ごとに配列を切り出すと
  形が変わるたび jit が作り直されて 1 fold 33 秒になる
- **種を変えて 3 回**測り、平均と幅を出す。1 回の数字を成績と呼ばない
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pipeline import curves, model

REPORT = Path(__file__).resolve().parents[1] / "data" / "model.json"

#: エポックの予算。実測 2026-09-08(57 群・種 0・予算 400)で最良点は lr=0.003 で p90 388、
#: lr=0.01 で p90 378、lr=0.03 で中央値 23。予算 300 では 4 fold が張り付いたので 600 を置く。
#: 所要は畳み込みを行列積に変えて 1 割強縮めてある(HC-213)。
#:
#: **予算に張り付いた fold があること自体は例外にしない。** 種が変われば内側分割も変わるので、
#: 張り付きは追いかけても終わらない。問うべきは「打ち切りが結論を動かすか」であり、
#: それには `budget_sensitivity` が答える(HC-218)。件数は報告に残す。
MAX_EPOCHS = 600
LEARNING_RATE = 1e-2
SEEDS = (0, 1, 2)

#: 学習率の候補。**選定は内側検証の誤差で行う**(試験側の成績を見て選ばない)。
#: 掃引の結果は試験側の数字も含めて報告に残す —— 隠れた選択を作らないため。
LR_CANDIDATES = (3e-3, 1e-2, 3e-2)

#: 内側の検証に回す群の割合(エポック数を選ぶためだけに使う)。
INNER_VAL_FRACTION = 0.25


def masked_mae(params, x, y, weights):
    """重み付きの平均絶対誤差。fold ごとの出入りを**形でなく重みで**表す。

    fold ごとに配列を切り出すと形が毎回変わり、jit がそのつど作り直される
    (実測 2026-09-06: 1 fold あたり 33 秒。57 群 × 3 種 × 3 系統で 4.7 時間)。
    重み 0 で外せば形は一定になり、コンパイルは一度で済む。
    """
    errors = jnp.abs(model.forward(params, x) - y)
    return jnp.sum(errors * weights) / jnp.sum(weights)


def _make_trainer():
    """全 fold を `vmap` で束ね、エポックの繰り返しを `lax.scan` の中で回す。

    返るのは **内側検証が最良だった時点の重み**である。エポック数を選んでから
    学習し直す二段構えにすると、選んだ数ごとに jit が作り直される。
    一度の走査の中で最良点を持ち回れば、コンパイルは一度で済む。

    代わりに、最終的な重みは学習側の 75%(内側検証を除いた分)だけで学んだものになる。
    **試験側は一度も見ていない**ので成績の意味は変わらない。
    """
    tx = optax.adam(LEARNING_RATE)  # 呼ばれた時点の学習率で閉じる(掃引はこの関数を作り直す)

    def one_fold(params, x, y, train_w, val_w):
        state = tx.init(params)
        init_val = masked_mae(params, x, y, val_w)

        def one_epoch(carry, _):
            p, s, best_val, best_p, best_epoch, epoch = carry
            grads = jax.grad(masked_mae)(p, x, y, train_w)
            updates, s = tx.update(grads, s, p)
            p = optax.apply_updates(p, updates)

            val = masked_mae(p, x, y, val_w)
            improved = val < best_val
            best_p = jax.tree.map(lambda new, old: jnp.where(improved, new, old), p, best_p)
            best_val = jnp.where(improved, val, best_val)
            best_epoch = jnp.where(improved, epoch + 1, best_epoch)
            return (p, s, best_val, best_p, best_epoch, epoch + 1), None

        carry = (params, state, init_val, params, jnp.int32(0), jnp.int32(0))
        (_, _, best_val, best_p, best_epoch, _), _ = jax.lax.scan(
            one_epoch, carry, None, length=MAX_EPOCHS
        )
        return best_p, best_val, best_epoch

    return jax.jit(jax.vmap(one_fold, in_axes=(0, None, None, 0, 0)))


def _inner_split(groups, held, seed):
    """外側で抜いた群を除いた中から、群単位で内側の検証を切り出す。"""
    pool = sorted({g for g in groups if g != held})
    rng = np.random.default_rng(20260906 + seed)
    rng.shuffle(pool)
    n_val = max(1, int(round(len(pool) * INNER_VAL_FRACTION)))
    val = set(pool[:n_val])
    return val


def _per_site_mae(examples, indices, preds, labels) -> float:
    by_site: dict[str, list] = {}
    for pos, i in enumerate(indices):
        by_site.setdefault(examples[i].code, []).append(abs(preds[pos] - labels[i]))
    return statistics.mean(statistics.mean(v) for v in by_site.values())


def sweep_learning_rate(examples, x, y, channels: int = 1, seed: int = 0) -> dict:
    """学習率を掃引する。**選定は内側検証の誤差だけで行う。**

    試験側の成績も一緒に記録して報告に載せるが、選ぶのには使わない。
    掃引を隠すと「試験を見て選んだのでは」という疑いが残るので、全部出す。
    """
    global LEARNING_RATE
    original = LEARNING_RATE
    rows = {}
    try:
        for lr in LR_CANDIDATES:
            LEARNING_RATE = lr
            started = time.time()
            result = run_logo(examples, x, y, channels, seed, strict_budget=False)
            row = {
                "val_mae": result["val_mae"],
                "test_mae_per_site": _per_site_mae(
                    examples, range(len(examples)), result["preds"], y
                ),
                "epochs_median": statistics.median(result["epochs"]),
                "epochs_max": max(result["epochs"]),
                "folds_at_budget": result["folds_at_budget"],
                "eligible": result["folds_at_budget"] == 0,
                "seconds": round(time.time() - started),
            }
            rows[f"{lr:g}"] = row
            print(
                f"  掃引 lr={lr:g}: 内側検証 {row['val_mae']:.2f} / "
                f"試験 {row['test_mae_per_site']:.2f} / "
                f"最良点 中央値 {row['epochs_median']:.0f} 最大 {row['epochs_max']} / "
                f"予算張り付き {row['folds_at_budget']} 件"
                f"{'' if row['eligible'] else ' → 失格'}"
            )
    finally:
        LEARNING_RATE = original

    # 予算内に内側検証が折り返さない候補は失格。**試験側は見ない**判定である。
    eligible = {k: v for k, v in rows.items() if v["eligible"]}
    if not eligible:
        raise RuntimeError(
            f"予算 {MAX_EPOCHS} 内に内側検証が折り返す学習率が無い。予算か候補を見直すこと"
        )
    chosen = min(eligible, key=lambda k: eligible[k]["val_mae"])
    return {
        "rows": rows,
        "chosen_lr": float(chosen),
        "chosen_by": "val_mae(予算内に折り返した候補のみ)",
        "seed": seed,
        "max_epochs": MAX_EPOCHS,
    }


def run_logo(
    examples,
    x,
    y,
    channels: int,
    seed: int,
    audit: list | None = None,
    strict_budget: bool = True,
) -> dict:
    """leave-one-group-out を一周する。返り値は例の順に並んだ予測。

    `strict_budget` は呼び出し側が決める(HC-218)。掃引では「予算内に内側検証が
    折り返さない」こと自体が候補の失格理由なので止めない。本測定でも止めない ——
    止めると、止めた理由(打ち切り)が結論に効いているかを永久に測れないからである。
    打ち切りの影響は `budget_sensitivity` が別に測る。
    """
    groups = [ex.group for ex in examples]
    folds = curves.leave_one_group_out(examples)
    n = len(examples)

    train_w = np.zeros((len(folds), n), dtype=np.float32)
    val_w = np.zeros((len(folds), n), dtype=np.float32)
    for f, (train_idx, test_idx) in enumerate(folds):
        held = groups[test_idx[0]]
        val_groups = _inner_split(groups, held, seed)
        for i in train_idx:
            if groups[i] in val_groups:
                val_w[f, i] = 1.0
            else:
                train_w[f, i] = 1.0
        if audit is not None:
            audit.append(
                {
                    "held_out": held,
                    "train_groups": sorted({groups[i] for i in train_idx}),
                    "test_groups": sorted({groups[i] for i in test_idx}),
                    "train_examples_seen": len(train_idx),
                    "test_examples_scored": len(test_idx),
                    "inner_val_groups": len(val_groups),
                }
            )

    # 重みが立っている例と、その fold の試験側は排他でなければならない(G-04)。
    for f, (_, test_idx) in enumerate(folds):
        for i in test_idx:
            assert train_w[f, i] == 0.0 and val_w[f, i] == 0.0, "試験側に重みが乗っている"

    stacked = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *[model.init_params(seed * 1000 + f, channels=channels) for f in range(len(folds))],
    )
    trainer = _make_trainer()
    best_params, best_vals, best_epochs = trainer(
        stacked, jnp.asarray(x), jnp.asarray(y), jnp.asarray(train_w), jnp.asarray(val_w)
    )

    preds = np.full(n, np.nan, dtype=np.float64)
    for f, (_, test_idx) in enumerate(folds):
        fold_params = jax.tree.map(lambda leaf: leaf[f], best_params)
        preds[test_idx] = np.asarray(
            model.forward(fold_params, jnp.asarray(x[test_idx]))
        )

    chosen_epochs = [int(e) for e in np.asarray(best_epochs)]
    at_budget = [e for e in chosen_epochs if e >= MAX_EPOCHS]
    if strict_budget:
        assert not at_budget, (
            f"最良点が予算 {MAX_EPOCHS} に張り付いた fold が {len(at_budget)} 件ある —— "
            "この成績は打ち切りの産物である"
        )
    if audit is not None:
        for entry, epoch in zip(audit, chosen_epochs):
            entry["epochs"] = epoch

    assert not np.isnan(preds).any(), "予測されていない例がある"
    return {
        "preds": preds,
        "epochs": chosen_epochs,
        "val_mae": float(np.mean(np.asarray(best_vals))),
        "folds_at_budget": len(at_budget),
    }


def evaluate(examples, x, y, channels: int, label: str, keep_audit: bool = False) -> dict:
    """種を変えて 3 回まわし、平均と幅を出す。

    **予算の張り付きは例外にしない。** 張り付きが結論を動かすかどうかは、
    件数ではなく `budget_sensitivity` が答える(SPEC §7.7 / HC-218)。
    ここで止めると、止めた理由(打ち切り)が結論に効いているかを永久に測れない。
    """
    maes, per_example, epochs_all = [], [], []
    at_budget = 0
    audit = [] if keep_audit else None
    for seed in SEEDS:
        started = time.time()
        result = run_logo(
            examples, x, y, channels, seed,
            audit if seed == SEEDS[0] else None,
            strict_budget=False,
        )
        at_budget += result["folds_at_budget"]
        preds = result["preds"]
        maes.append(_per_site_mae(examples, range(len(examples)), preds, y))
        per_example.append(float(np.mean(np.abs(preds - y))))
        epochs_all.extend(result["epochs"])
        print(
            f"  {label} seed={seed}: 地点MAE {maes[-1]:.2f} / 例MAE {per_example[-1]:.2f} "
            f"({time.time() - started:.0f} 秒)"
        )

    out = {
        "metric": "mae_per_site",
        "mae": statistics.mean(maes),
        "mae_per_seed": maes,
        "mae_spread": max(maes) - min(maes),
        "mae_per_example": statistics.mean(per_example),
        "channels": channels,
        "params": model.count_params(model.init_params(0, channels=channels)),
        "epochs_median": statistics.median(epochs_all),
        "epochs_max": max(epochs_all),
        "max_epochs_budget": MAX_EPOCHS,
        "folds_at_budget": at_budget,
        "folds_total": len(SEEDS) * len({ex.group for ex in examples}),
        "n_examples": len(examples),
        "n_sites": len({ex.code for ex in examples}),
        "n_groups": len({ex.group for ex in examples}),
    }
    if keep_audit:
        out["_audit"] = audit
    return out


def measure_unit(examples, x, y, channels: int) -> float:
    """本番規模に入る前に**一単位**(scan 内の 1 エポック)を測り、総所要を掛け算で出す。

    HC-207 / HC-213 の規範をコードの側に置く。**測ったのは実装であって仕事ではない**ので、
    実装を触ったら必ずここを通る。遅さはテストの赤として現れないから、
    走らせる前に数字を見る道をコードに埋めておく。
    """
    groups = [ex.group for ex in examples]
    folds = curves.leave_one_group_out(examples)
    train_w = np.zeros((len(folds), len(examples)), dtype=np.float32)
    for f, (train_idx, _) in enumerate(folds):
        for i in train_idx:
            train_w[f, i] = 1.0
    stacked = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *[model.init_params(f, channels=channels) for f in range(len(folds))],
    )
    grad = jax.jit(jax.vmap(jax.grad(masked_mae), in_axes=(0, None, None, 0)))
    xj, yj, wj = jnp.asarray(x), jnp.asarray(y), jnp.asarray(train_w)

    jax.block_until_ready(grad(stacked, xj, yj, wj)["conv1_w"])  # コンパイルを外に出す
    started = time.time()
    for _ in range(5):
        out = grad(stacked, xj, yj, wj)
    jax.block_until_ready(out["conv1_w"])
    return (time.time() - started) / 5


#: 予算感度の判定閾値(度)。相手との差(L1 実測で 12.0 − 10.95 = 1.05 度)の
#: 3 分の 1 未満なら、打ち切りは結論を作っていないと言える。**測定前に決めて動かさない。**
BUDGET_SENSITIVITY_MARGIN = 0.35


def _jsonable(value):
    """numpy の型を Python の型へ落とす。

    `json.dumps` は numpy.bool_ / numpy.float64 を書けない。書き出しは main の最後に
    一度だけ来るので、ここで落とし損ねると**計算がすべて終わってから**落ちる
    (loop_002 GEN-LOGIC: 14 走ぶん 2 時間を失った)。
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_report(report: dict) -> None:
    """報告を書く。**走り出す前に一度呼んで書き出し経路を通しておく**こと。"""
    REPORT.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def budget_sensitivity(examples, x, y, channels: int = 1, seed: int = 0, factor: int = 3) -> dict:
    """予算を `factor` 倍にして測り直し、成績がどれだけ動くかを出す。

    打ち切りは模型の成績を**下振れさせる**ので、「模型が 2 特徴に負けた」という
    結論に対しては都合の良い側に働く。打ち切りが負けを作っていないことを示さないと、
    その結論は主張できない(HC-218)。
    """
    global MAX_EPOCHS
    original = MAX_EPOCHS
    out = {}
    try:
        for budget in (original, original * factor):
            MAX_EPOCHS = budget
            started = time.time()
            result = run_logo(examples, x, y, channels, seed, strict_budget=False)
            out[str(budget)] = {
                "mae": _per_site_mae(examples, range(len(examples)), result["preds"], y),
                "folds_at_budget": result["folds_at_budget"],
                "epochs_median": statistics.median(result["epochs"]),
                "seconds": round(time.time() - started),
            }
            print(
                f"  予算感度 {budget} エポック: 地点MAE {out[str(budget)]['mae']:.2f} / "
                f"張り付き {out[str(budget)]['folds_at_budget']} 件"
            )
    finally:
        MAX_EPOCHS = original

    a, b = out[str(original)]["mae"], out[str(original * factor)]["mae"]
    return {
        "budgets": out,
        "delta_mae": b - a,
        "margin": BUDGET_SENSITIVITY_MARGIN,
        "truncation_drives_conclusion": abs(b - a) >= BUDGET_SENSITIVITY_MARGIN,
        "seed": seed,
    }


def shuffled_second_channel(paired, second, seed: int = 20260910):
    """CH4 チャンネルを**地点年の対応だけ壊して**入れ替える。

    G-10 は通ったが、下がったのが「CH4 が緯度の情報を足した」からか
    「第 2 チャンネルが学習を安定させた」からか分けられていなかった(SPEC §7.11)。

    入れ替えた CH4 は、**分布も滑らかさも本物のまま、その地点年のものではない**。
    これでも下がるなら、効いていたのは情報でなくチャンネルの存在である。
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(second))
    # 自分自身に当たった分は隣とずらす(入れ替えが一部で起きない事故を防ぐ)
    for i in range(len(order)):
        if order[i] == i:
            j = (i + 1) % len(order)
            order[i], order[j] = order[j], order[i]
    assert not any(order[i] == i for i in range(len(order))), "入れ替わっていない例がある"
    return [second[j] for j in order]


# ------------------------------------------------------------------ L8: CH4 の要素を潰す対照

def rotated_second_channel(second, seed: int = 20260914):
    """CH4 を地点年ごとに**巡回でずらす**。振幅と形は本物のまま、CO2 との位相差だけを壊す。

    位相なしの模型は二つのチャンネルを**揃って**回すことには不変だが、片方だけを回すと
    チャンネル間のずれとして見える。ずらし幅は 6〜18 ビン(四半期〜四分の三年)に限る ——
    1 ビンずらしでは位相差がほとんど壊れず、対照が何も言わなくなる。
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in second:
        shift = int(rng.integers(6, 19))
        out.append(dataclasses.replace(s, values=tuple(np.roll(np.asarray(s.values), shift).tolist())))
    return out


def equalised_second_channel(second):
    """CH4 の振幅を地点年ごとに**揃える**。形と CO2 との位相差は本物のまま、振幅の大小だけを消す。

    揃える先は全例の標準偏差の中央値にする。入力の桁を本物と同じに保たないと、
    「振幅が消えた」のか「桁が変わって学習が変わった」のかが分けられない。
    """
    stds = [float(np.std(np.asarray(s.values))) for s in second]
    assert min(stds) > 0, "振幅 0 の曲線は揃えられない"
    target = float(np.median(stds))
    return [
        dataclasses.replace(s, values=tuple((np.asarray(s.values) / sd * target).tolist()))
        for s, sd in zip(second, stds)
    ]


# ------------------------------------------------------------------ L8: 学習率の nested 選定

#: 外側の訓練群から切る二つの分割の割合。**止め時を選ぶ分割**と**学習率を選ぶ分割**は排他。
#: L7 では一つの内側検証を両方に使っていたので、学習率が高いほど内側検証の谷を深く掘れ、
#: その値は学習率の比較に使えなかった(SPEC §7.16)。**測る前に決めて動かさない。**
NESTED_STOP_FRACTION = 0.2
NESTED_SELECT_FRACTION = 0.2


def nested_masks(examples, seed: int):
    """fold ごとの (訓練, 止め時, 選定) の 0/1 重みを作る。三つは排他で、試験側はどれにも乗らない。"""
    groups = [ex.group for ex in examples]
    folds = curves.leave_one_group_out(examples)
    shape = (len(folds), len(examples))
    train_w = np.zeros(shape, dtype=np.float32)
    stop_w = np.zeros(shape, dtype=np.float32)
    select_w = np.zeros(shape, dtype=np.float32)
    for f, (train_idx, test_idx) in enumerate(folds):
        held = groups[test_idx[0]]
        pool = sorted({g for g in groups if g != held})
        rng = np.random.default_rng(20260914 + seed)
        rng.shuffle(pool)
        n_stop = max(1, int(round(len(pool) * NESTED_STOP_FRACTION)))
        n_select = max(1, int(round(len(pool) * NESTED_SELECT_FRACTION)))
        stop = set(pool[:n_stop])
        select = set(pool[n_stop:n_stop + n_select])
        for i in train_idx:
            if groups[i] in stop:
                stop_w[f, i] = 1.0
            elif groups[i] in select:
                select_w[f, i] = 1.0
            else:
                train_w[f, i] = 1.0
    return folds, train_w, stop_w, select_w


def choose_per_fold(scores_by_lr: dict) -> list:
    """fold ごとに、渡された誤差が最小の学習率を返す。**渡されたもの以外は見ない。**

    選定に使う誤差だけを引数に取る形にしておくと、試験側が入り込む経路がコードの上で無くなる。
    """
    keys = list(scores_by_lr)
    matrix = np.stack([np.asarray(scores_by_lr[k], dtype=np.float64) for k in keys])
    return [keys[i] for i in np.argmin(matrix, axis=0)]


def run_nested(examples, x, y, channels: int, seed: int) -> dict:
    """学習率の三候補を同じ初期値から学習し、fold ごとに選ぶ分割で学習率を選ぶ。

    返り値には、同じ重みの上で **L7 の規則(止め時の分割で選ぶ)** を当てた予測も入れる。
    重みが同じなので、二つの差は選定規則だけの差になる(P-09)。
    """
    global LEARNING_RATE
    folds, train_w, stop_w, select_w = nested_masks(examples, seed)
    groups = [ex.group for ex in examples]
    n = len(examples)

    # 排他の検算。止め時の分割が選定に漏れていたら、L7 の偏りがそのまま残る。
    assert float(np.max(train_w + stop_w + select_w)) <= 1.0, "分割が重なっている"
    for f, (_, test_idx) in enumerate(folds):
        assert not np.any(train_w[f, test_idx] + stop_w[f, test_idx] + select_w[f, test_idx]), (
            "試験側に重みが乗っている"
        )

    stacked = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *[model.init_params(seed * 1000 + f, channels=channels) for f in range(len(folds))],
    )
    xj, yj = jnp.asarray(x), jnp.asarray(y)
    fold_mae = jax.jit(jax.vmap(masked_mae, in_axes=(0, None, None, 0)))

    per_lr = {}
    original = LEARNING_RATE
    try:
        for lr in LR_CANDIDATES:
            LEARNING_RATE = lr
            started = time.time()
            trainer = _make_trainer()
            best_params, best_stop, best_epochs = trainer(
                stacked, xj, yj, jnp.asarray(train_w), jnp.asarray(stop_w)
            )
            preds = np.full(n, np.nan, dtype=np.float64)
            for f, (_, test_idx) in enumerate(folds):
                fold_params = jax.tree.map(lambda leaf: leaf[f], best_params)
                preds[test_idx] = np.asarray(model.forward(fold_params, jnp.asarray(x[test_idx])))
            assert not np.isnan(preds).any(), "予測されていない例がある"
            epochs = [int(e) for e in np.asarray(best_epochs)]
            per_lr[f"{lr:g}"] = {
                "preds": preds,
                "stop_mae": np.asarray(best_stop, dtype=np.float64),
                "select_mae": np.asarray(fold_mae(best_params, xj, yj, jnp.asarray(select_w)), dtype=np.float64),
                "epochs": epochs,
                "folds_at_budget": sum(1 for e in epochs if e >= MAX_EPOCHS),
                "seconds": round(time.time() - started),
            }
            print(f"    lr={lr:g}: {per_lr[f'{lr:g}']['seconds']} 秒", flush=True)
    finally:
        LEARNING_RATE = original

    def assemble(chosen):
        out = np.full(n, np.nan, dtype=np.float64)
        for f, (_, test_idx) in enumerate(folds):
            out[test_idx] = per_lr[chosen[f]]["preds"][test_idx]
        return out

    chosen_select = choose_per_fold({k: v["select_mae"] for k, v in per_lr.items()})
    chosen_stop = choose_per_fold({k: v["stop_mae"] for k, v in per_lr.items()})
    return {
        "preds": assemble(chosen_select),
        "preds_stop_rule": assemble(chosen_stop),
        "chosen": chosen_select,
        "chosen_stop_rule": chosen_stop,
        "held_out": [groups[test_idx[0]] for _, test_idx in folds],
        "per_lr": per_lr,
    }


def summarise_nested(examples, y, result: dict, seed: int) -> dict:
    """一つの種の nested 結果を、報告に書ける形(numpy を含まない要約)に落とす。"""
    everyone = range(len(examples))
    return {
        "seed": seed,
        "mae": _per_site_mae(examples, everyone, result["preds"], y),
        "mae_stop_rule": _per_site_mae(examples, everyone, result["preds_stop_rule"], y),
        "mae_fixed_lr": {
            k: _per_site_mae(examples, everyone, v["preds"], y) for k, v in result["per_lr"].items()
        },
        "chosen": dict(zip(result["held_out"], result["chosen"])),
        "chosen_counts": {k: result["chosen"].count(k) for k in result["per_lr"]},
        "chosen_stop_rule_counts": {k: result["chosen_stop_rule"].count(k) for k in result["per_lr"]},
        "epochs_median": {k: statistics.median(v["epochs"]) for k, v in result["per_lr"].items()},
        "folds_at_budget": {k: v["folds_at_budget"] for k, v in result["per_lr"].items()},
        "seconds": sum(v["seconds"] for v in result["per_lr"].values()),
    }


def main() -> int:
    baseline = json.loads(
        (REPORT.parent / "baseline.json").read_text(encoding="utf-8")
    )
    opponent_name = baseline["best_two_feature_system"]
    opponent_mae = baseline["systems"][opponent_name]["mae"]

    co2 = curves.build_examples("co2")
    x1, y1 = model.to_arrays(co2)

    unit = measure_unit(co2, x1, y1, channels=1)
    passes = len(SEEDS) * 3
    print(
        f"一単位(勾配 1 回): {unit*1000:.0f} ms → "
        f"1 走 {unit * MAX_EPOCHS * 3.3 / 60:.0f} 分見込み × {passes} 走 = "
        f"{unit * MAX_EPOCHS * 3.3 * passes / 60:.0f} 分見込み"
        f"(scan の中は素の勾配の 3.3 倍 — 実測 2026-09-08)"
    )

    print(f"主系統: {len(co2)} 地点年 / {len({e.group for e in co2})} 群")

    # **走り出す前に書き出し経路を一度通す。** 書き出しは最後に一度しか来ないので、
    # ここで落とし損ねると計算がすべて終わってから落ちる(loop_002 GEN-LOGIC)。
    report = {
        "generated_from": "pipeline/experiment.py",
        "status": "running",
        "max_epochs": MAX_EPOCHS,
        "seeds": list(SEEDS),
        "inner_val_fraction": INNER_VAL_FRACTION,
        "conv_path": "matmul",
        "opponent": {"system": opponent_name, "mae": opponent_mae},
        "p01_threshold_mae": baseline["p01_threshold_mae"],
        "systems": {},
    }
    write_report(report)

    print("学習率の掃引(選定は内側検証のみ・種 0):")
    sweep = sweep_learning_rate(co2, x1, y1, channels=1, seed=0)
    global LEARNING_RATE
    LEARNING_RATE = sweep["chosen_lr"]
    print(f"  → 採用 lr={LEARNING_RATE:g}(内側検証が最小)")
    report["lr_sweep"] = sweep
    report["learning_rate"] = LEARNING_RATE
    write_report(report)

    primary = evaluate(co2, x1, y1, channels=1, label="cnn_co2", keep_audit=True)
    report["fold_audit"] = primary.pop("_audit")
    report["systems"]["cnn_co2"] = primary
    write_report(report)

    # G-10: CH4 を足す比較は、**両系統を同じ 739 地点年**に載せて行う。
    ch4 = {(e.code, e.year): e for e in curves.build_examples("ch4")}
    paired = [e for e in co2 if (e.code, e.year) in ch4]
    second = [ch4[(e.code, e.year)] for e in paired]
    x_pair_1, y_pair = model.to_arrays(paired)
    x_pair_2, _ = model.to_arrays(paired, second=second)

    print(f"CH4 比較: {len(paired)} 地点年 / {len({e.group for e in paired})} 群")
    co2_only = evaluate(paired, x_pair_1, y_pair, channels=1, label="cnn_co2_paired")
    report["systems"]["cnn_co2_paired"] = co2_only
    write_report(report)

    with_ch4 = evaluate(paired, x_pair_2, y_pair, channels=2, label="cnn_co2_ch4_paired")
    report["systems"]["cnn_co2_ch4_paired"] = with_ch4
    write_report(report)

    print("予算感度(打ち切りが結論を作っていないかを測る):")
    sensitivity = budget_sensitivity(co2, x1, y1, channels=1, seed=0)
    report["budget_sensitivity"] = sensitivity
    report["status"] = "complete"
    write_report(report)

    print()
    print(f"相手(L1): {opponent_name} 地点MAE {opponent_mae:.2f}")
    print(f"模型     : cnn_co2 地点MAE {primary['mae']:.2f}(種ごと {['%.2f' % m for m in primary['mae_per_seed']]})")
    print(f"G-08 閾値 {report['p01_threshold_mae']:.2f} → "
          f"{'通過' if primary['mae'] <= report['p01_threshold_mae'] else '不通過'}")
    print(f"G-09 相手 {opponent_mae:.2f} → "
          f"{'通過' if primary['mae'] < opponent_mae else '不通過'}")
    print(f"G-10 CH4 {co2_only['mae']:.2f} → {with_ch4['mae']:.2f} → "
          f"{'通過' if with_ch4['mae'] < co2_only['mae'] else '不通過'}")
    print(f"予算感度: 予算 3 倍で {sensitivity['delta_mae']:+.2f} 度 "
          f"(閾値 {BUDGET_SENSITIVITY_MARGIN} 度) → "
          f"{'打ち切りが結論を左右する' if sensitivity['truncation_drives_conclusion'] else '打ち切りは結論を作っていない'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
