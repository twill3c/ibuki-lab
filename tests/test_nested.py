"""学習率の nested 選定と、CH4 の要素を潰す対照の検査(TEST_SPEC T-061..T-063)。"""
import numpy as np
import pytest

from pipeline import curves, experiment


def _examples(n_groups: int = 20, per_group: int = 2, seed: int = 20260914):
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_groups):
        for k in range(per_group):
            values = tuple(float(v) for v in rng.normal(size=curves.N_BINS))
            out.append(curves.Example(
                code=f"s{g:02d}", year=2000 + k, group=f"g{g:02d}", latitude=float(rng.uniform(-80, 80)),
                values=values, n_samples=24, filled_bins=curves.N_BINS, interpolated=0,
            ))
    return out


# ---------------------------------------------------------------- T-061

def test_t061_nested_splits_are_exclusive_and_never_touch_the_test_group():
    examples = _examples()
    folds, train_w, stop_w, select_w = experiment.nested_masks(examples, seed=0)
    groups = np.array([e.group for e in examples])
    assert float(np.max(train_w + stop_w + select_w)) == 1.0
    for f, (train_idx, test_idx) in enumerate(folds):
        # 試験側はどれにも乗らず、訓練側はどれか一つに必ず乗る
        assert not np.any(train_w[f, test_idx] + stop_w[f, test_idx] + select_w[f, test_idx])
        assert np.all((train_w[f, train_idx] + stop_w[f, train_idx] + select_w[f, train_idx]) == 1.0)
        stop_groups = set(groups[stop_w[f] > 0])
        select_groups = set(groups[select_w[f] > 0])
        assert stop_groups and select_groups
        assert not stop_groups & select_groups
        # 19 群の 20% = 4 群ずつ(群単位で切る。例の数ではない)
        assert len(stop_groups) == 4 and len(select_groups) == 4


def test_t061_positive_control_l7_style_split_is_caught():
    """L7 の組み方(一つの内側検証を止め時にも学習率選びにも使う)は、重なりの検査で落ちる。"""
    examples = _examples()
    _, _, stop_w, select_w = experiment.nested_masks(examples, seed=0)
    assert float(np.sum(stop_w * select_w)) == 0.0
    l7_style_select = stop_w  # 同じ分割を二役に使う
    assert float(np.sum(stop_w * l7_style_select)) > 0.0


def test_t061_splits_depend_on_the_seed_and_are_deterministic():
    examples = _examples()
    a = experiment.nested_masks(examples, seed=0)
    b = experiment.nested_masks(examples, seed=0)
    c = experiment.nested_masks(examples, seed=1)
    assert np.array_equal(a[3], b[3])
    assert not np.array_equal(a[3], c[3])


# ---------------------------------------------------------------- T-062

def test_t062_choice_reads_only_the_scores_it_is_given():
    scores = {"0.003": [3.0, 1.0, 2.0], "0.01": [2.0, 2.0, 2.0], "0.03": [1.0, 3.0, 2.5]}
    assert experiment.choose_per_fold(scores) == ["0.03", "0.003", "0.003"]


@pytest.fixture
def tiny_budget(monkeypatch):
    monkeypatch.setattr(experiment, "MAX_EPOCHS", 4)
    monkeypatch.setattr(experiment, "LR_CANDIDATES", (1e-3, 3e-2))


def test_t062_test_labels_cannot_move_the_choice_for_their_fold(tiny_budget):
    """fold 0 の試験群の緯度を書き換えても、fold 0 の学習率の選択は動かない。

    **陽性対照**: fold 0 の選定群の緯度を書き換えると、fold 0 の選定誤差は動く
    (選定が実際にその分割を読んでいることの証拠)。
    """
    examples = _examples(n_groups=8)
    x = np.array([[[v] for v in e.values] for e in examples], dtype=np.float32)
    y = np.array([e.latitude for e in examples], dtype=np.float32)
    base = experiment.run_nested(examples, x, y, channels=1, seed=0)

    folds, _, _, select_w = experiment.nested_masks(examples, seed=0)
    test_idx = folds[0][1]
    y_test_moved = y.copy()
    y_test_moved[test_idx] += 50.0
    moved = experiment.run_nested(examples, x, y_test_moved, channels=1, seed=0)
    assert moved["chosen"][0] == base["chosen"][0]
    for k in base["per_lr"]:
        assert moved["per_lr"][k]["select_mae"][0] == pytest.approx(base["per_lr"][k]["select_mae"][0])

    y_select_moved = y.copy()
    y_select_moved[select_w[0] > 0] += 50.0
    control = experiment.run_nested(examples, x, y_select_moved, channels=1, seed=0)
    k = next(iter(base["per_lr"]))
    assert abs(control["per_lr"][k]["select_mae"][0] - base["per_lr"][k]["select_mae"][0]) > 1.0


# ---------------------------------------------------------------- T-063

def _ch4_like(n: int = 30, seed: int = 20260914):
    rng = np.random.default_rng(seed)
    t = 2 * np.pi * (np.arange(curves.N_BINS) + 0.5) / curves.N_BINS
    out = []
    for i in range(n):
        amp = float(rng.uniform(5, 40))
        phase = float(rng.uniform(0, 2 * np.pi))
        values = amp * np.cos(t - phase) + 0.1 * amp * np.cos(2 * t)
        out.append(curves.Example(
            code=f"c{i:02d}", year=2000, group=f"g{i:02d}", latitude=0.0,
            values=tuple(float(v) for v in values), n_samples=24, filled_bins=curves.N_BINS, interpolated=0,
        ))
    return out


def _changed(before, after):
    return sum(
        1 for a, b in zip(before, after)
        if np.max(np.abs(np.asarray(a.values) - np.asarray(b.values))) > 1e-6
    )


def test_t063_rotation_keeps_amplitude_and_breaks_every_phase():
    second = _ch4_like()
    rotated = experiment.rotated_second_channel(second)
    assert _changed(second, rotated) == len(second)
    for a, b in zip(second, rotated):
        va, vb = np.asarray(a.values), np.asarray(b.values)
        # 値の集合(したがって振幅)は保たれる
        assert np.allclose(np.sort(va), np.sort(vb))
        # 実際のずれ幅が 6〜18 ビンにある
        shifts = [s for s in range(curves.N_BINS) if np.allclose(np.roll(va, s), vb)]
        assert shifts and 6 <= shifts[0] <= 18
    assert rotated[0] is not second[0] and second[0].values == _ch4_like()[0].values


def test_t063_equalising_removes_amplitude_and_keeps_shape():
    second = _ch4_like()
    equalised = experiment.equalised_second_channel(second)
    assert _changed(second, equalised) == len(second) - sum(
        1 for s in second if np.isclose(np.std(s.values), np.median([np.std(q.values) for q in second]))
    )
    stds = [float(np.std(e.values)) for e in equalised]
    assert max(stds) - min(stds) < 1e-9
    for a, b in zip(second, equalised):
        assert np.corrcoef(a.values, b.values)[0, 1] == pytest.approx(1.0)


def test_t063_negative_control_identity_is_not_a_control():
    """何もしない「対照」は、変わった例の数の検査で落ちる。対照が効いているかは数えて確かめる。"""
    second = _ch4_like()
    assert _changed(second, list(second)) == 0
