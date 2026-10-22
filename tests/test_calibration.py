"""Causal calibration, independent oracle sums, and checkpoint integrity."""
import json

import numpy as np
import pandas as pd
import pytest

from hoao import Calibration, ReturnsPanel, Sample, calibrate
from hoao._numerics import pair_core


def values():
    return np.random.default_rng(71).normal(size=(100, 5)) * np.linspace(0.01, 0.03, 5)


def run(panel=None, **kwargs):
    options = dict(n_assets=3, train_size=12, test_size=8, n_pairs=7,
                   before=70, random_state=42)
    options.update(kwargs)
    return calibrate(ReturnsPanel(values()) if panel is None else panel, **options)


def test_sampled_pair_sum_matches_independent_reconstruction():
    panel = ReturnsPanel(values())
    fitted = run(panel)
    a, b = np.zeros(3), np.zeros((3, 3))
    latest = -1
    signals = np.arange(11, 70 - 1 - 8)
    for identifier in range(7):
        rng = np.random.default_rng(np.random.SeedSequence([42, identifier]))
        signal = int(rng.choice(signals))
        cols = np.sort(rng.choice(np.arange(5), 3, replace=False))
        first = signal + 2
        c2, c4 = pair_core(panel.values[signal - 11:signal + 1, cols],
                           panel.values[first:first + 8, cols])
        a += c2
        b += c4
        latest = max(latest, first + 7)
    np.testing.assert_allclose(fitted.correlation_core, a / 7, rtol=1e-13)
    np.testing.assert_allclose(fitted.cokurtosis_core, b / 7, rtol=1e-13)
    assert fitted.n_pairs == 7
    assert fitted.latest_target == latest < fitted.before


def test_exact_resume_and_worker_count_invariance(tmp_path):
    panel = ReturnsPanel(values())
    direct = run(panel, n_pairs=13, n_jobs=1)
    split = run(panel, n_pairs=5, n_jobs=2)
    path = tmp_path / "core.npz"
    split.save(path)
    resumed = Calibration.load(path)
    assert resumed.update(panel, n_pairs=8, before=70, n_jobs=3) is resumed
    np.testing.assert_array_equal(resumed.correlation_core, direct.correlation_core)
    np.testing.assert_array_equal(resumed.cokurtosis_core, direct.cokurtosis_core)
    assert resumed.n_pairs == direct.n_pairs == 13
    assert resumed.latest_target == direct.latest_target


def test_array_dataframe_equivalence_and_fitting_labels():
    frame = pd.DataFrame(values(), index=pd.bdate_range("2020-01-01", periods=100),
                         columns=list("ABCDE"))
    positional = run()
    dated = run(ReturnsPanel(frame), before=frame.index[70])
    np.testing.assert_array_equal(dated.correlation_core, positional.correlation_core)
    np.testing.assert_array_equal(dated.cokurtosis_core, positional.cokurtosis_core)
    fit = frame.iloc[70:82, :3]
    fitted = dated.fit(fit)
    plain = positional.fit(fit.to_numpy(), as_of=81)
    assert fitted.reference_variance == Sample().fit(fit).reference_variance
    np.testing.assert_array_equal(fitted.correlation, plain.correlation)
    assert fitted.assets == ("A", "B", "C")
    np.testing.assert_array_equal(dated.fit_correlation(fit).correlation, fitted.correlation)
    np.testing.assert_allclose(dated.fit_cokurtosis(fit).fourth_moment(np.ones(3) / 3),
                               fitted.cokurtosis.fourth_moment(np.ones(3) / 3))


def test_fitting_rejects_wrong_dimensions_and_unsafe_cutoffs():
    core = run()
    x = values()[70:82, :3]
    with pytest.raises(ValueError, match="as_of is required"):
        core.fit(x)
    with pytest.raises(ValueError, match="cutoff"):
        core.fit(x, as_of=69)
    with pytest.raises(ValueError, match="exactly"):
        core.fit(x[:-1], as_of=81)
    with pytest.raises(ValueError, match="exactly"):
        core.fit(values()[70:82], as_of=81)
    frame = pd.DataFrame(values(), index=pd.bdate_range("2020-01-01", periods=100))
    dated = run(ReturnsPanel(frame), before=frame.index[70])
    with pytest.raises(ValueError, match="last fitting"):
        dated.fit(frame.iloc[70:82, :3], as_of=frame.index[80])


def test_zero_fill_keeps_assets_counts_locations_and_inputs():
    x = values()
    # The sole signal is 11: complete history, execution row 12, targets 13:21.
    available = {11: [0, 1, 2]}
    x[15, 0] = np.nan
    x[18, 2] = np.nan
    original = x.copy()
    panel = ReturnsPanel(x, available=available)
    core = run(panel, before=21, n_pairs=3)
    assert core.filled_count == 6
    expected = np.array([[i, row, col] for i in range(3) for row, col in ((15, 0), (18, 2))])
    np.testing.assert_array_equal(core.filled_locations, expected)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(panel.values, original)
    with pytest.raises(ValueError, match="Missing target return"):
        run(panel, before=21, n_pairs=3, missing="raise")
    target = x[13:21, :3].copy()
    target[np.isnan(target)] = 0
    a, b = pair_core(x[:12, :3], target)
    np.testing.assert_allclose(core.correlation_core, a)
    np.testing.assert_allclose(core.cokurtosis_core, b)


def test_complete_constant_target_is_retained():
    x = values()
    x[13:21, 0] = 0.0
    panel = ReturnsPanel(x, available={11: [0, 1, 2]})
    core = run(panel, before=21)
    assert core.filled_count == 0
    assert np.isfinite(core.cokurtosis_core).all()


def test_future_values_cannot_affect_sample_selection(monkeypatch):
    import hoao.calibration as module

    records = []
    original = module.pair_core
    def record(train, test, **kwargs):
        records.append(train.copy())
        return original(train, test, **kwargs)
    monkeypatch.setattr(module, "pair_core", record)
    x = values()
    universe = {11: [0, 1, 2, 3, 4]}
    run(ReturnsPanel(x, available=universe), before=21, n_pairs=5)
    first = [a.copy() for a in records]
    records.clear()
    x[13:21] = np.nan
    run(ReturnsPanel(x, available=universe), before=21, n_pairs=5)
    for left, right in zip(first, records):
        np.testing.assert_array_equal(left, right)


def test_failed_update_leaves_accumulator_unchanged():
    core = run(missing="raise")
    a, b, count, cutoff = core.correlation_core, core.cokurtosis_core, core.n_pairs, core.before
    x = values()
    x[90:98] = np.nan
    panel = ReturnsPanel(x, available={88: [0, 1, 2]})
    with pytest.raises(ValueError, match="Missing target"):
        core.update(panel, n_pairs=5, before=99)
    np.testing.assert_array_equal(core.correlation_core, a)
    np.testing.assert_array_equal(core.cokurtosis_core, b)
    assert core.n_pairs == count and core.before == cutoff


@pytest.mark.parametrize("change", ["schema", "basis", "count", "shape", "nan", "cutoff", "fills", "object", "fill_bounds"])
def test_corrupt_checkpoints_are_rejected(tmp_path, change):
    file = tmp_path / "core.npz"
    run().save(file)
    with np.load(file, allow_pickle=False) as archive:
        state = {key: archive[key].copy() for key in archive.files}
    meta = json.loads(str(state["metadata"].item()))
    if change == "schema":
        meta["schema"] = 99
    elif change == "basis":
        meta["basis"] = "different"
    elif change == "count":
        meta["n_pairs"] = 1.5
    elif change == "shape":
        state["sum2"] = state["sum2"][:-1]
    elif change == "nan":
        state["sum4"][0, 0] = np.nan
    elif change == "cutoff":
        meta["latest_target"] = meta["before"]
    elif change == "fills":
        meta["filled_count"] = 1
    elif change == "object":
        state["sum2"] = state["sum2"].astype(object)
    elif change == "fill_bounds":
        meta["filled_count"] = 1
        state["filled_locations"] = np.array([[0, 9999, 0]], dtype=np.int64)
    state["metadata"] = np.array(json.dumps(meta))
    np.savez(file, **state)
    with pytest.raises(ValueError):
        Calibration.load(file)


def test_dated_checkpoint_round_trip_and_new_history_update(tmp_path):
    frame = pd.DataFrame(values(), index=pd.bdate_range("2020-01-01", periods=100))
    frame.index = frame.index + pd.Timedelta(nanoseconds=37)
    panel = ReturnsPanel(frame)
    core = run(panel, before=frame.index[70])
    file = tmp_path / "dates.npz"
    core.save(file)
    loaded = Calibration.load(file)
    assert loaded.before == core.before and loaded.latest_target == core.latest_target
    loaded.update(panel, n_pairs=2, before=frame.index[95], start=frame.index[70])
    assert loaded.n_pairs == core.n_pairs + 2
    assert loaded.latest_target < frame.index[95].to_datetime64()
    with pytest.raises(ValueError, match="previous cutoff"):
        loaded.update(panel, n_pairs=1, before=frame.index[70])


def test_cuda_dispatch_uses_bounded_batches_with_same_sampling(monkeypatch):
    """Test dispatch using the scalar CPU oracle; this is not a hardware test."""
    import hoao.calibration as module

    batches = []
    original = module.pair_cores_batch
    def emulate(train, test, **kwargs):
        batches.append(len(train))
        assert kwargs["device"] == "cuda"
        return original(train, test, device="cpu", order=kwargs["order"])
    monkeypatch.setattr(module, "pair_cores_batch", emulate)
    gpu_dispatch = run(n_pairs=9, device="cuda")
    cpu = run(n_pairs=9)
    assert batches == [4, 4, 1]
    np.testing.assert_allclose(gpu_dispatch.correlation_core, cpu.correlation_core, rtol=1e-13)
    np.testing.assert_allclose(gpu_dispatch.cokurtosis_core, cpu.cokurtosis_core, rtol=1e-13)
    assert gpu_dispatch.latest_target == cpu.latest_target
    with pytest.raises(ValueError, match="n_jobs=1"):
        run(device="cuda", n_jobs=2)


@pytest.mark.parametrize("stage", ["write", "sync", "replace"])
def test_failed_atomic_save_preserves_prior_checkpoint(tmp_path, monkeypatch, stage):
    import hoao.calibration as module

    path = tmp_path / "core.npz"
    core = run()
    core.save(path)
    original = path.read_bytes()
    def fail(*args, **kwargs):
        if stage == "write":
            args[0].write(b"incomplete archive")
        raise OSError("Injected checkpoint failure")
    if stage == "write":
        monkeypatch.setattr(module.np, "savez_compressed", fail)
    elif stage == "sync":
        monkeypatch.setattr(module.os, "fsync", fail)
    else:
        monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="Injected"):
        core.save(path)
    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}
    assert Calibration.load(path).n_pairs == core.n_pairs
