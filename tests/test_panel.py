"""Independent checks of availability and chronological panel semantics."""
import numpy as np
import pandas as pd
import pytest

from hoao import ReturnsPanel


def data():
    rng = np.random.default_rng(19)
    return pd.DataFrame(rng.normal(size=(40, 4)),
                        index=pd.bdate_range("2020-01-01", periods=40),
                        columns=["A", "B", "C", "D"])


def test_pandas_array_and_universe_formats_agree():
    frame = data()
    mask = np.ones(frame.shape, dtype=bool)
    mask[10:20, 1] = False
    lists = [list(frame.columns[row]) for row in mask]
    mapping = dict(zip(frame.index, lists))
    flags = pd.DataFrame(mask, index=frame.index, columns=frame.columns)
    panels = [ReturnsPanel(frame, available=available) for available in (mask, lists, mapping, flags)]
    panels.append(ReturnsPanel(frame.to_numpy(), dates=list(frame.index),
                              assets=list(frame.columns), available=flags))
    for panel in panels:
        np.testing.assert_array_equal(panel.values, frame.to_numpy())
        np.testing.assert_array_equal(panel.available, mask)
        np.testing.assert_array_equal(panel.eligible(train_size=6), panels[0].eligible(train_size=6))
        assert panel.assets == tuple(frame.columns)


def test_automatic_eligibility_matches_direct_history_checks():
    frame = data()
    frame.iloc[11, 0] = np.nan
    frame.iloc[15:25, 2] = 1.234
    frame.iloc[5:12, 3] = 0.0
    panel = ReturnsPanel(frame)
    actual = panel.eligible(train_size=7)
    expected = np.zeros(frame.shape, dtype=bool)
    for row in range(6, len(frame)):
        window = frame.iloc[row - 6:row + 1].to_numpy()
        expected[row] = np.isfinite(window).all(axis=0) & (np.ptp(window, axis=0) > 0)
    np.testing.assert_array_equal(actual, expected)
    assert not panel.eligible(train_size=50).any()


def test_future_changes_do_not_change_historical_eligibility():
    frame = data()
    original = ReturnsPanel(frame).eligible(train_size=6)
    frame.iloc[25:] = np.nan
    changed = ReturnsPanel(frame).eligible(train_size=6)
    np.testing.assert_array_equal(original[:25], changed[:25])


def test_exact_signal_gap_and_cutoff_boundaries():
    panel = ReturnsPanel(np.random.default_rng(2).normal(size=(30, 3)))
    signals, _ = panel._signals(n_assets=3, train_size=5, test_size=4,
                                gap=1, before=20, start=7)
    np.testing.assert_array_equal(signals, np.arange(7, 15))
    for signal in signals:
        assert signal + 1 + 4 < 20
    with pytest.raises(ValueError, match="No signal"):
        panel._signals(n_assets=3, train_size=20, test_size=10, gap=1, before=30)


def test_inputs_and_public_arrays_are_not_mutable_aliases():
    values = data().to_numpy().copy()
    available = np.ones(values.shape, dtype=bool)
    panel = ReturnsPanel(values, available=available)
    values[:] = 0
    available[:] = False
    assert np.any(panel.values != 0)
    assert panel.available.all()
    for array in (panel.values, panel.available):
        with pytest.raises(ValueError):
            array[0] = 0
    returned = panel.eligible(train_size=3)
    returned[:] = False
    assert panel.eligible(train_size=3)[2:].all()


def test_mapping_missing_dates_are_empty_and_unknown_labels_raise():
    frame = data()
    panel = ReturnsPanel(frame, available={frame.index[10]: ["A", "C"]})
    assert panel.available.sum() == 2
    with pytest.raises(ValueError, match="Unknown"):
        ReturnsPanel(frame, available={frame.index[10]: ["ZZ"]})
    with pytest.raises(ValueError, match="Duplicate"):
        ReturnsPanel(frame, available={frame.index[10]: ["A", "A"]})
    with pytest.raises(ValueError, match="absent"):
        ReturnsPanel(frame, available={"2000-01-01": ["A"]})


def test_bad_axes_and_universe_alignment_fail():
    frame = data()
    with pytest.raises(ValueError, match="unique"):
        ReturnsPanel(frame.set_axis(["A", "A", "C", "D"], axis=1))
    with pytest.raises(ValueError, match="increasing"):
        ReturnsPanel(frame.iloc[::-1])
    mask = pd.DataFrame(True, index=frame.index, columns=frame.columns)
    with pytest.raises(ValueError, match="same row and column"):
        ReturnsPanel(frame, available=mask.iloc[:, ::-1])
    with pytest.raises(ValueError, match="boolean"):
        ReturnsPanel(frame, available=mask.astype(float))
    mask = mask.astype("boolean")
    mask.iloc[5, 0] = pd.NA
    with pytest.raises(ValueError, match="missing"):
        ReturnsPanel(frame, available=mask)
    with pytest.raises(ValueError, match="infinity"):
        ReturnsPanel([[1, 2], [3, np.inf]])


def test_dates_preserve_timezone_instants_and_accept_nanosecond_precision():
    frame = data()
    frame.index = frame.index.tz_localize("Europe/Paris") + pd.Timedelta(nanoseconds=37)
    panel = ReturnsPanel(frame)
    expected = frame.index.tz_convert("UTC").tz_localize(None).to_numpy()
    np.testing.assert_array_equal(panel.dates, expected)
    assert panel._row(frame.index[4]) == 4
    assert panel._row(frame.index[4].isoformat()) == 4
    with pytest.raises(ValueError, match="date-like"):
        panel._cutoff(5)


def test_range_index_is_positional():
    frame = data().reset_index(drop=True)
    panel = ReturnsPanel(frame, available={5: ["A", "B"]})
    assert panel.dates is None
    assert panel.available[5].sum() == 2


def test_complex_and_boolean_return_values_are_not_silently_cast():
    for values in (np.ones((10, 3), dtype=complex), np.ones((10, 3), dtype=bool)):
        with pytest.raises(ValueError, match="numeric"):
            ReturnsPanel(values)


@pytest.mark.parametrize("label", [None, np.nan, pd.NA, pd.NaT])
def test_missing_asset_labels_are_rejected(label):
    with pytest.raises(ValueError, match="missing"):
        ReturnsPanel(np.ones((10, 2)), assets=["A", label])


@pytest.mark.parametrize("stamp", ["2262-04-12", "2500-01-01", "1600-01-01",
    "2262-04-12T00:00:00.000000037Z", "2500-01-01T12:00:00.000000037+01:00"])
def test_dates_outside_nanosecond_range_do_not_wrap(stamp):
    from hoao.panel import _date

    with pytest.raises(ValueError, match="Invalid date"):
        _date(stamp)


@pytest.mark.parametrize("stamp", ["2262-04-12", "2500-01-01", "1600-01-01"])
def test_wide_numpy_dates_outside_nanosecond_range_do_not_wrap(stamp):
    from hoao.panel import _date

    with pytest.raises(ValueError, match="Invalid date"):
        _date(np.datetime64(stamp, "D"))


def test_exact_nanosecond_boundaries_and_timezone_conversion_are_retained():
    from hoao.panel import _date

    minimum = "1677-09-21T00:12:43.145224193"
    maximum = "2262-04-11T23:47:16.854775807"
    assert str(_date(minimum)) == minimum
    assert str(_date(maximum)) == maximum
    assert str(_date("2262-04-12T00:47:16.854775807+01:00")) == maximum
    for stamp in ("1677-09-21T00:12:43.145224192", "2262-04-11T23:47:16.854775808"):
        with pytest.raises(ValueError, match="Invalid date"):
            _date(stamp)
