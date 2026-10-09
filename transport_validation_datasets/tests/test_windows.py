"""Unit tests for the time window helpers in windows.py.

Pure functions: no device, no source data, only small files in tmp_path.
They decide which Thomson samples get fit and how pooled rows are laid out, so a
change here changes every windowed dataset.
"""

import numpy as np
import pytest
from loguru import logger

import transport_validation_datasets.windows as windows_module
from transport_validation_datasets.gp_fitting.batch_io import ShotFitInput
from transport_validation_datasets.windows import (
    in_any_window,
    pool_windows,
    read_shotlist,
    restrict_to_windows,
    window_bounds,
    window_centers,
    window_membership,
)

N_CH = 3
# Six Thomson samples: three in the first window, two in the second, none in the third
SAMPLE_TIMES = np.array([0.61, 0.70, 0.85, 1.05, 1.15, 1.90], dtype=np.float32)
WINDOWS = [(0.6, 0.9), (1.0, 1.2), (1.2, 1.5)]
# The sample at 0.85 s sits in both of these
OVERLAPPING = [(0.6, 0.9), (0.8, 1.2)]


def make_fit_input() -> ShotFitInput:
    x = np.tile([0.2, 0.5, 0.9], (SAMPLE_TIMES.size, 1))
    x = x + np.arange(SAMPLE_TIMES.size)[:, None] * 1e-3
    te = np.arange(SAMPLE_TIMES.size * N_CH, dtype=float).reshape(-1, N_CH)
    return ShotFitInput(
        x=x, te_y=te, te_err=te + 100, ne_y=te * 2, ne_err=te + 200, time=SAMPLE_TIMES
    )


class TestReadShotlist:
    def test_plain_file_keeps_order_drops_repeats_comments_and_blanks(self, tmp_path):
        path = tmp_path / "shots.txt"
        path.write_text("# header, 2 shots\n7\n5\n\n  # note\n5\n\n")

        shots, windows = read_shotlist(path)

        assert shots == [7, 5]
        assert windows is None

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("7\nshot # note\n", "line 2: 'shot # note' is not a shot number"),
            (
                '"shot","t_start","t_end"\n5,0.1,0.2\n',
                "line 1: .* is not a shot number",
            ),
            ("shot;t_start;t_end\n5;0.1;0.2\n", "line 1: .* is not a shot number"),
            # A byte order mark hides the shot column of an otherwise windowed header
            (
                chr(0xFEFF) + "shot,t_start,t_end\n5,0.1,0.2\n",
                "needs a shot or pulse_no column",
            ),
            ("# only a comment\n\n", "no shot"),
        ],
    )
    def test_unparsable_shotlist_rejected(self, tmp_path, text, message):
        path = tmp_path / "shots.txt"
        path.write_text(text, encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            read_shotlist(path)

    def test_windowed_csv_groups_windows_per_shot_sorted_by_start(self, tmp_path):
        path = tmp_path / "shots.csv"
        path.write_text(
            "index,original_index,shot,t_start,t_end\n"
            "0,0,5,1.0,1.2\n"
            "1,1,5,0.6,0.9\n"
            "2,4,7,0.65,0.78\n"
            "3,5,5,1.25,1.5\n"
        )

        shots, windows = read_shotlist(path)

        assert shots == [5, 7]  # first appearance order, once each
        assert windows == {5: [(0.6, 0.9), (1.0, 1.2), (1.25, 1.5)], 7: [(0.65, 0.78)]}

    def test_shot_column_may_be_called_pulse_no(self, tmp_path):
        path = tmp_path / "shots.csv"
        path.write_text("pulse_no,t_start,t_end\n9,0.1,0.2\n")

        shots, windows = read_shotlist(path)

        assert shots == [9]
        assert windows == {9: [(0.1, 0.2)]}

    def test_accepts_str_path(self, tmp_path):
        path = tmp_path / "shots.txt"
        path.write_text("3\n")

        assert read_shotlist(str(path)) == ([3], None)

    def test_touching_windows_allowed(self, tmp_path):
        path = tmp_path / "shots.csv"
        path.write_text("shot,t_start,t_end\n5,1.0,1.2\n5,1.2,1.5\n")

        _, windows = read_shotlist(path)

        assert windows == {5: [(1.0, 1.2), (1.2, 1.5)]}

    def test_overlapping_windows_allowed(self, tmp_path):
        path = tmp_path / "shots.csv"
        path.write_text("shot,t_start,t_end\n5,1.0,1.5\n5,1.2,1.7\n")

        _, windows = read_shotlist(path)

        assert windows == {5: [(1.0, 1.5), (1.2, 1.7)]}

    def test_windows_with_the_same_center_rejected(self, tmp_path):
        path = tmp_path / "shots.csv"
        path.write_text("shot,t_start,t_end\n5,0.6,0.9\n5,0.7,0.8\n")

        with pytest.raises(ValueError, match="same grid time"):
            read_shotlist(path)

    def test_centers_closer_than_grid_step_rejected(self, tmp_path):
        path = tmp_path / "shots.csv"
        path.write_text("shot,t_start,t_end\n5,0.6,0.9\n5,0.6005,0.9005\n")

        with pytest.raises(ValueError, match="closer than 0.001 s"):
            read_shotlist(path)

    def test_window_must_run_forward(self, tmp_path):
        path = tmp_path / "shots.csv"
        for start, end in (("1.0", "0.5"), ("1.0", "1.0")):
            path.write_text(f"shot,t_start,t_end\n5,{start},{end}\n")

            with pytest.raises(ValueError, match="t_start < t_end"):
                read_shotlist(path)

    def test_window_must_be_finite(self, tmp_path):
        path = tmp_path / "shots.csv"
        path.write_text("shot,t_start,t_end\n5,0.5,inf\n")

        with pytest.raises(ValueError, match="finite"):
            read_shotlist(path)

    def test_missing_shot_column_rejected(self, tmp_path):
        path = tmp_path / "shots.csv"
        path.write_text("shotno,t_start,t_end\n5,1.0,1.5\n")

        with pytest.raises(ValueError, match="shot or pulse_no"):
            read_shotlist(path)

    def test_unparsable_bound_identifies_line(self, tmp_path):
        path = tmp_path / "shots.csv"
        path.write_text("shot,t_start,t_end\n5,0.5,0.7\n5,,0.9\n")

        with pytest.raises(ValueError, match="line 3"):
            read_shotlist(path)


class TestWindowBounds:
    def test_pairs_become_a_float_array(self):
        bounds = window_bounds(WINDOWS)
        assert bounds.shape == (3, 2)
        assert bounds.dtype == np.float64
        assert bounds.tolist() == [list(w) for w in WINDOWS]

    def test_array_passes_through_unchanged(self):
        bounds = window_bounds(np.asarray(WINDOWS, dtype=np.float32))
        np.testing.assert_array_equal(bounds, np.asarray(WINDOWS, dtype=np.float32))

    def test_no_windows_is_zero_by_two(self):
        assert window_bounds([]).shape == (0, 2)
        assert window_bounds(np.zeros((0, 2))).shape == (0, 2)


class TestWindowCenters:
    def test_midpoints(self):
        np.testing.assert_allclose(window_centers(WINDOWS), [0.75, 1.1, 1.35])

    def test_agrees_with_pooled_row_times(self):
        pooled = pool_windows(make_fit_input(), WINDOWS, shot=1)
        np.testing.assert_array_equal(pooled.time, window_centers(WINDOWS))

    def test_no_windows_gives_no_centers(self):
        assert window_centers([]).shape == (0,)


class TestWindowMembership:
    def test_times_outside_every_window_are_in_none(self):
        member = window_membership([0.5, 0.95, 1.6], WINDOWS)

        assert member.shape == (3, 3)
        assert not member.any()

    def test_times_inside_are_in_their_window_only(self):
        member = window_membership([0.7, 1.1, 1.4], WINDOWS)

        assert member.tolist() == [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]

    def test_bounds_are_inclusive(self):
        member = window_membership([0.6, 0.9, 1.0, 1.5], WINDOWS)

        assert member.any(axis=1).all()

    def test_touching_boundary_is_in_both_windows(self):
        # 1.2 ends the second window and starts the third
        assert window_membership([1.2], WINDOWS).tolist() == [[False, True, True]]

    def test_overlap_is_in_both_windows(self):
        assert window_membership([0.85], OVERLAPPING).tolist() == [[True, True]]

    def test_float32_time_on_a_bound_stays_inside(self):
        # float32(0.65) is just below 0.65, the tolerance keeps it in
        times = np.array([0.65, 0.78], dtype=np.float32)

        member = window_membership(times, [(0.65, 0.78)])

        assert member.all()

    def test_scalar_time_gives_one_row(self):
        member = window_membership(np.array(1.1), np.asarray(WINDOWS))

        assert member.shape == (3,)
        assert member.tolist() == [False, True, False]


class TestInAnyWindow:
    def test_collapses_membership_over_windows(self):
        member = window_membership(SAMPLE_TIMES, OVERLAPPING)
        np.testing.assert_array_equal(
            in_any_window(SAMPLE_TIMES, OVERLAPPING), member.any(axis=1)
        )
        assert in_any_window(SAMPLE_TIMES, WINDOWS).tolist() == [
            True,
            True,
            True,
            True,
            True,
            False,
        ]

    def test_keeps_shape_of_times(self):
        assert in_any_window(SAMPLE_TIMES.reshape(2, 3), WINDOWS).shape == (2, 3)

    def test_no_windows_hold_nothing(self):
        assert not in_any_window(SAMPLE_TIMES, []).any()


class TestRestrictToWindows:
    def test_keeps_only_rows_inside_windows(self):
        restricted = restrict_to_windows(make_fit_input(), WINDOWS)

        assert restricted.time.tolist() == SAMPLE_TIMES[:5].tolist()

    def test_tags_rows_with_their_window(self):
        restricted = restrict_to_windows(make_fit_input(), WINDOWS)

        assert restricted.window_index.tolist() == [0, 0, 0, 1, 1]
        assert np.array_equal(restricted.windows, np.asarray(WINDOWS))

    def test_every_channel_array_sliced_alike(self):
        fit_input = make_fit_input()

        restricted = restrict_to_windows(fit_input, WINDOWS)

        for name in ("x", "te_y", "te_err", "ne_y", "ne_err"):
            assert np.array_equal(
                getattr(restricted, name), getattr(fit_input, name)[:5]
            )

    def test_sample_in_overlapping_windows_kept_once_under_earlier(self):
        restricted = restrict_to_windows(make_fit_input(), OVERLAPPING)

        assert restricted.time.tolist() == SAMPLE_TIMES[:5].tolist()
        assert restricted.window_index.tolist() == [0, 0, 0, 1, 1]

    def test_no_row_inside_returns_none(self):
        assert restrict_to_windows(make_fit_input(), [(1.7, 1.8)]) is None

    def test_input_not_mutated(self):
        fit_input = make_fit_input()
        before = fit_input.te_y.copy()

        restrict_to_windows(fit_input, WINDOWS)

        assert np.array_equal(fit_input.te_y, before)
        assert fit_input.windows.size == 0
        assert (fit_input.window_index == -1).all()


class TestPoolWindows:
    def test_one_row_per_window_in_window_order(self):
        pooled = pool_windows(make_fit_input(), WINDOWS, shot=1)

        assert pooled.te_y.shape[0] == len(WINDOWS)
        assert pooled.window_index.tolist() == [0, 1, 2]
        assert np.array_equal(pooled.windows, np.asarray(WINDOWS))

    def test_row_is_sample_rows_laid_end_to_end(self):
        fit_input = make_fit_input()

        pooled = pool_windows(fit_input, WINDOWS, shot=1)

        assert np.array_equal(pooled.te_y[0], fit_input.te_y[:3].reshape(-1))
        assert np.array_equal(pooled.x[0], fit_input.x[:3].reshape(-1))
        assert np.array_equal(pooled.ne_err[0], fit_input.ne_err[:3].reshape(-1))

    def test_rows_padded_to_widest_window_in_samples(self):
        fit_input = make_fit_input()

        pooled = pool_windows(fit_input, WINDOWS, shot=1)

        assert pooled.te_y.shape[1] == 3 * N_CH
        assert np.array_equal(
            pooled.te_y[1, : 2 * N_CH], fit_input.te_y[3:5].reshape(-1)
        )
        assert np.isnan(pooled.te_y[1, 2 * N_CH :]).all()

    def test_sample_in_overlapping_windows_pooled_into_each(self):
        fit_input = make_fit_input()

        pooled = pool_windows(fit_input, OVERLAPPING, shot=1)

        # 0.85 s is the third sample of the first window and the first of the second
        assert np.array_equal(pooled.te_y[0], fit_input.te_y[:3].reshape(-1))
        assert np.array_equal(pooled.te_y[1], fit_input.te_y[2:5].reshape(-1))

    def test_time_is_window_center(self):
        pooled = pool_windows(make_fit_input(), WINDOWS, shot=1)

        assert np.allclose(pooled.time, [0.75, 1.1, 1.35])

    def test_window_without_samples_keeps_all_nan_row(self):
        pooled = pool_windows(make_fit_input(), WINDOWS, shot=1)

        for name in ("x", "te_y", "te_err", "ne_y", "ne_err"):
            assert np.isnan(getattr(pooled, name)[2]).all()

    def test_samples_already_restricted_pool_same(self):
        fit_input = make_fit_input()

        from_all = pool_windows(fit_input, WINDOWS, shot=1)
        from_restricted = pool_windows(
            restrict_to_windows(fit_input, WINDOWS), WINDOWS, shot=1
        )

        assert np.array_equal(from_all.te_y, from_restricted.te_y, equal_nan=True)
        assert np.array_equal(from_all.x, from_restricted.x, equal_nan=True)

    def test_window_above_point_limit_logged_critical(self, monkeypatch):
        monkeypatch.setattr(windows_module, "POOLED_POINTS_WARN", 5)
        messages = []
        sink = logger.add(messages.append, level="CRITICAL")
        try:
            pool_windows(make_fit_input(), WINDOWS, shot=1)
        finally:
            logger.remove(sink)

        # The first window pools 9 points, the second 6, the third none
        assert len(messages) == 2
        assert "[0.600, 0.900]" in messages[0]
        assert "[1.000, 1.200]" in messages[1]
