"""The fit 5 model: a logistic signal scored on chronological, purged splits.

Decisions overlap in time, so a split that trains on an outcome not yet known
at the test date would flatter the model. These pin the splits, the fit, and
the evaluation's verdicts on synthetic data with a known answer.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from tradingagents import outcome_model as om


def _days(n, start=date(2026, 1, 5), step=7):
    return [(start + timedelta(days=step * i)).isoformat() for i in range(n)]


def _later(day, days=7):
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_last_dates_are_held_out_and_training_stops_at_what_was_known():
    days = _days(8)
    dates = [d for d in days for _ in range(3)]
    resolved = [_later(d) for d in dates]  # each outcome is known a week later

    dev, train, test = om.holdout_split(dates, resolved, 0.25)

    assert {dates[i] for i in test} == set(days[-2:])
    assert {dates[i] for i in dev} == set(days[:-2])
    # The week before the holdout settled on its first day, so it cannot train.
    assert {dates[i] for i in train} == set(days[:-3])


@pytest.mark.unit
def test_walk_forward_folds_train_only_on_outcomes_known_before_the_test_block():
    days = _days(10)
    dates = [d for d in days for _ in range(2)]
    resolved = [_later(d, 10) for d in dates]
    rows = np.arange(len(dates))

    folds = om.walk_forward(dates, resolved, rows, n_folds=4)

    assert len(folds) == 4
    tested = []
    for train, test in folds:
        first = min(dates[i] for i in test)
        assert all(dates[i] < first and resolved[i] < first for i in train)
        tested += [dates[i] for i in test]
    assert sorted(set(tested)) == days[2:]  # the first block only ever trains
    assert len(folds[-1][0]) > len(folds[0][0])  # the window expands


@pytest.mark.unit
def test_a_holdout_outside_zero_and_one_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        om.holdout_split(_days(10), _days(10), 1.0)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_fit_finds_the_direction_of_a_predictor_and_ignores_a_missing_value():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = (x + rng.normal(scale=0.5, size=200) > 0).astype(float)
    X = np.column_stack([x, np.full(200, 3.0)])  # a constant column must not break it

    model = om.fit_logistic(X, y, l2=1.0)

    assert model.weights[1] > 1.0
    p = model.predict(np.array([[2.0, 3.0], [-2.0, 3.0], [np.nan, 3.0]]))
    assert p[0] > 0.9 and p[1] < 0.1
    # A missing value is the training mean, which here is near zero.
    assert abs(p[2] - model.predict(np.array([[x.mean(), 3.0]]))[0]) < 1e-9


@pytest.mark.unit
def test_with_no_columns_the_model_is_the_base_rate():
    y = np.array([1.0] * 30 + [0.0] * 10)
    p = om.fit_logistic(np.empty((40, 0)), y, l2=1.0).predict(np.empty((5, 0)))
    assert np.allclose(p, 0.75, atol=1e-3)


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------


def _dataset(n_days=16, per_day=4, seed=1):
    """Decisions where one question predicts the outcome and the rating does not."""
    rng = np.random.default_rng(seed)
    dates = [d for d in _days(n_days) for _ in range(per_day)]
    n = len(dates)
    signal = rng.uniform(0, 4, size=n)
    alpha = (signal - 2) / 50 + rng.normal(scale=0.01, size=n)
    rows = [{"rating": 1.0, "review": 0.0, "signal": s, "signal_sd": 0.3,
             "noise": float(rng.uniform()), "flat": 0.5}
            for s in signal]
    columns = ["rating", "review", "signal", "signal_sd", "noise", "flat"]
    return om.Dataset.from_rows(
        rows, columns, alpha, dates, [_later(d) for d in dates],
        [f"T{i} {d} Buy" for i, d in enumerate(dates)],
    )


QUESTIONS = {"signal": ("signal", "signal_sd"), "noise": ("noise",), "flat": ("flat",)}


@pytest.mark.unit
def test_an_informative_question_beats_the_rating_on_the_held_out_dates():
    ev = om.evaluate(_dataset(), QUESTIONS, baseline=("rating", "review"))

    base, rating, full = (ev.model(n) for n in ("base rate", "rating", "rating + Jev features"))
    assert full.holdout_loss < rating.holdout_loss - 0.1
    assert full.holdout_loss < base.holdout_loss - 0.1
    assert full.cv_loss < rating.cv_loss
    # Where the model favours the stock, the stock did better.
    assert full.favoured[0] > full.rest[0]
    assert rating.favoured is None  # every decision is a Buy, so the rating favours none
    assert ev.holdout_dates == (_days(16)[-4], _days(16)[-1])


@pytest.mark.unit
def test_the_question_that_carries_the_signal_ranks_first_and_a_constant_one_is_flat():
    ev = om.evaluate(_dataset(), QUESTIONS, baseline=("rating", "review"))

    by_id = {q.id: q for q in ev.questions}
    assert ev.questions[0].id == "signal"
    assert by_id["signal"].gain > 0.05
    assert by_id["flat"].flat and not by_id["signal"].flat
    assert by_id["flat"].gain == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_the_worst_predicted_decisions_are_listed_for_proposing_new_questions():
    ev = om.evaluate(_dataset(), QUESTIONS, baseline=("rating", "review"))

    assert len(ev.worst) == om.WORST_SHOWN
    label, p, alpha = ev.worst[0]
    assert label.startswith("T")
    assert (p > 0.5) != (alpha > 0)  # the worst miss is a wrong call

    text = ev.render()
    assert "rating + Jev features" in text
    assert "signal: +" in text and "(keep)" in text
    assert "flat: drop" in text
    assert "Worst-predicted dev decisions" in text


@pytest.mark.unit
def test_too_few_decisions_is_refused_before_any_fit():
    ds = _dataset(n_days=5, per_day=4)
    with pytest.raises(om.NotEnoughData, match="at least 40 settled decisions"):
        om.evaluate(ds, QUESTIONS, baseline=("rating", "review"))


@pytest.mark.unit
def test_a_dev_period_with_one_outcome_only_cannot_be_cross_validated():
    ds = _dataset()
    ds.y[:] = 1.0
    with pytest.raises(om.NotEnoughData, match="cross-validate"):
        om.evaluate(ds, QUESTIONS, baseline=("rating", "review"))
