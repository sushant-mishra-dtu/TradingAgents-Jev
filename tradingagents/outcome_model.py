"""A small signal learned from settled decisions: did the decision beat its benchmark?

Fit 5 in docs/jev-use-cases.md. ``report_features`` turns each decision's
reports into numeric columns; this fits an L2-regularised logistic regression of
"alpha > 0" on them and asks whether they predict better than the rating alone.
The result is a probability, so it can sit next to the rating as a calibrated
signal, but only once it has beaten the rating on decisions it never saw.

Decisions are ordered in time and their holding windows overlap, so every split
is chronological: the latest analysis dates are held out, the rest are
cross-validated walk-forward, and a decision trains a model only if its outcome
was known before the first date that model is tested on (its resolution date is
earlier). A shuffled split would train on outcomes from the test period.

numpy only: the data is a few dozen to a few hundred rows, and a linear model
with a penalty chosen by cross-validation is what that supports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# Below these, the held-out block and the folds are too small to say anything.
MIN_DECISIONS = 40
MIN_DATES = 6
MIN_TRAIN = 10
N_FOLDS = 4
L2_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
# A question whose columns barely move across decisions cannot explain outcomes
# (the cookbook's cut-off, on each column's own scale).
FLAT_SD = 0.05
WORST_SHOWN = 5


class NotEnoughData(ValueError):
    pass


def check_size(dates: Sequence[str]) -> None:
    n, n_dates = len(dates), len(set(dates))
    if n < MIN_DECISIONS or n_dates < MIN_DATES:
        raise NotEnoughData(
            f"Learning needs at least {MIN_DECISIONS} settled decisions over at least "
            f"{MIN_DATES} analysis dates, to train on the earlier ones and test on the "
            f"later ones; there are {n} over {n_dates}."
        )


@dataclass
class Dataset:
    X: np.ndarray            # decisions x columns; NaN where a document was missing
    y: np.ndarray            # 1.0 when the decision beat its benchmark
    alpha: np.ndarray
    dates: list[str]
    resolved: list[str]      # when each outcome became known
    columns: list[str]
    labels: list[str]

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, float]], columns: Sequence[str],
                  alpha: Sequence[float], dates: Sequence[str], resolved: Sequence[str],
                  labels: Sequence[str]) -> Dataset:
        X = np.array([[row.get(c, np.nan) for c in columns] for row in rows],
                     dtype=float).reshape(len(rows), len(columns))
        alpha = np.asarray(alpha, dtype=float)
        return cls(X, (alpha > 0).astype(float), alpha, list(dates), list(resolved),
                   list(columns), list(labels))

    def indices(self, names: Sequence[str]) -> list[int]:
        return [self.columns.index(n) for n in names]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def column_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of each column over its present values (0 if none)."""
    present = ~np.isnan(X)
    count = np.maximum(present.sum(axis=0), 1)
    mean = np.where(present, X, 0.0).sum(axis=0) / count
    var = (np.where(present, X - mean, 0.0) ** 2).sum(axis=0) / count
    return mean, np.sqrt(var)


@dataclass
class LogisticModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray      # intercept first

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.nan_to_num((X - self.mean) / self.scale)
        return _sigmoid(self.weights[0] + Z @ self.weights[1:])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> LogisticModel:
    """Newton's method on the L2-penalised log loss, over standardised columns.

    A missing value is set to the column's training mean, i.e. 0 once
    standardised, so it moves the prediction neither way.
    """
    mean, sd = column_stats(X)
    scale = np.where(sd > 1e-6, sd, 1.0)
    Z = np.nan_to_num((X - mean) / scale)

    A = np.hstack([np.ones((len(y), 1)), Z])
    penalty = np.eye(A.shape[1]) * l2
    penalty[0, 0] = 1e-6  # the intercept is not shrunk, only kept finite
    base = np.clip(y.mean(), 1e-3, 1 - 1e-3)
    w = np.zeros(A.shape[1])
    w[0] = np.log(base / (1 - base))
    for _ in range(100):
        p = _sigmoid(A @ w)
        gradient = A.T @ (p - y) + penalty @ w
        hessian = A.T @ (A * (p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return LogisticModel(mean, scale, w)


# ---------------------------------------------------------------------------
# Chronological splits
# ---------------------------------------------------------------------------


def holdout_split(dates: Sequence[str], resolved: Sequence[str], fraction: float):
    """Dev rows, the rows the holdout model trains on, and the held-out rows.

    The held-out rows are the last ``fraction`` of analysis dates. The training
    rows are the dev rows whose outcome was known before the first of them.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"the holdout must be a fraction between 0 and 1, got {fraction}")
    days = sorted(set(dates))
    first = days[-min(len(days) - 1, max(1, round(len(days) * fraction)))]
    dates, resolved = np.asarray(dates), np.asarray(resolved)
    dev = np.flatnonzero(dates < first)
    return dev, dev[resolved[dev] < first], np.flatnonzero(dates >= first)


def walk_forward(dates: Sequence[str], resolved: Sequence[str], rows: np.ndarray,
                 n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over ``rows``: test each block of dates on the ones before.

    The dates are cut into ``n_folds + 1`` consecutive blocks; fold k tests block
    k and trains on the earlier rows whose outcome was known before block k began.
    """
    dates, resolved = np.asarray(dates), np.asarray(resolved)
    days = sorted(set(dates[rows]))
    folds = []
    for block in np.array_split(np.asarray(days), n_folds + 1)[1:]:
        if not len(block):
            continue
        first, last = block[0], block[-1]
        test = rows[(dates[rows] >= first) & (dates[rows] <= last)]
        train = rows[(dates[rows] < first) & (resolved[rows] < first)]
        folds.append((train, test))
    return folds


def _trainable(y: np.ndarray) -> bool:
    return len(y) >= MIN_TRAIN and 0 < y.sum() < len(y)


def cross_validate(ds: Dataset, cols: Sequence[int], folds, l2: float):
    """Log loss of out-of-fold predictions, and the predictions (NaN where none)."""
    out = np.full(len(ds.y), np.nan)
    for train, test in folds:
        model = fit_logistic(ds.X[np.ix_(train, cols)], ds.y[train], l2)
        out[test] = model.predict(ds.X[np.ix_(test, cols)])
    scored = ~np.isnan(out)
    return log_loss(ds.y[scored], out[scored]), out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class ModelScore:
    name: str
    l2: float | None
    cv_loss: float
    holdout_loss: float
    favoured: tuple[float, int] | None = None   # mean held-out alpha and count, p above base rate
    rest: tuple[float, int] | None = None


@dataclass
class QuestionScore:
    id: str
    gain: float      # dev CV log loss without the question, minus with it
    flat: bool


@dataclass
class Evaluation:
    decisions: int
    dates: int
    holdout_dates: tuple[str, str]
    holdout_rows: int
    train_rows: int
    base_rate: float
    models: list[ModelScore]
    questions: list[QuestionScore] = field(default_factory=list)
    worst: list[tuple[str, float, float]] = field(default_factory=list)  # label, p, alpha

    def model(self, name: str) -> ModelScore:
        return next(m for m in self.models if m.name == name)

    def render(self) -> str:
        first, last = self.holdout_dates
        span = first if first == last else f"{first} to {last}"
        lines = [
            f"Settled decisions: {self.decisions} over {self.dates} analysis dates. "
            f"Held out {span}: {self.holdout_rows} decisions. Trained on the "
            f"{self.train_rows} settled before {first}, of which {self.base_rate:.0%} "
            "beat the benchmark.",
            "",
            "Log loss of P(beats the benchmark), lower is better (0.693 is a coin flip):",
        ]
        for m in self.models:
            line = f"- {m.name}: dev CV {m.cv_loss:.3f}, held out {m.holdout_loss:.3f}"
            if m.l2 is not None:
                line += f" (L2 {m.l2:g})"
            if m.favoured and m.rest:
                line += (f"; held-out alpha {m.favoured[0]:+.2%} (n={m.favoured[1]}) where it "
                         f"favours the stock vs {m.rest[0]:+.2%} (n={m.rest[1]}) elsewhere")
            lines.append(line)
        if self.questions:
            lines += ["", "Questions, by how much dropping each one raises the dev CV log "
                      "loss of the full model. Keep one only while this is above 0; a "
                      "small gain is often noise:"]
            for q in self.questions:
                verdict = ("flat: drop" if q.flat else
                           "keep" if q.gain > 0 else "drop: no worse without it")
                lines.append(f"- {q.id}: {q.gain:+.4f} ({verdict})")
        if self.worst:
            lines += ["", "Worst-predicted dev decisions, the cases to read when proposing "
                      "new questions:"]
            lines += [f"- {label}: P(beats) {p:.2f}, alpha {alpha:+.1%}"
                      for label, p, alpha in self.worst]
        lines += ["", "One sampling of each report and a few dozen decisions make these "
                  "figures noisy. Trust a gain only when it holds on the held-out dates and "
                  "again on a longer or wider sweep."]
        return "\n".join(lines)


def _alpha_split(alpha: np.ndarray, p: np.ndarray, base: float):
    favoured = p > base + 1e-9
    if not favoured.any() or favoured.all():
        return None, None
    return ((float(alpha[favoured].mean()), int(favoured.sum())),
            (float(alpha[~favoured].mean()), int((~favoured).sum())))


def evaluate(ds: Dataset, questions: Mapping[str, Sequence[str]], baseline: Sequence[str],
             holdout: float = 0.25) -> Evaluation:
    """Score the base rate, the baseline columns, and baseline plus every question.

    The L2 penalty of each model is chosen by walk-forward cross-validation on
    the dev dates, and each model is then scored once on the held-out dates.
    """
    check_size(ds.dates)
    dev, train, test = holdout_split(ds.dates, ds.resolved, holdout)
    folds = [(tr, te) for tr, te in walk_forward(ds.dates, ds.resolved, dev) if _trainable(ds.y[tr])]
    if len(folds) < 2 or not _trainable(ds.y[train]):
        raise NotEnoughData(
            "Too few settled decisions before the held-out dates, or all of one outcome, "
            "to cross-validate; widen the sweep or hold out less."
        )

    base_cols = ds.indices(baseline)
    all_cols = base_cols + ds.indices([c for cols in questions.values() for c in cols])
    base_rate = float(ds.y[train].mean())
    models = []
    for name, cols in (("base rate", []), ("rating", base_cols),
                       ("rating + Jev features", all_cols)):
        grid = L2_GRID if cols else (1.0,)
        l2, cv = min(((l2, cross_validate(ds, cols, folds, l2)[0]) for l2 in grid),
                     key=lambda pair: pair[1])
        model = fit_logistic(ds.X[np.ix_(train, cols)], ds.y[train], l2)
        p = model.predict(ds.X[np.ix_(test, cols)])
        favoured, rest = _alpha_split(ds.alpha[test], p, base_rate) if cols else (None, None)
        models.append(ModelScore(name, l2 if cols else None, cv, log_loss(ds.y[test], p),
                                 favoured, rest))

    full_l2 = models[-1].l2
    full_cv, oof = cross_validate(ds, all_cols, folds, full_l2)
    scored = []
    for qid, cols in questions.items():
        own = set(ds.indices(cols))
        without = [c for c in all_cols if c not in own]
        gain = cross_validate(ds, without, folds, full_l2)[0] - full_cv
        flat = bool(np.all(column_stats(ds.X[np.ix_(dev, sorted(own))])[1] < FLAT_SD))
        scored.append(QuestionScore(qid, gain, flat))
    scored.sort(key=lambda q: q.gain, reverse=True)

    predicted = np.flatnonzero(~np.isnan(oof))
    row_loss = [(-np.log(np.clip(oof[i] if ds.y[i] else 1 - oof[i], 1e-6, 1)), i) for i in predicted]
    worst = [(ds.labels[i], float(oof[i]), float(ds.alpha[i]))
             for _, i in sorted(row_loss, reverse=True)[:WORST_SHOWN]]

    held = [ds.dates[i] for i in test]
    return Evaluation(
        decisions=len(ds.y), dates=len(set(ds.dates)), holdout_dates=(min(held), max(held)),
        holdout_rows=len(test), train_rows=len(train), base_rate=base_rate,
        models=models, questions=scored, worst=worst,
    )
