"""Pipeline tests.

decide_promotion is tested exhaustively with hand-built TrainingResults
since it's the single most important function in the project — the
actual "safe automated retraining" claim lives entirely here. run_pipeline
itself is tested with fakes (no real training, no real network) so these
tests run in milliseconds and never touch the database's model-training
code path, which is already covered by test_train.py.
"""

from matchcast.pipeline import PipelineResult, decide_promotion, run_pipeline
from matchcast.train import TrainingResult


def _result(holdout_brier: float, beats_baseline: bool = True) -> TrainingResult:
    return TrainingResult(
        model_path="artifacts/fake.json",
        data_hash="abc",
        n_train=70,
        n_holdout=18,
        train_brier=0.4,
        holdout_brier=holdout_brier,
        holdout_log_loss=0.9,
        beats_random_baseline=beats_baseline,
        trained_at="20260101T000000Z",
    )


def test_rejects_challenger_that_fails_random_baseline_even_with_no_champion():
    challenger = _result(holdout_brier=0.9, beats_baseline=False)
    should_promote, reason = decide_promotion(challenger, champion=None)
    assert should_promote is False
    assert "random-guess baseline" in reason


def test_rejects_challenger_that_fails_random_baseline_even_if_better_than_champion():
    # Pathological but real case: challenger scores better than champion
    # numerically, but neither actually beats random guessing. Both are
    # bad; we should never promote a model that fails the baseline check,
    # regardless of how it compares to an equally-bad champion.
    challenger = _result(holdout_brier=0.7, beats_baseline=False)
    champion = _result(holdout_brier=0.75, beats_baseline=False)
    should_promote, reason = decide_promotion(challenger, champion)
    assert should_promote is False
    assert "random-guess baseline" in reason


def test_promotes_first_ever_model_if_it_beats_baseline():
    challenger = _result(holdout_brier=0.5, beats_baseline=True)
    should_promote, reason = decide_promotion(challenger, champion=None)
    assert should_promote is True
    assert "no existing champion" in reason


def test_promotes_challenger_that_beats_champion_beyond_tolerance():
    challenger = _result(holdout_brier=0.50)
    champion = _result(holdout_brier=0.60)
    should_promote, reason = decide_promotion(challenger, champion, tolerance=0.0)
    assert should_promote is True
    assert "improves on champion's" in reason


def test_rejects_challenger_within_tolerance_of_champion():
    challenger = _result(holdout_brier=0.599)
    champion = _result(holdout_brier=0.600)
    should_promote, reason = decide_promotion(challenger, champion, tolerance=0.01)
    assert should_promote is False
    assert "does not improve" in reason


def test_rejects_challenger_that_is_worse_than_champion():
    challenger = _result(holdout_brier=0.65)
    champion = _result(holdout_brier=0.55)
    should_promote, reason = decide_promotion(challenger, champion, tolerance=0.0)
    assert should_promote is False


def test_rejects_challenger_exactly_at_tolerance_boundary_is_promoted():
    # improvement == tolerance exactly: our rule uses >=, so this SHOULD
    # promote. This test exists specifically to pin down that boundary,
    # since off-by-one errors on >= vs > are a classic source of bugs.
    challenger = _result(holdout_brier=0.50)
    champion = _result(holdout_brier=0.55)
    should_promote, _ = decide_promotion(challenger, champion, tolerance=0.05)
    assert should_promote is True


class FakeClient:
    def get_competition_matches(self, competition="WC"):
        return []


def test_run_pipeline_returns_a_pipeline_result(session_factory, monkeypatch):
    # Patch train_model to avoid needing 10+ real feature rows in this
    # unit test — run_pipeline's own orchestration is what's under test
    # here, not the training math (already covered by test_train.py).
    import matchcast.pipeline as pipeline_module

    def fake_train_model(rows, train_fraction=0.8):
        return None, _result(holdout_brier=0.5)

    monkeypatch.setattr(pipeline_module, "train_model", fake_train_model)

    result = run_pipeline(session_factory, FakeClient())

    assert isinstance(result, PipelineResult)
    assert result.decision == "promoted"
    assert result.model_version_id is not None