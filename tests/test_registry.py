"""Model registry tests.

Uses a fake TrainingResult (not a real trained model) since this
file only tests record-keeping, never training logic — that
separation is deliberate, see registry.py's module docstring.
"""


from matchcast.registry import (
    get_current_champion,
    list_history,
    promote_to_champion,
    register_model,
)
from matchcast.train import TrainingResult


def _fake_result(holdout_brier: float = 0.5) -> TrainingResult:
    return TrainingResult(
        model_path=f"artifacts/fake_{holdout_brier}.json",
        data_hash="abc123",
        n_train=70,
        n_holdout=18,
        train_brier=0.4,
        holdout_brier=holdout_brier,
        holdout_log_loss=0.9,
        beats_random_baseline=holdout_brier < (2 / 3),
        trained_at="20260101T000000Z",
    )


def test_register_model_rejects_invalid_status(session_factory):
    with session_factory() as s:
        try:
            register_model(s, _fake_result(), status="not_a_real_status")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "invalid status" in str(exc)


def test_register_model_requires_reason_when_rejected(session_factory):
    with session_factory() as s:
        try:
            register_model(s, _fake_result(), status="rejected")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "rejection_reason" in str(exc)


def test_no_champion_when_registry_is_empty(session_factory):
    with session_factory() as s:
        assert get_current_champion(s) is None


def test_promote_sets_status_to_champion(session_factory):
    with session_factory() as s:
        entry = register_model(s, _fake_result(), status="rejected", rejection_reason="testing")
        promote_to_champion(s, entry)
        s.commit()

        champ = get_current_champion(s)
        assert champ.id == entry.id
        assert champ.status == "champion"


def test_promoting_a_new_model_retires_the_old_champion(session_factory):
    with session_factory() as s:
        first = register_model(s, _fake_result(0.6), status="rejected", rejection_reason="testing")
        promote_to_champion(s, first)
        s.commit()

        second = register_model(s, _fake_result(0.5), status="rejected", rejection_reason="testing")
        promote_to_champion(s, second)
        s.commit()

        # Exactly one champion must exist at any time — this is the
        # property the whole registry design depends on.
        s.refresh(first)
        assert first.status == "retired"
        assert get_current_champion(s).id == second.id


def test_list_history_returns_every_model_in_order(session_factory):
    with session_factory() as s:
        first = register_model(s, _fake_result(0.6), status="rejected", rejection_reason="worse")
        second = register_model(s, _fake_result(0.5), status="rejected", rejection_reason="testing")
        promote_to_champion(s, second)
        s.commit()

        history = list_history(s)
        assert [m.id for m in history] == [first.id, second.id]
        assert history[0].status == "rejected"
        assert history[1].status == "champion"