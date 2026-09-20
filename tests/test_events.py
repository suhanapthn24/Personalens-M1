import pytest

from personalens.ingestion.events import load_quiz_csv
from personalens.schemas import RawEvent


def _csv(tmp_path, text, name="q.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_quiz_csv_default_columns(tmp_path):
    p = _csv(tmp_path, "timestamp,question_id,concept,correct\n2026-09-01 10:00,q1,Bayes Theorem,yes\n2026-09-02,q2,Naive Bayes,0\n")
    events, warnings = load_quiz_csv(p, "u1")
    assert [e.outcome for e in events] == [1.0, 0.0]
    assert events[0].concept == "Bayes Theorem" and events[0].item_id == "q1"
    assert events[0].family == "performance"
    assert events[0].timestamp.tzinfo is not None
    assert warnings == []


def test_score_columns_and_column_map(tmp_path):
    p = _csv(tmp_path, "When,Q,Topic,Marks,Out of\n2026-09-01,q1,Gradient Descent,3,4\n2026-09-01,q2,Gradient Descent,9,4\n")
    events, warnings = load_quiz_csv(
        p, "u1", event_type="task_grade",
        column_map={"timestamp": "When", "item_id": "Q", "concept": "Topic", "score": "Marks", "max_score": "Out of"},
    )
    assert len(events) == 1 and events[0].outcome == 0.75 and events[0].event_type == "task_grade"
    assert any("Skipped 1" in w for w in warnings)  # score > max is unusable


def test_bad_timestamps_are_imputed_and_flagged(tmp_path):
    p = _csv(tmp_path, "timestamp,correct\nnot a date,1\n2026-09-01,0\n")
    events, warnings = load_quiz_csv(p, "u1")
    assert events[0].payload.get("timestamp_imputed") is True
    assert "timestamp_imputed" not in events[1].payload
    assert any("no parseable timestamp" in w for w in warnings)


def test_missing_outcome_columns_raises_helpfully(tmp_path):
    p = _csv(tmp_path, "foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="Found columns"):
        load_quiz_csv(p, "u1")


def test_event_store_roundtrip(pipeline, tmp_path):
    p = _csv(tmp_path, "timestamp,concept,correct\n2026-09-01,A,1\n2026-09-03,B,0\n")
    res = pipeline.ingest_quiz_csv(p, "u1")
    assert res.status == "events_ingested" and res.n_events == 2

    pipeline.events.add(RawEvent(user_id="u1", event_type="time_on_page", concept="A", value=95.0, payload={"page": 4}))
    all_events = pipeline.events.list("u1")
    assert len(all_events) == 3
    only_perf = pipeline.events.list("u1", event_types=["quiz_answer"])
    assert {e.concept for e in only_perf} == {"A", "B"}
    assert pipeline.events.list("u1", concept="A", event_types=["time_on_page"])[0].payload == {"page": 4}
    assert pipeline.events.list("someone-else") == []


def test_event_outcome_validated():
    with pytest.raises(ValueError):
        RawEvent(user_id="u", event_type="quiz_answer", outcome=1.5)
