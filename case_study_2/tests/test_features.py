"""
Unit tests for extract_features() and normalize_events().

Each test targets a specific feature or edge case so failures point directly
to the broken calculation.
"""
import pytest
from main import extract_features, normalize_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METADATA = {
    "agent_id": "receptionist",
    "org_id": "org_a",
    "call_purpose": "sdoh_screening",
    "caller_phone_type": "mobile",
    "time_of_day": "morning",
    "day_of_week": "monday",
}


def make_events(*speech_pairs, tool_calls=0, ts_start=0, ts_end=120):
    """
    Build a normalized event list from a compact description.

    speech_pairs: sequence of (type, ts, duration_ms, words)
                  where type is "agent_speech" or "user_speech"
    tool_calls:   number of tool_call events to append at ts=50
    ts_start/end: call_start and call_end timestamps
    """
    events = [{"timestamp": ts_start, "type": "call_start"}]
    for etype, ts, duration_ms, words in speech_pairs:
        events.append({"timestamp": ts, "type": etype, "duration_ms": duration_ms, "words": words})
    for i in range(tool_calls):
        events.append({"timestamp": 50 + i, "type": "tool_call"})
    if ts_end is not None:
        events.append({"timestamp": ts_end, "type": "call_end"})
    return events


def features(*args, survey_completion_rate=0.0, meta=None, **kwargs):
    """Thin wrapper so tests don't repeat the metadata / scr arguments."""
    return extract_features(
        make_events(*args, **kwargs),
        meta or METADATA,
        survey_completion_rate,
    )


# ---------------------------------------------------------------------------
# Categorical features (metadata passthrough)
# ---------------------------------------------------------------------------

def test_categorical_agent_id():
    fd = extract_features([], {"agent_id": "nurse_bot"}, 0.0)
    assert fd["agent_id"] == "nurse_bot"


def test_categorical_call_purpose():
    fd = extract_features([], {"call_purpose": "appointment_scheduling"}, 0.0)
    assert fd["call_purpose"] == "appointment_scheduling"


def test_categorical_caller_phone_type():
    fd = extract_features([], {"caller_phone_type": "landline"}, 0.0)
    assert fd["caller_phone_type"] == "landline"


def test_categorical_time_of_day():
    fd = extract_features([], {"time_of_day": "evening"}, 0.0)
    assert fd["time_of_day"] == "evening"


def test_categorical_day_of_week():
    fd = extract_features([], {"day_of_week": "friday"}, 0.0)
    assert fd["day_of_week"] == "friday"


def test_categorical_defaults_to_unknown_when_missing():
    fd = extract_features([], {}, 0.0)
    for key in ("agent_id", "call_purpose", "caller_phone_type",
                "time_of_day", "day_of_week"):
        assert fd[key] == "unknown", f"{key} should default to 'unknown'"


# ---------------------------------------------------------------------------
# total_duration
# ---------------------------------------------------------------------------

def test_total_duration_normal():
    fd = features(ts_start=0, ts_end=120)
    assert fd["total_duration"] == 120.0


def test_total_duration_non_zero_start():
    fd = features(ts_start=10, ts_end=70)
    assert fd["total_duration"] == 60.0


def test_total_duration_zero_when_no_call_end():
    fd = features(ts_start=0, ts_end=None)
    assert fd["total_duration"] == 0.0


# ---------------------------------------------------------------------------
# agent_speech_count / user_speech_count
# ---------------------------------------------------------------------------

def test_agent_speech_count():
    fd = features(
        ("agent_speech", 2, 3000, 30),
        ("agent_speech", 8, 2000, 20),
        ("agent_speech", 15, 1000, 10),
    )
    assert fd["agent_speech_count"] == 3.0


def test_user_speech_count():
    fd = features(
        ("user_speech", 5, 1000, 10),
        ("user_speech", 12, 800, 8),
    )
    assert fd["user_speech_count"] == 2.0


def test_speech_counts_zero_when_no_speech():
    fd = features()
    assert fd["agent_speech_count"] == 0.0
    assert fd["user_speech_count"] == 0.0


# ---------------------------------------------------------------------------
# total_agent_ms / total_user_ms
# ---------------------------------------------------------------------------

def test_total_agent_ms():
    fd = features(
        ("agent_speech", 2, 3500, 40),
        ("agent_speech", 10, 1500, 15),
    )
    assert fd["total_agent_ms"] == 5000.0


def test_total_user_ms():
    fd = features(
        ("user_speech", 5, 2000, 20),
        ("user_speech", 12, 1000, 10),
    )
    assert fd["total_user_ms"] == 3000.0


def test_total_ms_zero_when_no_speech():
    fd = features()
    assert fd["total_agent_ms"] == 0.0
    assert fd["total_user_ms"] == 0.0


# ---------------------------------------------------------------------------
# total_agent_words / total_user_words
# ---------------------------------------------------------------------------

def test_total_agent_words():
    fd = features(
        ("agent_speech", 2, 3000, 25),
        ("agent_speech", 10, 2000, 15),
    )
    assert fd["total_agent_words"] == 40.0


def test_total_user_words():
    fd = features(
        ("user_speech", 5, 1000, 12),
        ("user_speech", 12, 800, 8),
    )
    assert fd["total_user_words"] == 20.0


# ---------------------------------------------------------------------------
# tool_call_count
# ---------------------------------------------------------------------------

def test_tool_call_count():
    fd = features(tool_calls=3)
    assert fd["tool_call_count"] == 3.0


def test_tool_call_count_zero():
    fd = features()
    assert fd["tool_call_count"] == 0.0


# ---------------------------------------------------------------------------
# Silence: total_silence_ms, silence_count
# ---------------------------------------------------------------------------

def test_total_silence_ms_single_gap():
    # agent ends at ts=2 + 3000ms/1000 = 5.0s; user starts at ts=8
    # gap = (8 - 5) * 1000 = 3000ms
    fd = features(
        ("agent_speech", 2, 3000, 30),
        ("user_speech", 8, 1000, 10),
    )
    assert fd["total_silence_ms"] == pytest.approx(3000.0)


def test_total_silence_ms_multiple_gaps():
    # gap 1: agent ends at 2+3=5s, user starts at 8  → 3000ms
    # gap 2: user ends at 8+1=9s, agent starts at 12 → 3000ms
    fd = features(
        ("agent_speech", 2, 3000, 30),
        ("user_speech", 8, 1000, 10),
        ("agent_speech", 12, 2000, 20),
    )
    assert fd["total_silence_ms"] == pytest.approx(6000.0)


def test_total_silence_ms_zero_when_no_gap():
    # agent ends exactly when user starts → no gap
    # agent at ts=0, dur=2000ms → ends at 2.0s; user at ts=2 → gap = 0
    fd = features(
        ("agent_speech", 0, 2000, 20),
        ("user_speech", 2, 1000, 10),
    )
    assert fd["total_silence_ms"] == pytest.approx(0.0)


def test_total_silence_ms_no_negative_for_overlap():
    # Overlapping events must not produce negative silence
    # agent at ts=0, dur=5000ms → ends at 5s; user starts at ts=3 → gap = -2000ms (overlap)
    fd = features(
        ("agent_speech", 0, 5000, 50),
        ("user_speech", 3, 1000, 10),
    )
    assert fd["total_silence_ms"] >= 0.0


def test_silence_count_counts_gaps_at_threshold():
    # gap = exactly 500ms (threshold) → should be counted
    # agent at ts=0, dur=500ms → ends at 0.5s; user at ts=1 → gap = 500ms
    fd = features(
        ("agent_speech", 0, 500, 5),
        ("user_speech", 1, 500, 5),
    )
    assert fd["silence_count"] == 1.0


def test_silence_count_excludes_gaps_below_threshold():
    # gap = 200ms < 500ms → silence_count stays 0, but total_silence_ms > 0
    # agent at ts=0, dur=800ms → ends at 0.8s; user at ts=1 → gap = 200ms
    fd = features(
        ("agent_speech", 0, 800, 8),
        ("user_speech", 1, 500, 5),
    )
    assert fd["silence_count"] == 0.0
    assert fd["total_silence_ms"] == pytest.approx(200.0)


def test_silence_count_multiple():
    # gap 1: agent ends at 3s, user starts at 5  → 2000ms ≥ 500ms  ✓
    # gap 2: user ends at 6s, agent starts at 8  → 2000ms ≥ 500ms  ✓
    # gap 3: agent ends at 9s, user starts at 9  → 0ms             ✗
    fd = features(
        ("agent_speech", 0, 3000, 30),
        ("user_speech", 5, 1000, 10),
        ("agent_speech", 8, 1000, 10),
        ("user_speech", 9, 500, 5),
    )
    assert fd["silence_count"] == 2.0


def test_silence_zero_when_no_speech():
    fd = features()
    assert fd["total_silence_ms"] == 0.0
    assert fd["silence_count"] == 0.0


# ---------------------------------------------------------------------------
# silence_ratio
# ---------------------------------------------------------------------------

def test_silence_ratio():
    # agent 3000ms, user 1000ms, silence 2000ms → total 6000ms
    # ratio = 2000/6000 ≈ 0.3333
    fd = features(
        ("agent_speech", 0, 3000, 30),  # ends at 3s
        ("user_speech", 5, 1000, 10),   # gap = 2000ms
    )
    expected = 2000.0 / (3000.0 + 1000.0 + 2000.0)
    assert fd["silence_ratio"] == pytest.approx(expected)


def test_silence_ratio_zero_when_no_audio():
    fd = features()
    assert fd["silence_ratio"] == 0.0


def test_silence_ratio_zero_when_no_silence():
    # back-to-back events with no gap
    fd = features(
        ("agent_speech", 0, 2000, 20),
        ("user_speech", 2, 1000, 10),
    )
    assert fd["silence_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# user_speech_ratio
# ---------------------------------------------------------------------------

def test_user_speech_ratio():
    # agent 3000ms, user 1000ms → ratio = 1000/4000 = 0.25
    fd = features(
        ("agent_speech", 0, 3000, 30),
        ("user_speech", 5, 1000, 10),
    )
    assert fd["user_speech_ratio"] == pytest.approx(0.25)


def test_user_speech_ratio_zero_when_no_speech():
    fd = features()
    assert fd["user_speech_ratio"] == 0.0


def test_user_speech_ratio_one_when_only_user():
    fd = features(("user_speech", 5, 2000, 20))
    assert fd["user_speech_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# turn_count
# ---------------------------------------------------------------------------

def test_turn_count_alternating():
    # agent → user → agent → user = 4 turns
    fd = features(
        ("agent_speech", 2,  3000, 30),
        ("user_speech",  7,  1000, 10),
        ("agent_speech", 10, 2000, 20),
        ("user_speech",  14, 800,  8),
    )
    assert fd["turn_count"] == 4.0


def test_turn_count_consecutive_same_speaker():
    # agent, agent, user = 2 turns (agent counted once; consecutive repeats don't add a turn)
    fd = features(
        ("agent_speech", 2,  3000, 30),
        ("agent_speech", 7,  2000, 20),
        ("user_speech",  11, 1000, 10),
    )
    assert fd["turn_count"] == 2.0


def test_turn_count_zero_when_no_speech():
    fd = features()
    assert fd["turn_count"] == 0.0


def test_turn_count_one_speaker_only():
    fd = features(
        ("agent_speech", 2, 3000, 30),
        ("agent_speech", 7, 2000, 20),
    )
    assert fd["turn_count"] == 1.0


# ---------------------------------------------------------------------------
# avg_user_words_per_turn / avg_agent_words_per_turn
# ---------------------------------------------------------------------------

def test_avg_user_words_per_turn():
    # 3 user turns with 10, 20, 30 words → avg = 60/3 = 20
    fd = features(
        ("user_speech", 2,  1000, 10),
        ("user_speech", 5,  1000, 20),
        ("user_speech", 8,  1000, 30),
    )
    assert fd["avg_user_words_per_turn"] == pytest.approx(20.0)


def test_avg_agent_words_per_turn():
    # 2 agent turns with 40, 60 words → avg = 50
    fd = features(
        ("agent_speech", 2,  3000, 40),
        ("agent_speech", 8,  2000, 60),
    )
    assert fd["avg_agent_words_per_turn"] == pytest.approx(50.0)


def test_avg_user_words_per_turn_zero_when_no_user_speech():
    fd = features(("agent_speech", 2, 3000, 30))
    assert fd["avg_user_words_per_turn"] == 0.0


def test_avg_agent_words_per_turn_zero_when_no_agent_speech():
    fd = features(("user_speech", 2, 1000, 10))
    assert fd["avg_agent_words_per_turn"] == 0.0


# ---------------------------------------------------------------------------
# survey_completion_rate
# ---------------------------------------------------------------------------

def test_survey_completion_rate_passed_through():
    fd = features(survey_completion_rate=0.75)
    assert fd["survey_completion_rate"] == pytest.approx(0.75)


def test_survey_completion_rate_default_zero():
    fd = features()
    assert fd["survey_completion_rate"] == 0.0


# ---------------------------------------------------------------------------
# normalize_events
# ---------------------------------------------------------------------------

def test_normalize_speech_detected_with_timestamp_key():
    raw = [{"timestamp": 5, "type": "speech_detected",
            "data": {"speaker": "agent", "duration_ms": 3000, "words": 30}}]
    result = normalize_events(raw)
    assert result == [{"timestamp": 5, "type": "agent_speech", "duration_ms": 3000, "words": 30}]


def test_normalize_speech_detected_with_ts_key():
    raw = [{"timestamp": 3, "type": "speech_detected",
            "data": {"speaker": "user", "duration_ms": 1500, "words": 15}}]
    result = normalize_events(raw)
    assert result == [{"timestamp": 3, "type": "user_speech", "duration_ms": 1500, "words": 15}]


def test_normalize_agent_speaker():
    raw = [{"timestamp": 0, "type": "speech_detected",
            "data": {"speaker": "agent", "duration_ms": 2000, "words": 20}}]
    assert normalize_events(raw)[0]["type"] == "agent_speech"


def test_normalize_user_speaker():
    raw = [{"timestamp": 0, "type": "speech_detected",
            "data": {"speaker": "user", "duration_ms": 1000, "words": 10}}]
    assert normalize_events(raw)[0]["type"] == "user_speech"


def test_normalize_call_start_passthrough():
    raw = [{"timestamp": 0, "type": "call_start"}]
    result = normalize_events(raw)
    assert result[0]["type"] == "call_start"
    assert result[0]["timestamp"] == 0


def test_normalize_call_end_passthrough():
    raw = [{"timestamp": 120, "type": "call_end"}]
    result = normalize_events(raw)
    assert result[0]["type"] == "call_end"
    assert result[0]["timestamp"] == 120


def test_normalize_tool_call_passthrough():
    raw = [{"timestamp": 45, "type": "tool_call", "tool": "submit_survey_response"}]
    result = normalize_events(raw)
    assert result[0]["type"] == "tool_call"
    assert result[0]["timestamp"] == 45
    assert result[0]["tool"] == "submit_survey_response"


def test_normalize_agent_speech_with_nested_data():
    raw = [{"timestamp": 2, "type": "agent_speech",
            "data": {"speaker": "agent", "duration_ms": 3000, "words": 30}}]
    result = normalize_events(raw)
    assert result[0] == {"timestamp": 2, "type": "agent_speech", "duration_ms": 3000, "words": 30}


def test_normalize_user_speech_with_nested_data():
    raw = [{"timestamp": 5, "type": "user_speech",
            "data": {"speaker": "user", "duration_ms": 1500, "words": 15}}]
    result = normalize_events(raw)
    assert result[0] == {"timestamp": 5, "type": "user_speech", "duration_ms": 1500, "words": 15}


def test_normalize_preserves_event_order():
    raw = [
        {"timestamp": 0,  "type": "call_start"},
        {"timestamp": 2,  "type": "speech_detected",
         "data": {"speaker": "agent", "duration_ms": 3000, "words": 30}},
        {"timestamp": 8,  "type": "speech_detected",
         "data": {"speaker": "user",  "duration_ms": 1000, "words": 10}},
        {"timestamp": 12, "type": "call_end"},
    ]
    result = normalize_events(raw)
    assert [e["type"] for e in result] == ["call_start", "agent_speech", "user_speech", "call_end"]
