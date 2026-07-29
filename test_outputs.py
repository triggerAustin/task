"""Verifier for the multi-node log reconciliation task.

Every assertion traces to a requirement stated in instruction.md. Tests fall into
two groups: exact comparisons against frozen ground truth in tests/expected/, and
self-consistency checks recomputed from the agent's own output so that the CSV
must agree with the timeline it was derived from.
"""

import csv
import json
import math
import os
import re
from datetime import datetime, timezone

import pytest

OUTPUT_ROOT = "/app/output"
TIMELINE = "/app/output/global_timeline.json"
SESSIONS = "/app/output/session_anomalies.csv"
EXPECTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected")

EVENT_KEYS = {"event_id", "node_id", "user_id", "timestamp", "latency_ms", "z_score"}
CSV_HEADER = ["session_id", "total_events", "duration_seconds", "anomaly_flag"]
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

SESSION_GAP_MS = 300_000
Z_THRESHOLD = 2.5
Z_TOL = 1e-6        # accepts a correctly rounded score or an unrounded one
DURATION_TOL = 1e-6
LATENCY_TOL = 1e-9


def millis(stamp):
    """Parse an emitted ISO-8601 millisecond timestamp into integer epoch millis."""
    dt = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return round(dt.timestamp() * 1000)


def graded_open(path):
    """Open a graded artifact, refusing anything that is not a real file under /app.

    A submission must not be able to point an output path at the verifier's own
    reference data. Every component of the path is resolved, so neither the file
    nor any parent directory may be a symlink that escapes /app.
    """
    assert os.path.lexists(path), f"missing {path}"
    assert not os.path.islink(path), (
        f"{path} is a symbolic link; graded outputs must be regular files")
    real = os.path.realpath(path)
    assert real == os.path.abspath(path), (
        f"{path} resolves through a link to {real}; graded outputs must be real "
        f"files at the stated path")
    assert real.startswith(OUTPUT_ROOT + os.sep), f"{path} resolves outside {OUTPUT_ROOT}: {real}"
    assert os.path.isfile(real) and not os.path.isdir(real), f"{path} is not a regular file"
    return open(real, encoding="utf-8", newline="")


def is_number(value):
    """True for a real JSON number: finite, and not a bool masquerading as an int."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


@pytest.fixture(scope="module")
def timeline():
    with graded_open(TIMELINE) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def events(timeline):
    return timeline["events"]


@pytest.fixture(scope="module")
def sessions():
    with graded_open(SESSIONS) as fh:
        return list(csv.reader(fh))


@pytest.fixture(scope="module")
def expected_timeline():
    with open(os.path.join(EXPECTED, "global_timeline.json"), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def expected_sessions():
    with open(os.path.join(EXPECTED, "session_anomalies.csv"), encoding="utf-8",
              newline="") as fh:
        return list(csv.reader(fh))


# --------------------------------------------------------------- output files

def test_output_files_exist():
    """Both artifacts are real files at the stated paths, not links to other data."""
    for path in (TIMELINE, SESSIONS):
        graded_open(path).close()


def test_timeline_top_level_schema(timeline):
    """global_timeline.json is an object carrying quarantined_count and events."""
    assert isinstance(timeline, dict), "timeline must be a JSON object"
    assert set(timeline) == {"quarantined_count", "events"}, (
        f"unexpected top-level keys: {sorted(timeline)}")
    assert isinstance(timeline["quarantined_count"], int)
    assert isinstance(timeline["events"], list)


def test_event_record_schema(events):
    """Each event carries exactly the six normative fields with the right types."""
    assert events, "timeline contains no events"
    for event in events:
        assert set(event) == EVENT_KEYS, f"bad key set on {event.get('event_id')}: {sorted(event)}"
        assert isinstance(event["event_id"], str) and event["event_id"]
        assert isinstance(event["node_id"], str) and event["node_id"]
        assert isinstance(event["user_id"], str) and event["user_id"]
        assert event["latency_ms"] is None or is_number(event["latency_ms"]), (
            f"{event['event_id']}: latency_ms {event['latency_ms']!r} is not a finite number")
        assert event["z_score"] is None or is_number(event["z_score"]), (
            f"{event['event_id']}: z_score {event['z_score']!r} is not a finite number")


def test_timestamp_format(events):
    """Corrected timestamps are ISO-8601 UTC with exactly millisecond precision."""
    bad = [e["event_id"] for e in events if not TS_RE.match(e["timestamp"])]
    assert not bad, f"{len(bad)} malformed timestamps, e.g. {bad[:5]}"


# ------------------------------------------------------------ ingest / dedupe

def test_quarantined_count(timeline, expected_timeline):
    """Unparsable, incomplete and duplicate-id records are counted, not emitted."""
    assert timeline["quarantined_count"] == expected_timeline["quarantined_count"]


def test_event_ids_unique(events):
    """Only the first record bearing a given event_id survives quarantine."""
    ids = [e["event_id"] for e in events]
    assert len(ids) == len(set(ids)), "duplicate event_id in the emitted timeline"


def test_event_set_matches(events, expected_timeline):
    """Exactly the recoverable records appear, across all three log formats."""
    got = {e["event_id"] for e in events}
    want = {e["event_id"] for e in expected_timeline["events"]}
    assert got == want, (
        f"{len(want - got)} missing (e.g. {sorted(want - got)[:5]}), "
        f"{len(got - want)} unexpected (e.g. {sorted(got - want)[:5]})")


def test_node_attribution(events, expected_timeline):
    """Each event is attributed to the node directory its log file came from."""
    want = {e["event_id"]: e["node_id"] for e in expected_timeline["events"]}
    bad = [e["event_id"] for e in events if want.get(e["event_id"]) != e["node_id"]]
    assert not bad, f"{len(bad)} events on the wrong node, e.g. {bad[:5]}"


def test_user_alias_resolution(events, expected_timeline):
    """uid / user_id / account all normalise onto the same user_id field."""
    want = {e["event_id"]: e["user_id"] for e in expected_timeline["events"]}
    bad = [e["event_id"] for e in events if want.get(e["event_id"]) != e["user_id"]]
    assert not bad, f"{len(bad)} events with the wrong user_id, e.g. {bad[:5]}"


# ------------------------------------------------------- clock correction

def test_corrected_timestamps(events, expected_timeline):
    """Local clocks are mapped onto node_alpha's frame by the fitted linear model."""
    want = {e["event_id"]: e["timestamp"] for e in expected_timeline["events"]}
    bad = [(e["event_id"], e["timestamp"], want.get(e["event_id"]))
           for e in events if want.get(e["event_id"]) != e["timestamp"]]
    assert not bad, f"{len(bad)} wrong corrected timestamps, e.g. {bad[:3]}"


def test_timeline_ordering(events, expected_timeline):
    """Events appear in the order the stated rule produces.

    Ordering is by full-precision corrected time, which the emitted millisecond
    timestamps cannot express, so the sequence is compared against ground truth
    rather than merely checked for monotonicity.
    """
    got = [e["event_id"] for e in events]
    want = [e["event_id"] for e in expected_timeline["events"]]
    if got != want:
        first = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        raise AssertionError(
            f"timeline order diverges at index {first}: got {got[first]!r}, "
            f"expected {want[first]!r}")

    keys = [(millis(e["timestamp"]), e["event_id"]) for e in events]
    assert keys == sorted(keys), "emitted timestamps are not non-decreasing"


# ------------------------------------------------------------------ latency

def test_latency_values(events, expected_timeline):
    """Recorded latencies are carried through; unreadable ones become null."""
    want = {e["event_id"]: e["latency_ms"] for e in expected_timeline["events"]}
    for event in events:
        exp = want.get(event["event_id"])
        got = event["latency_ms"]
        if exp is None or got is None:
            assert exp == got, f"{event['event_id']}: latency {got!r}, expected {exp!r}"
        else:
            assert is_number(got), f"{event['event_id']}: latency {got!r} is not finite"
            assert abs(got - exp) <= LATENCY_TOL, f"{event['event_id']}: {got} != {exp}"


def test_null_latency_has_null_zscore(events):
    """A record with no latency carries no score; the two are null together."""
    bad = [e["event_id"] for e in events
           if (e["latency_ms"] is None) != (e["z_score"] is None)]
    assert not bad, f"{len(bad)} events where latency/z_score nullity disagrees: {bad[:5]}"


def test_z_scores(events, expected_timeline):
    """Trailing 20-record sample z-scores match, including the short-window zeros."""
    want = {e["event_id"]: e["z_score"] for e in expected_timeline["events"]}
    bad = []
    for event in events:
        exp, got = want.get(event["event_id"]), event["z_score"]
        if exp is None or got is None:
            if exp != got:
                bad.append((event["event_id"], got, exp))
        elif not is_number(got) or abs(got - exp) > Z_TOL:
            bad.append((event["event_id"], got, exp))
    assert not bad, f"{len(bad)} wrong z_scores, e.g. {bad[:3]}"


def test_zero_variance_window_scores_zero(events, expected_timeline):
    """A full window whose latencies are all identical scores 0.0, not a divide-by-zero.

    Targets only events whose trailing window is both complete and flat, so this is
    independent of the short-window rule that also yields 0.0.
    """
    window, flat = [], []
    for event in expected_timeline["events"]:
        if event["latency_ms"] is None:
            continue
        window.append(event["latency_ms"])
        if len(window) > 20:
            window.pop(0)
        if len(window) == 20 and len(set(window)) == 1:
            flat.append(event["event_id"])
    assert flat, "fixture exposes no complete zero-variance window"

    got = {e["event_id"]: e["z_score"] for e in events}
    bad = [(eid, got.get(eid)) for eid in flat if got.get(eid) != 0.0]
    assert not bad, f"{len(bad)} flat windows not scored 0.0, e.g. {bad[:5]}"


# ------------------------------------------------------------------ sessions

def test_csv_header(sessions):
    """session_anomalies.csv carries exactly the normative header and four columns per row."""
    assert sessions, "session_anomalies.csv is empty"
    assert sessions[0] == CSV_HEADER, f"header is {sessions[0]}"
    wrong = [i for i, row in enumerate(sessions[1:], 2) if len(row) != len(CSV_HEADER)]
    assert not wrong, f"rows with the wrong column count, e.g. line {wrong[:5]}"


def test_csv_sorted_by_session_id(sessions):
    """Session rows ascend by session_id."""
    ids = [row[0] for row in sessions[1:]]
    assert ids == sorted(ids), "session rows are not sorted by session_id"


def test_session_rows_match(sessions, expected_sessions):
    """Sessions split on inactivity gaps over 300 s, and are named from their start."""
    ids = [row[0] for row in sessions[1:]]
    assert len(ids) == len(set(ids)), "session_anomalies.csv repeats a session_id"
    got = set(ids)
    want = {row[0] for row in expected_sessions[1:]}
    assert got == want, (
        f"{len(want - got)} missing (e.g. {sorted(want - got)[:5]}), "
        f"{len(got - want)} unexpected (e.g. {sorted(got - want)[:5]})")


def test_session_event_counts(sessions, expected_sessions):
    """total_events counts the records belonging to each session."""
    want = {row[0]: int(row[1]) for row in expected_sessions[1:]}
    bad = [(row[0], row[1], want[row[0]]) for row in sessions[1:]
           if int(row[1]) != want[row[0]]]
    assert not bad, f"{len(bad)} wrong total_events, e.g. {bad[:3]}"


def test_session_durations(sessions, expected_sessions):
    """duration_seconds spans first to last emitted timestamp, to six decimals."""
    want = {row[0]: float(row[2]) for row in expected_sessions[1:]}
    for row in sessions[1:]:
        decimals = row[2].split(".")[1] if "." in row[2] else ""
        assert len(decimals) == 6, f"{row[0]}: duration {row[2]!r} is not 6 decimals"
        assert abs(float(row[2]) - want[row[0]]) <= DURATION_TOL, (
            f"{row[0]}: duration {row[2]} != {want[row[0]]}")


def test_anomaly_flags(sessions, expected_sessions):
    """A session is flagged exactly when one of its records scores |z| > 2.5."""
    want = {row[0]: row[3] for row in expected_sessions[1:]}
    for row in sessions[1:]:
        assert row[3] in ("true", "false"), f"{row[0]}: anomaly_flag {row[3]!r}"
    bad = [(row[0], row[3], want[row[0]]) for row in sessions[1:] if row[3] != want[row[0]]]
    assert not bad, f"{len(bad)} wrong anomaly_flag, e.g. {bad[:3]}"


# ------------------------------------------- self-consistency (fixture-free)

def test_sessions_consistent_with_timeline(events, sessions):
    """The report is rederivable from the agent's own timeline under the stated rules.

    Recomputes session boundaries, names, counts, durations and flags directly from
    global_timeline.json, so the two artifacts must tell the same story regardless
    of the frozen fixture.
    """
    rebuilt, open_by_user = {}, {}
    for event in events:
        user, stamp = event["user_id"], millis(event["timestamp"])
        current = open_by_user.get(user)
        if current is None or stamp - current["last"] > SESSION_GAP_MS:
            name = f"{user}_{datetime.fromtimestamp(stamp // 1000, timezone.utc):%Y%m%d%H%M%S}"
            current = {"first": stamp, "last": stamp, "count": 0, "anom": False}
            rebuilt[name] = current
            open_by_user[user] = current
        current["last"] = stamp
        current["count"] += 1
        z = event["z_score"]
        if z is not None and abs(z) > Z_THRESHOLD:
            current["anom"] = True

    reported = {row[0]: row for row in sessions[1:]}
    assert set(reported) == set(rebuilt), (
        f"session ids disagree with the timeline: "
        f"{sorted(set(rebuilt) - set(reported))[:5]} missing, "
        f"{sorted(set(reported) - set(rebuilt))[:5]} unexpected")
    for name, info in rebuilt.items():
        row = reported[name]
        assert int(row[1]) == info["count"], f"{name}: total_events {row[1]} != {info['count']}"
        expected_duration = (info["last"] - info["first"]) / 1000.0
        assert abs(float(row[2]) - expected_duration) <= DURATION_TOL, (
            f"{name}: duration {row[2]} != {expected_duration:.6f}")
        assert row[3] == ("true" if info["anom"] else "false"), (
            f"{name}: anomaly_flag {row[3]} contradicts its own events")
