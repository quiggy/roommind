"""Tests for the CSV-based history store."""

from __future__ import annotations

import os
import time

import pytest

from custom_components.roommind.history_store import HistoryStore


@pytest.fixture
def history_dir(tmp_path):
    return str(tmp_path / "history")


def test_write_and_read(history_dir):
    """Write one record and read it back."""
    store = HistoryStore(history_dir)
    store.record("living_room", {
        "room_temp": 21.0,
        "outdoor_temp": 5.0,
        "target_temp": 21.0,
        "mode": "idle",
        "predicted_temp": 21.1,
    })
    rows = store.read_detail("living_room")
    assert len(rows) == 1
    assert rows[0]["room_temp"] == "21.0"


def test_multiple_rooms(history_dir):
    """Each room has separate CSV files."""
    store = HistoryStore(history_dir)
    store.record("room_a", {"room_temp": 20.0, "outdoor_temp": 5.0, "target_temp": 21.0, "mode": "heating", "predicted_temp": 20.5})
    store.record("room_b", {"room_temp": 25.0, "outdoor_temp": 30.0, "target_temp": 23.0, "mode": "cooling", "predicted_temp": 24.0})
    assert len(store.read_detail("room_a")) == 1
    assert len(store.read_detail("room_b")) == 1


def test_creates_directory(history_dir):
    """Directory is created on first write."""
    store = HistoryStore(history_dir)
    store.record("test_room", {"room_temp": 20.0, "outdoor_temp": 5.0, "target_temp": 21.0, "mode": "idle", "predicted_temp": 20.0})
    assert os.path.isdir(history_dir)


def test_read_empty_room(history_dir):
    """Reading nonexistent room returns empty list."""
    store = HistoryStore(history_dir)
    rows = store.read_detail("nonexistent")
    assert rows == []


def test_remove_room(history_dir):
    """remove_room deletes all files for that room."""
    store = HistoryStore(history_dir)
    store.record("room_a", {"room_temp": 20.0, "outdoor_temp": 5.0, "target_temp": 21.0, "mode": "idle", "predicted_temp": 20.0})
    store.remove_room("room_a")
    assert store.read_detail("room_a") == []


def test_multiple_records(history_dir):
    """Multiple records are stored and read back."""
    store = HistoryStore(history_dir)
    base = time.time()
    for i in range(10):
        store.record("room_a", {
            "room_temp": 20.0 + i * 0.1,
            "outdoor_temp": 5.0,
            "target_temp": 21.0,
            "mode": "heating",
            "predicted_temp": 20.0 + i * 0.1,
        }, timestamp=base + i * 60)
    detail_rows = store.read_detail("room_a")
    assert len(detail_rows) == 10


def test_window_open_in_csv(history_dir):
    """window_open field is written and read correctly."""
    store = HistoryStore(history_dir)
    store.record("room_a", {
        "room_temp": 20.0,
        "outdoor_temp": 5.0,
        "target_temp": 21.0,
        "mode": "idle",
        "predicted_temp": 20.0,
        "window_open": True,
    })
    store.record("room_a", {
        "room_temp": 20.0,
        "outdoor_temp": 5.0,
        "target_temp": 21.0,
        "mode": "idle",
        "predicted_temp": 20.0,
        "window_open": False,
    })
    rows = store.read_detail("room_a")
    assert len(rows) == 2
    assert rows[0]["window_open"] == "True"
    assert rows[1]["window_open"] == "False"


def test_downsample_preserves_window_open(history_dir):
    """_downsample takes first window_open value from each bucket."""
    store = HistoryStore(history_dir)
    rows = [
        {"timestamp": "1000", "room_temp": "20.0", "outdoor_temp": "5.0",
         "target_temp": "21.0", "mode": "idle", "predicted_temp": "20.0", "window_open": "True"},
        {"timestamp": "1060", "room_temp": "20.0", "outdoor_temp": "5.0",
         "target_temp": "21.0", "mode": "idle", "predicted_temp": "20.0", "window_open": "True"},
        {"timestamp": "1500", "room_temp": "21.0", "outdoor_temp": "5.0",
         "target_temp": "21.0", "mode": "heating", "predicted_temp": "21.0", "window_open": "False"},
    ]
    result = store._downsample(rows, bucket_seconds=300)
    # Two buckets: 0-300s and 300-600s
    assert len(result) >= 1
    assert result[0]["window_open"] == "True"
    # Second bucket should have window_open=False
    if len(result) > 1:
        assert result[1]["window_open"] == "False"


def test_rotate_moves_old_to_history(history_dir):
    """Rotation moves old detail rows to history file."""
    store = HistoryStore(history_dir)
    now = time.time()
    # Write rows that are "old" (> 48h ago)
    old_ts = now - 50 * 3600  # 50 hours ago
    for i in range(5):
        store.record("room_a", {
            "room_temp": 20.0 + i * 0.1,
            "outdoor_temp": 5.0,
            "target_temp": 21.0,
            "mode": "heating",
            "predicted_temp": 20.0,
        }, timestamp=old_ts + i * 60)
    # Write some recent rows
    for i in range(3):
        store.record("room_a", {
            "room_temp": 21.0,
            "outdoor_temp": 5.0,
            "target_temp": 21.0,
            "mode": "idle",
            "predicted_temp": 21.0,
        }, timestamp=now + i * 60)

    store.rotate("room_a")

    detail = store.read_detail("room_a")
    history = store.read_history("room_a")
    assert len(detail) == 3  # only recent rows remain
    assert len(history) >= 1  # old rows downsampled to history
