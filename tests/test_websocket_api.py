"""Tests for RoomMind WebSocket API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.roommind.const import DOMAIN
from custom_components.roommind.websocket_api import (
    websocket_delete_room,
    websocket_get_settings,
    websocket_list_rooms,
    websocket_override_clear,
    websocket_override_set,
    websocket_save_room,
    websocket_save_settings,
    websocket_thermal_reset,
    websocket_thermal_reset_all,
)

# The HA @async_response decorator wraps async handlers into synchronous
# schedulers.  Access the original coroutine via ``__wrapped__`` so we can
# ``await`` them directly in tests without needing a running event-loop task
# factory on the mock hass object.
_list_rooms = websocket_list_rooms.__wrapped__
_save_room = websocket_save_room.__wrapped__
_delete_room = websocket_delete_room.__wrapped__
_override_set = websocket_override_set.__wrapped__
_override_clear = websocket_override_clear.__wrapped__
_get_settings = websocket_get_settings.__wrapped__
_save_settings = websocket_save_settings.__wrapped__
_thermal_reset = websocket_thermal_reset.__wrapped__
_thermal_reset_all = websocket_thermal_reset_all.__wrapped__


@pytest.fixture
def connection():
    """Return a mocked WebSocket connection."""
    conn = MagicMock()
    conn.send_result = MagicMock()
    conn.send_error = MagicMock()
    return conn


@pytest.fixture
def ws_hass(hass, store):
    """Return a hass instance with the store wired into hass.data."""
    hass.data[DOMAIN] = {"store": store}
    return hass


@pytest.mark.asyncio
async def test_list_rooms_empty(ws_hass, store, connection):
    """Listing rooms on a fresh store returns an empty dict."""
    await store.async_load()

    msg = {"id": 1, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(1, {
        "rooms": {},
        "outdoor_temp": None,
        "outdoor_humidity": None,
        "vacation_active": False,
        "vacation_temp": None,
        "vacation_until": None,
        "hidden_rooms": [],
        "control_mode": "bangbang",
        "presence_enabled": False,
        "presence_persons": [],
        "anyone_home": True,
    })


@pytest.mark.asyncio
async def test_save_room_creates_new(ws_hass, store, connection):
    """Saving a room with a new area_id creates the room with defaults."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "living_room",
        "thermostats": ["climate.living_room_trv"],
        "temperature_sensor": "sensor.living_room_temp",
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    assert call_args[0][0] == 2
    room = call_args[0][1]["room"]
    assert room["area_id"] == "living_room"
    assert room["thermostats"] == ["climate.living_room_trv"]
    assert room["temperature_sensor"] == "sensor.living_room_temp"
    # Defaults for fields not provided
    assert room["acs"] == []
    assert room["climate_mode"] == "auto"
    assert room["schedules"] == []
    assert room["schedule_selector_entity"] == ""
    assert room["comfort_temp"] == 21.0
    assert room["eco_temp"] == 17.0


@pytest.mark.asyncio
async def test_save_room_updates_existing(ws_hass, store, connection):
    """Saving a room with an existing area_id updates only the provided fields."""
    await store.async_load()

    # First create a room
    create_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "office",
        "thermostats": ["climate.office_trv"],
        "temperature_sensor": "sensor.office_temp",
        "climate_mode": "heat_only",
    }
    await _save_room(ws_hass, connection, create_msg)
    connection.send_result.reset_mock()

    # Now update it - only change thermostats
    update_msg = {
        "id": 3,
        "type": "roommind/rooms/save",
        "area_id": "office",
        "thermostats": ["climate.office_trv", "climate.office_trv_2"],
    }
    await _save_room(ws_hass, connection, update_msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    assert call_args[0][0] == 3
    room = call_args[0][1]["room"]
    assert room["area_id"] == "office"
    assert room["thermostats"] == ["climate.office_trv", "climate.office_trv_2"]
    # Fields not in update_msg should be preserved
    assert room["temperature_sensor"] == "sensor.office_temp"
    assert room["climate_mode"] == "heat_only"


@pytest.mark.asyncio
async def test_list_rooms_after_save(ws_hass, store, connection):
    """After saving a room, list_rooms includes it with live state."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
        "thermostats": ["climate.kitchen_trv"],
        "temperature_sensor": "sensor.kitchen_temp",
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    # Add coordinator mock before listing so live state can be merged
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}  # No live data yet
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    list_msg = {"id": 3, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, list_msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    assert call_args[0][0] == 3
    rooms = call_args[0][1]["rooms"]
    assert len(rooms) == 1
    assert "kitchen" in rooms
    room = rooms["kitchen"]
    assert room["thermostats"] == ["climate.kitchen_trv"]
    assert "live" in room
    assert room["live"]["mode"] == "idle"


@pytest.mark.asyncio
async def test_save_room_display_name_roundtrip(ws_hass, store, connection):
    """display_name is persisted through save and returned in list."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "bedroom",
        "thermostats": ["climate.bedroom_trv"],
        "display_name": "Schlafzimmer OG",
    }
    await _save_room(ws_hass, connection, save_msg)

    call_args = connection.send_result.call_args
    room = call_args[0][1]["room"]
    assert room["display_name"] == "Schlafzimmer OG"
    connection.send_result.reset_mock()

    # Verify it comes back in list_rooms
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {}
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    list_msg = {"id": 3, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, list_msg)

    rooms = connection.send_result.call_args[0][1]["rooms"]
    assert rooms["bedroom"]["display_name"] == "Schlafzimmer OG"


@pytest.mark.asyncio
async def test_save_room_display_name_defaults_empty(ws_hass, store, connection):
    """Rooms created without display_name default to empty string."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
        "thermostats": ["climate.kitchen_trv"],
    }
    await _save_room(ws_hass, connection, save_msg)

    room = connection.send_result.call_args[0][1]["room"]
    assert room["display_name"] == ""


@pytest.mark.asyncio
async def test_save_room_with_schedules(ws_hass, store, connection):
    """Saving a room with schedules persists the reference."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "bedroom",
        "thermostats": ["climate.bedroom_trv"],
        "temperature_sensor": "sensor.bedroom_temp",
        "schedules": [{"entity_id": "schedule.bedroom_heating"}],
        "schedule_selector_entity": "",
        "comfort_temp": 22.0,
        "eco_temp": 18.0,
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["schedules"] == [{"entity_id": "schedule.bedroom_heating"}]
    assert room["schedule_selector_entity"] == ""
    assert room["comfort_temp"] == 22.0
    assert room["eco_temp"] == 18.0


@pytest.mark.asyncio
async def test_delete_room(ws_hass, store, connection):
    """Deleting a room removes it from the store."""
    await store.async_load()

    # First create a room
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "garage",
        "thermostats": ["climate.garage_trv"],
        "temperature_sensor": "sensor.garage_temp",
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    # Now delete it
    delete_msg = {
        "id": 3,
        "type": "roommind/rooms/delete",
        "area_id": "garage",
    }
    await _delete_room(ws_hass, connection, delete_msg)

    connection.send_result.assert_called_once_with(3, {"success": True})

    # Verify room is gone
    assert store.get_rooms() == {}


@pytest.mark.asyncio
async def test_delete_nonexistent_room_sends_error(ws_hass, store, connection):
    """Deleting a room that doesn't exist sends an error."""
    await store.async_load()

    delete_msg = {
        "id": 4,
        "type": "roommind/rooms/delete",
        "area_id": "nonexistent_area",
    }
    await _delete_room(ws_hass, connection, delete_msg)

    connection.send_error.assert_called_once()
    call_args = connection.send_error.call_args
    assert call_args[0][0] == 4
    assert call_args[0][1] == "not_found"


@pytest.mark.asyncio
async def test_save_room_minimal_only_area_id(ws_hass, store, connection):
    """Saving with only area_id creates a room with all defaults."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "hallway",
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["area_id"] == "hallway"
    assert room["thermostats"] == []
    assert room["acs"] == []
    assert room["temperature_sensor"] == ""
    assert room["humidity_sensor"] == ""
    assert room["climate_mode"] == "auto"
    assert room["schedules"] == []
    assert room["schedule_selector_entity"] == ""
    assert room["comfort_temp"] == 21.0
    assert room["eco_temp"] == 17.0


@pytest.mark.asyncio
async def test_save_room_notifies_coordinator(ws_hass, store, connection):
    """Saving a room notifies the coordinator via async_room_added."""
    await store.async_load()

    mock_coordinator = MagicMock()
    mock_coordinator.async_room_added = AsyncMock()
    # hasattr check used by _get_coordinator
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "balcony",
        "thermostats": ["climate.balcony_trv"],
    }
    await _save_room(ws_hass, connection, msg)

    mock_coordinator.async_room_added.assert_called_once()
    room = mock_coordinator.async_room_added.call_args[0][0]
    assert room["area_id"] == "balcony"


@pytest.mark.asyncio
async def test_delete_room_notifies_coordinator(ws_hass, store, connection):
    """Deleting a room notifies the coordinator via async_room_removed."""
    await store.async_load()

    # First create the room
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "cellar",
    }
    await _save_room(ws_hass, connection, save_msg)

    mock_coordinator = MagicMock()
    mock_coordinator.async_room_removed = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    delete_msg = {
        "id": 3,
        "type": "roommind/rooms/delete",
        "area_id": "cellar",
    }
    await _delete_room(ws_hass, connection, delete_msg)

    mock_coordinator.async_room_removed.assert_called_once_with("cellar")


@pytest.mark.asyncio
async def test_override_set_boost(ws_hass, store, connection):
    """Setting a boost override uses the room's comfort_temp."""
    await store.async_load()

    # Create room first
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "living",
        "thermostats": ["climate.living"],
        "temperature_sensor": "sensor.living_temp",
        "comfort_temp": 22.0,
        "eco_temp": 17.0,
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "living",
        "override_type": "boost",
        "duration": 2.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(3, {"success": True})

    room = store.get_room("living")
    assert room["override_temp"] == 22.0
    assert room["override_type"] == "boost"
    assert room["override_until"] is not None


@pytest.mark.asyncio
async def test_override_set_eco(ws_hass, store, connection):
    """Setting an eco override uses the room's eco_temp."""
    await store.async_load()

    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "bed",
        "comfort_temp": 22.0,
        "eco_temp": 16.0,
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "bed",
        "override_type": "eco",
        "duration": 4.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(3, {"success": True})
    room = store.get_room("bed")
    assert room["override_temp"] == 16.0
    assert room["override_type"] == "eco"


@pytest.mark.asyncio
async def test_override_set_custom(ws_hass, store, connection):
    """Setting a custom override uses the provided temperature."""
    await store.async_load()

    save_msg = {"id": 2, "type": "roommind/rooms/save", "area_id": "office"}
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "office",
        "override_type": "custom",
        "temperature": 24.5,
        "duration": 1.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(3, {"success": True})
    room = store.get_room("office")
    assert room["override_temp"] == 24.5
    assert room["override_type"] == "custom"


@pytest.mark.asyncio
async def test_override_set_custom_without_temp_errors(ws_hass, store, connection):
    """Custom override without temperature sends an error."""
    await store.async_load()

    save_msg = {"id": 2, "type": "roommind/rooms/save", "area_id": "hall"}
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()
    connection.send_error.reset_mock()

    msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "hall",
        "override_type": "custom",
        "duration": 1.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "invalid"


@pytest.mark.asyncio
async def test_override_clear(ws_hass, store, connection):
    """Clearing an override removes override fields."""
    await store.async_load()

    save_msg = {"id": 2, "type": "roommind/rooms/save", "area_id": "bath"}
    await _save_room(ws_hass, connection, save_msg)

    # Set override
    set_msg = {
        "id": 3,
        "type": "roommind/override/set",
        "area_id": "bath",
        "override_type": "boost",
        "duration": 2.0,
    }
    await _override_set(ws_hass, connection, set_msg)
    connection.send_result.reset_mock()

    # Clear it
    clear_msg = {
        "id": 4,
        "type": "roommind/override/clear",
        "area_id": "bath",
    }
    await _override_clear(ws_hass, connection, clear_msg)

    connection.send_result.assert_called_once_with(4, {"success": True})
    room = store.get_room("bath")
    assert room.get("override_temp") is None
    assert room.get("override_until") is None
    assert room.get("override_type") is None


@pytest.mark.asyncio
async def test_override_set_nonexistent_room_errors(ws_hass, store, connection):
    """Setting override on nonexistent room sends an error."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/override/set",
        "area_id": "nope",
        "override_type": "boost",
        "duration": 1.0,
    }
    await _override_set(ws_hass, connection, msg)

    connection.send_error.assert_called_once()
    assert connection.send_error.call_args[0][1] == "not_found"


@pytest.mark.asyncio
async def test_save_room_with_multiple_schedules_and_selector(
    ws_hass, store, connection
):
    """Saving with 2 schedules and a selector entity persists correctly."""
    await store.async_load()

    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "wohnzimmer",
        "thermostats": ["climate.wz_trv"],
        "temperature_sensor": "sensor.wz_temp",
        "schedules": [
            {"entity_id": "schedule.morning"},
            {"entity_id": "schedule.evening"},
        ],
        "schedule_selector_entity": "input_boolean.schedule_toggle",
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["schedules"] == [
        {"entity_id": "schedule.morning"},
        {"entity_id": "schedule.evening"},
    ]
    assert room["schedule_selector_entity"] == "input_boolean.schedule_toggle"


@pytest.mark.asyncio
async def test_list_rooms_includes_active_schedule_index(
    ws_hass, store, connection
):
    """Verify active_schedule_index appears in live data from list_rooms."""
    await store.async_load()

    # Create a room
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "buero",
        "thermostats": ["climate.buero_trv"],
        "temperature_sensor": "sensor.buero_temp",
        "schedules": [{"entity_id": "schedule.buero"}],
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    # Add coordinator mock with live data including active_schedule_index
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {
        "buero": {
            "current_temp": 20.0,
            "target_temp": 21.0,
            "mode": "heating",
            "active_schedule_index": 0,
        }
    }
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    list_msg = {"id": 3, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, list_msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    rooms = call_args[0][1]["rooms"]
    assert "buero" in rooms
    live = rooms["buero"]["live"]
    assert "active_schedule_index" in live
    assert live["active_schedule_index"] == 0


@pytest.mark.asyncio
async def test_save_room_with_window_sensors(ws_hass, store, connection):
    """Saving a room with window_sensors persists them; default is empty list."""
    await store.async_load()

    # Save a room WITH window_sensors
    msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "kitchen",
        "thermostats": ["climate.kitchen_trv"],
        "temperature_sensor": "sensor.kitchen_temp",
        "window_sensors": ["binary_sensor.kitchen_window"],
    }
    await _save_room(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    room = connection.send_result.call_args[0][1]["room"]
    assert room["window_sensors"] == ["binary_sensor.kitchen_window"]
    connection.send_result.reset_mock()

    # Save a room WITHOUT window_sensors — should default to []
    msg2 = {
        "id": 3,
        "type": "roommind/rooms/save",
        "area_id": "hallway",
        "thermostats": ["climate.hallway_trv"],
    }
    await _save_room(ws_hass, connection, msg2)

    connection.send_result.assert_called_once()
    room2 = connection.send_result.call_args[0][1]["room"]
    assert room2["window_sensors"] == []


@pytest.mark.asyncio
async def test_list_rooms_includes_window_open(ws_hass, store, connection):
    """Verify window_open appears in live data from list_rooms."""
    await store.async_load()

    # Create a room
    save_msg = {
        "id": 2,
        "type": "roommind/rooms/save",
        "area_id": "wohnzimmer",
        "thermostats": ["climate.wz_trv"],
        "temperature_sensor": "sensor.wz_temp",
        "window_sensors": ["binary_sensor.wz_window"],
    }
    await _save_room(ws_hass, connection, save_msg)
    connection.send_result.reset_mock()

    # Add coordinator mock with live data including window_open
    mock_coordinator = MagicMock()
    mock_coordinator.rooms = {
        "wohnzimmer": {
            "current_temp": 20.0,
            "target_temp": 21.0,
            "mode": "idle",
            "window_open": True,
            "active_schedule_index": 0,
        }
    }
    mock_coordinator.async_request_refresh = AsyncMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator

    list_msg = {"id": 3, "type": "roommind/rooms/list"}
    await _list_rooms(ws_hass, connection, list_msg)

    connection.send_result.assert_called_once()
    call_args = connection.send_result.call_args
    rooms = call_args[0][1]["rooms"]
    assert "wohnzimmer" in rooms
    live = rooms["wohnzimmer"]["live"]
    assert "window_open" in live
    assert live["window_open"] is True


@pytest.mark.asyncio
async def test_get_settings_empty(ws_hass, store, connection):
    """Getting settings on a fresh store returns empty dict."""
    await store.async_load()

    msg = {"id": 10, "type": "roommind/settings/get"}
    await _get_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(10, {"settings": {}})


@pytest.mark.asyncio
async def test_save_settings(ws_hass, store, connection):
    """Saving outdoor_temp_sensor persists and returns updated settings."""
    await store.async_load()

    msg = {
        "id": 11,
        "type": "roommind/settings/save",
        "outdoor_temp_sensor": "sensor.outdoor",
    }
    await _save_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    result = connection.send_result.call_args[0][1]
    assert result["settings"]["outdoor_temp_sensor"] == "sensor.outdoor"


# ---------------------------------------------------------------------------
# Vacation mode tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_settings_vacation(ws_hass, store, connection):
    """Saving vacation fields persists and returns updated settings."""
    await store.async_load()

    until = 1771900000.0
    msg = {
        "id": 12,
        "type": "roommind/settings/save",
        "vacation_temp": 15.0,
        "vacation_until": until,
    }
    await _save_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    result = connection.send_result.call_args[0][1]
    assert result["settings"]["vacation_temp"] == 15.0
    assert result["settings"]["vacation_until"] == until


@pytest.mark.asyncio
async def test_save_settings_vacation_clear(ws_hass, store, connection):
    """Setting vacation_until to None clears vacation mode."""
    await store.async_load()

    msg = {
        "id": 13,
        "type": "roommind/settings/save",
        "vacation_until": None,
    }
    await _save_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    result = connection.send_result.call_args[0][1]
    assert result["settings"]["vacation_until"] is None


# ---------------------------------------------------------------------------
# Thermal model reset tests
# ---------------------------------------------------------------------------


def _make_coordinator_with_model(ws_hass):
    """Create a coordinator mock with thermal model data."""
    from custom_components.roommind.thermal_model import RoomModelManager

    mock_coordinator = MagicMock()
    mgr = RoomModelManager()
    mgr.update("room_a", 20.5, 5.0, "heating", 5)
    mgr.update("room_b", 24.5, 30.0, "cooling", 5)
    mock_coordinator._model_manager = mgr
    mock_coordinator._last_temps = {"room_a": 20.5, "room_b": 24.5}
    mock_coordinator._history_store = MagicMock()
    mock_coordinator._history_store.remove_room = MagicMock()
    ws_hass.data[DOMAIN]["coordinator"] = mock_coordinator
    return mock_coordinator


@pytest.mark.asyncio
async def test_thermal_reset_room(ws_hass, store, connection):
    """Resetting one room clears its model but keeps others."""
    await store.async_load()
    await store.async_save_thermal_data({"room_a": {"n_samples": 10}, "room_b": {"n_samples": 5}})

    coordinator = _make_coordinator_with_model(ws_hass)

    msg = {"id": 20, "type": "roommind/thermal/reset", "area_id": "room_a"}
    await _thermal_reset(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(20, {"success": True})

    # room_a cleared from model manager
    assert "room_a" not in coordinator._model_manager._estimators
    # room_b still present
    assert "room_b" in coordinator._model_manager._estimators
    # History cleared for room_a
    coordinator._history_store.remove_room.assert_called_once_with("room_a")
    # Persisted thermal data cleared for room_a
    assert "room_a" not in store.get_thermal_data()
    assert "room_b" in store.get_thermal_data()


@pytest.mark.asyncio
async def test_thermal_reset_all(ws_hass, store, connection):
    """Resetting all rooms clears all models and history."""
    await store.async_load()
    await store.async_save_thermal_data({"room_a": {"n_samples": 10}, "room_b": {"n_samples": 5}})

    coordinator = _make_coordinator_with_model(ws_hass)

    msg = {"id": 21, "type": "roommind/thermal/reset_all"}
    await _thermal_reset_all(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(21, {"success": True})

    # All estimators cleared
    assert len(coordinator._model_manager._estimators) == 0
    # History cleared for all rooms
    assert coordinator._history_store.remove_room.call_count == 2
    # Persisted thermal data empty
    assert store.get_thermal_data() == {}
    # last_temps cleared
    assert len(coordinator._last_temps) == 0


@pytest.mark.asyncio
async def test_thermal_reset_nonexistent_room(ws_hass, store, connection):
    """Resetting a room that has no model data still succeeds (idempotent)."""
    await store.async_load()

    coordinator = _make_coordinator_with_model(ws_hass)

    msg = {"id": 22, "type": "roommind/thermal/reset", "area_id": "nonexistent"}
    await _thermal_reset(ws_hass, connection, msg)

    connection.send_result.assert_called_once_with(22, {"success": True})


# --- Mold risk settings tests ---


@pytest.mark.asyncio
async def test_save_settings_mold_fields(ws_hass, store, connection):
    """Mold detection/prevention settings should be accepted and persisted."""
    await store.async_load()

    msg = {
        "id": 30,
        "type": "roommind/settings/save",
        "mold_detection_enabled": True,
        "mold_humidity_threshold": 65.0,
        "mold_sustained_minutes": 15,
        "mold_notification_cooldown": 30,
        "mold_notifications_enabled": True,
        "mold_notification_targets": [
            {"entity_id": "notify.mobile_app_kevin", "person_entity": "person.kevin", "notify_when": "always"},
        ],
        "mold_prevention_enabled": True,
        "mold_prevention_intensity": "strong",
        "mold_prevention_notify_enabled": True,
        "mold_prevention_notify_targets": [],
    }
    await _save_settings(ws_hass, connection, msg)

    connection.send_result.assert_called_once()
    result = connection.send_result.call_args[0][1]
    settings = result["settings"]
    assert settings["mold_detection_enabled"] is True
    assert settings["mold_humidity_threshold"] == 65.0
    assert settings["mold_sustained_minutes"] == 15
    assert settings["mold_notification_cooldown"] == 30
    assert settings["mold_notifications_enabled"] is True
    assert len(settings["mold_notification_targets"]) == 1
    assert settings["mold_prevention_enabled"] is True
    assert settings["mold_prevention_intensity"] == "strong"


@pytest.mark.asyncio
async def test_save_settings_mold_partial_update(ws_hass, store, connection):
    """Updating only one mold field should not affect others (merge behavior)."""
    await store.async_load()

    # First save all fields
    msg1 = {
        "id": 31,
        "type": "roommind/settings/save",
        "mold_detection_enabled": True,
        "mold_humidity_threshold": 75.0,
    }
    await _save_settings(ws_hass, connection, msg1)
    connection.send_result.reset_mock()

    # Now save only one field
    msg2 = {
        "id": 32,
        "type": "roommind/settings/save",
        "mold_prevention_enabled": True,
    }
    await _save_settings(ws_hass, connection, msg2)

    result = connection.send_result.call_args[0][1]
    settings = result["settings"]
    # Original fields should still be there
    assert settings["mold_detection_enabled"] is True
    assert settings["mold_humidity_threshold"] == 75.0
    # New field should be set
    assert settings["mold_prevention_enabled"] is True


@pytest.mark.asyncio
async def test_compute_target_forecast_includes_mold_delta(ws_hass):
    """_compute_target_forecast should add mold_prevention_delta to all targets."""
    from custom_components.roommind.websocket_api import _compute_target_forecast

    room = {"comfort_temp": 21.0, "eco_temp": 17.0, "schedules": []}
    settings: dict = {}

    # Without delta
    forecast_base = await _compute_target_forecast(ws_hass, room, settings)
    assert forecast_base[0]["target_temp"] == 21.0

    # With delta
    forecast_mold = await _compute_target_forecast(
        ws_hass, room, settings, mold_prevention_delta=2.0,
    )
    assert forecast_mold[0]["target_temp"] == 23.0

    # All forecast points should have the delta applied
    for point in forecast_mold:
        assert point["target_temp"] == 23.0


