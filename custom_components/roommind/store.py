"""Room persistence layer for RoomMind."""

from __future__ import annotations

import copy

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DEFAULT_COMFORT_TEMP, DEFAULT_ECO_TEMP, DOMAIN

STORAGE_VERSION = 1
STORAGE_KEY = DOMAIN


class RoomMindStore:
    """Manage room configuration storage for RoomMind."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialise the store."""
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self._data: dict[str, dict] = {}
        self._settings: dict = {}
        self._thermal_data: dict = {}

    async def async_load(self) -> None:
        """Load room data from the HA store."""
        stored = await self._store.async_load()
        if stored and "rooms" in stored:
            self._data = stored["rooms"]
        else:
            self._data = {}

        self._settings = stored.get("settings", {}) if stored else {}
        self._thermal_data = stored.get("thermal_data", {}) if stored else {}

    async def _async_save(self) -> None:
        """Persist current room data to the HA store."""
        await self._store.async_save({"rooms": self._data, "settings": self._settings, "thermal_data": self._thermal_data})

    def get_rooms(self) -> dict[str, dict]:
        """Return a deep copy of all rooms."""
        return copy.deepcopy(dict(self._data))

    def get_room(self, area_id: str) -> dict | None:
        """Return a deep copy of a single room by area ID, or None if not found."""
        room = self._data.get(area_id)
        return copy.deepcopy(room) if room is not None else None

    def get_settings(self) -> dict:
        """Return a deep copy of global settings."""
        return copy.deepcopy(dict(self._settings))

    async def async_save_settings(self, changes: dict) -> dict:
        """Merge changes into global settings and persist."""
        self._settings.update(changes)
        await self._async_save()
        return dict(self._settings)

    def get_thermal_data(self) -> dict:
        """Return a deep copy of thermal learning data."""
        return copy.deepcopy(dict(self._thermal_data))

    async def async_save_thermal_data(self, data: dict) -> None:
        """Replace thermal learning data and persist."""
        self._thermal_data = data
        await self._async_save()

    async def async_clear_thermal_data_room(self, area_id: str) -> None:
        """Clear thermal learning data for a single room."""
        self._thermal_data.pop(area_id, None)
        await self._async_save()

    async def async_clear_all_thermal_data(self) -> None:
        """Clear all thermal learning data."""
        self._thermal_data = {}
        await self._async_save()

    async def async_save_room(self, area_id: str, config: dict) -> dict:
        """Create or update room configuration for an area."""
        if area_id in self._data:
            # Update existing
            existing = self._data[area_id]
            for key, value in config.items():
                if key != "area_id":
                    existing[key] = value
            await self._async_save()
            return existing
        else:
            # Create new
            room = {
                "area_id": area_id,
                "thermostats": config.get("thermostats", []),
                "acs": config.get("acs", []),
                "temperature_sensor": config.get("temperature_sensor", ""),
                "humidity_sensor": config.get("humidity_sensor", ""),
                "climate_mode": config.get("climate_mode", "auto"),
                "schedules": config.get("schedules", []),
                "schedule_selector_entity": config.get("schedule_selector_entity", ""),
                "window_sensors": config.get("window_sensors", []),
                "window_open_delay": config.get("window_open_delay", 0),
                "window_close_delay": config.get("window_close_delay", 0),
                "comfort_temp": config.get("comfort_temp", DEFAULT_COMFORT_TEMP),
                "eco_temp": config.get("eco_temp", DEFAULT_ECO_TEMP),
                "presence_persons": config.get("presence_persons", []),
                "display_name": config.get("display_name", ""),
            }
            self._data[area_id] = room
            await self._async_save()
            return room

    async def async_update_room(self, area_id: str, changes: dict) -> dict:
        """Merge changes into an existing room. Raises KeyError if not found."""
        if area_id not in self._data:
            raise KeyError(f"Room '{area_id}' not found")

        # Prevent overriding the area_id
        changes.pop("area_id", None)

        self._data[area_id].update(changes)
        await self._async_save()
        return self._data[area_id]

    async def async_delete_room(self, area_id: str) -> None:
        """Delete a room. Raises KeyError if not found."""
        if area_id not in self._data:
            raise KeyError(f"Room '{area_id}' not found")

        del self._data[area_id]
        await self._async_save()
