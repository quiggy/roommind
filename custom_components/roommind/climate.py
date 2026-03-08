"""Climate platform for RoomMind."""

from __future__ import annotations

import time
from typing import Any

from homeassistant.components.climate import (
    ClimateEntity,
    ClimateEntityFeature,
    HVACMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DEFAULT_COMFORT_TEMP, DOMAIN, OVERRIDE_CUSTOM
from .coordinator import RoomMindCoordinator


def _create_room_climates(
    coordinator: RoomMindCoordinator,
    area_id: str,
) -> list[ClimateEntity]:
    """Create climate entities for a room."""
    return [RoomMindOverrideClimate(coordinator, area_id)]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up RoomMind climate entities from a config entry."""
    coordinator: RoomMindCoordinator = hass.data[DOMAIN][entry.entry_id]
    store = hass.data[DOMAIN]["store"]
    coordinator.async_add_climate_entities = async_add_entities
    rooms = store.get_rooms()
    entities: list[ClimateEntity] = []
    for area_id in rooms:
        entities.extend(_create_room_climates(coordinator, area_id))
        coordinator._climate_entity_areas.add(area_id)
    if entities:
        async_add_entities(entities)


class RoomMindOverrideClimate(CoordinatorEntity, ClimateEntity):
    """Climate entity for room override control."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:thermometer-alert"
    _attr_hvac_modes = [HVACMode.OFF, HVACMode.AUTO]
    _attr_supported_features = (
        ClimateEntityFeature.TARGET_TEMPERATURE | ClimateEntityFeature.TURN_ON | ClimateEntityFeature.TURN_OFF
    )
    _attr_temperature_unit = UnitOfTemperature.CELSIUS
    _attr_target_temperature_step = 0.5
    _attr_min_temp = 5.0
    _attr_max_temp = 35.0

    def __init__(self, coordinator: RoomMindCoordinator, area_id: str) -> None:
        super().__init__(coordinator)
        self._area_id = area_id
        self._attr_unique_id = f"{DOMAIN}_{area_id}_override"
        self._attr_name = f"{area_id} Override"
        self.entity_id = f"climate.{DOMAIN}_{area_id}_override"

    def _is_override_active(self) -> bool:
        """Return True if override is currently active."""
        store = self.coordinator.hass.data[DOMAIN]["store"]
        room = store.get_room(self._area_id)
        if not room:
            return False
        override_temp = room.get("override_temp")
        if override_temp is None:
            return False
        override_until = room.get("override_until")
        return override_until is None or time.time() < override_until

    @property
    def hvac_mode(self) -> HVACMode:
        """Return AUTO if override is active, OFF otherwise."""
        return HVACMode.AUTO if self._is_override_active() else HVACMode.OFF

    @property
    def target_temperature(self) -> float:
        """Return override temp if active, else DEFAULT_COMFORT_TEMP."""
        if self._is_override_active():
            store = self.coordinator.hass.data[DOMAIN]["store"]
            room = store.get_room(self._area_id)
            if room:
                val = room.get("override_temp")
                if isinstance(val, (int, float)):
                    return float(val)
        return DEFAULT_COMFORT_TEMP

    @property
    def current_temperature(self) -> float | None:
        """Return the room's current temperature from coordinator data."""
        data = self.coordinator.data
        if not data:
            return None
        room_data = data.get("rooms", {}).get(self._area_id)
        if not room_data:
            return None
        val = room_data.get("current_temp")
        return float(val) if isinstance(val, (int, float)) else None

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set override temperature."""
        temperature = kwargs.get("temperature")
        if temperature is None:
            return
        store = self.coordinator.hass.data[DOMAIN]["store"]
        await store.async_update_room(
            self._area_id,
            {
                "override_temp": temperature,
                "override_until": None,
                "override_type": OVERRIDE_CUSTOM,
            },
        )
        await self.coordinator.async_request_refresh()

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set HVAC mode: OFF clears override, AUTO activates."""
        store = self.coordinator.hass.data[DOMAIN]["store"]
        if hvac_mode == HVACMode.OFF:
            await store.async_update_room(
                self._area_id,
                {
                    "override_temp": None,
                    "override_until": None,
                    "override_type": None,
                },
            )
        elif hvac_mode == HVACMode.AUTO:
            if not self._is_override_active():
                await store.async_update_room(
                    self._area_id,
                    {
                        "override_temp": DEFAULT_COMFORT_TEMP,
                        "override_until": None,
                        "override_type": OVERRIDE_CUSTOM,
                    },
                )
        await self.coordinator.async_request_refresh()
