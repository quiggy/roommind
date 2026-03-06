"""Sensor reading utilities for RoomMind."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


def read_sensor_value(
    hass: HomeAssistant,
    entity_id: str | None,
    area_id: str,
    value_name: str,
) -> float | None:
    """Read a numeric sensor value, returning None on failure.

    Parameters
    ----------
    hass:
        Home Assistant instance.
    entity_id:
        The sensor entity to read (e.g. ``sensor.living_room_temp``).
        If *None* or empty, returns *None* immediately.
    area_id:
        Used only for log messages.
    value_name:
        Human-readable name of the value (e.g. "temperature", "humidity")
        used in warning messages.
    """
    if not entity_id:
        return None

    state = hass.states.get(entity_id)
    if state is None or state.state in ("unavailable", "unknown"):
        return None

    try:
        return float(state.state)
    except (ValueError, TypeError):
        _LOGGER.warning(
            "Room '%s': could not parse %s from '%s'",
            area_id,
            value_name,
            state.state,
        )
        return None
