"""Presence detection utilities for RoomMind."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


def is_presence_away(hass: HomeAssistant, room: dict, settings: dict) -> bool:
    """Return True if presence detection says all relevant persons are away.

    Per-room persons take precedence over global persons.
    Fail-safe: unavailable/unknown entities are treated as "home".
    """
    if not settings.get("presence_enabled", False):
        return False
    global_persons = settings.get("presence_persons", [])
    if not global_persons:
        return False

    room_persons = room.get("presence_persons", [])
    persons = room_persons if room_persons else global_persons

    for pid in persons:
        state = hass.states.get(pid)
        if state is None or state.state in ("unavailable", "unknown"):
            return False  # fail-safe: treat as home
        if state.state == "home":
            return False
    return True
