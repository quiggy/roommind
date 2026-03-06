from .utils.notification_utils import *  # noqa: F401,F403
from .utils import notification_utils as _real  # noqa: E402

# Re-export persistent_notification names so that existing
# ``patch("custom_components.roommind.notification_utils.async_create")``
# targets in tests keep working.  The wrapper functions below look up
# ``async_create`` / ``async_dismiss`` through *this* module's globals so
# that unittest.mock.patch can intercept them.
from homeassistant.components.persistent_notification import (  # noqa: F401,E402
    async_create,
    async_dismiss,
)

import sys as _sys  # noqa: E402


async def async_send_mold_notification(*args, **kwargs):  # noqa: D103
    _mod = _sys.modules[__name__]
    # Temporarily inject this module's (potentially patched) references
    _orig_create = _real.async_create
    _real.async_create = _mod.async_create
    try:
        return await _real.async_send_mold_notification(*args, **kwargs)
    finally:
        _real.async_create = _orig_create


def dismiss_mold_notification(*args, **kwargs):  # noqa: D103
    _mod = _sys.modules[__name__]
    _orig_dismiss = _real.async_dismiss
    _real.async_dismiss = _mod.async_dismiss
    try:
        return _real.dismiss_mold_notification(*args, **kwargs)
    finally:
        _real.async_dismiss = _orig_dismiss
