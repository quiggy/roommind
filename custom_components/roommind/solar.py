from .control import solar as _mod  # noqa: E402
from .control.solar import *  # noqa: F401,F403

# Re-export private names so that existing callers (incl. tests) keep working.
_clear_sky_ghi = _mod._clear_sky_ghi  # noqa: F811
_cloud_attenuation = _mod._cloud_attenuation  # noqa: F811
_solar_elevation = _mod._solar_elevation  # noqa: F811
