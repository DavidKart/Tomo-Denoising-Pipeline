"""tomo_denoising - cryo-ET processing and denoising pipeline.

Chains WarpTools (motion correction, CTF, alignment, reconstruction),
AreTomo, optional MissAlignment refinement and IsoNet into a reproducible,
configuration-driven workflow.
"""

import os as _os
import re as _re


def _read_version():
    """pyproject.toml is the ONE place the version is set.

    A source checkout reads it straight from there - so bumping it takes
    effect immediately, even for an editable install whose package metadata
    is only refreshed on reinstall. An installed copy has no pyproject.toml
    next to it and reads its metadata instead.
    """
    pyproject = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              _os.pardir, _os.pardir, "pyproject.toml")
    try:
        with open(pyproject, encoding="utf-8") as handle:
            match = _re.search(r'^version\s*=\s*"([^"]+)"', handle.read(), _re.M)
        if match:
            return match.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version
        return version("tomo-denoising-pipeline")
    except Exception:  # not installed, or py<3.8
        return "0+unknown"


__version__ = _read_version()

__all__ = ["__version__"]
