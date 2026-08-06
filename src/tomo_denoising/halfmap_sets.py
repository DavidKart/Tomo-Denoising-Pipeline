"""Selecting the even/odd half maps belonging to a set of mdoc files."""

import os

from .mdoc import collect_tilt_series
from .utils import info, warn


def half_map_paths(config, tomo_name):
    """The canonical half-map locations for one tomogram."""
    even = os.path.join(config.raw_halfmaps_dir, "even_%s_rec_tomo.mrc" % tomo_name)
    odd = os.path.join(config.raw_halfmaps_dir, "odd_%s_rec_tomo.mrc" % tomo_name)
    return even, odd


def collect_half_maps(config, which):
    """Return ``[(tomo_name, even_path, odd_path), ...]`` for an mdoc set.

    *which* is ``"train"`` or ``"predict"``; the mdoc directory comes from the
    ``mdocs`` section of the configuration.
    """
    directory = config.mdoc_dir(which)
    series = collect_tilt_series(directory, which)

    pairs = []
    missing = []
    for entry in series:
        even, odd = half_map_paths(config, entry.name)
        if os.path.exists(even) and os.path.exists(odd):
            pairs.append((entry.name, even, odd))
        else:
            missing.append(entry.name)

    if missing:
        warn(
            "no half maps for: %s\n"
            "  Run 'tomo_denoising create_halfMaps' first, or check whether "
            "AreTomo2 failed for these tilt series." % ", ".join(missing)
        )
    if not pairs:
        raise SystemExit(
            "ERROR: none of the tomograms listed in %s have half maps in %s."
            % (directory, config.raw_halfmaps_dir)
        )

    info("%d tomograms selected for %s (from %s)"
         % (len(pairs), which, directory))
    return pairs
