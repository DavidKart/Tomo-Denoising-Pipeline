"""Defocus estimation with CTFFIND4, used to fill in the IsoNet star file.

CTFFIND4 is run on the motion-corrected average of the FIRST COLLECTED tilt of
each series - that is the ``*_averaged.mrc`` file, never one of the ``_EVN`` /
``_ODD`` half images, so the estimate uses the full dose of that tilt.

Results are written to ``<processing_directory>/isonet_config/<tomo>_PS.txt``
and reused on later runs, so re-running a training step is cheap.
"""

import os

from .mdoc import parse_mdoc, resolve_pixel_size
from .utils import ensure_dir, info, require_executable, run_with_stdin, warn


def ps_txt_path(config, tomo_name):
    return os.path.join(config.isonet_config_dir, "%s_PS.txt" % tomo_name)


def first_tilt_average(config, series):
    """Path to the full motion-corrected average of the first collected tilt.

    Falls back to the next acquired tilt if the first one is missing (e.g.
    MotionCor3 failed on it). Half images (_EVN/_ODD) are never considered.
    """
    motioncor_dir = os.path.join(config.rec_dir(series.name), "motioncor")
    gain_name = config.gain_name

    for frame in series.frames:
        if gain_name and os.path.basename(frame) == gain_name:
            continue
        stem = os.path.splitext(frame)[0]
        candidate = os.path.join(motioncor_dir, stem + "_averaged.mrc")
        if os.path.exists(candidate):
            return candidate
    return None


def parse_ctffind_output(path):
    """Return the mean of defocus 1 and defocus 2, in Angstrom.

    Column layout of the CTFFIND4 text output::

        #1 micrograph number  #2 defocus 1 [A]  #3 defocus 2 [A]  ...
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                return (float(parts[1]) + float(parts[2])) / 2.0
            except ValueError:
                continue
    raise ValueError("no defocus values found in %s" % path)


def run_ctffind(config, series, force=False, dry_run=False):
    """Estimate the defocus for one tilt series; returns Angstrom or None."""
    output_txt = ps_txt_path(config, series.name)

    if os.path.exists(output_txt) and not force:
        try:
            defocus = parse_ctffind_output(output_txt)
            info("%s: cached defocus %.1f A (%s)"
                 % (series.name, defocus, os.path.basename(output_txt)))
            return defocus
        except (ValueError, OSError):
            warn("could not read %s - re-running CTFFIND4." % output_txt)

    image = first_tilt_average(config, series)
    if image is None:
        warn(
            "%s: no motion-corrected first tilt found. Run "
            "'tomo_denoising create_halfMaps' before IsoNet." % series.name
        )
        return None

    config_dir = ensure_dir(config.isonet_config_dir)
    diagnostic = os.path.join(config_dir, "%s_PS.mrc" % series.name)

    apix = resolve_pixel_size(
        config, series, os.path.join(config.input_frames, series.frames[0])
        if series.frames else None
    )
    apix_binned = apix * int(config["motioncor"]["binning"])

    acquisition = config["acquisition"]
    settings = config["ctffind"]

    info("%s: CTFFIND4 on %s (pixel size %.4f A)"
         % (series.name, os.path.basename(image), apix_binned))

    # CTFFIND4 (4.1.x) reads its parameters from stdin, in this order.
    answers = [
        image,
        diagnostic,
        "%.4f" % apix_binned,
        "%.1f" % float(acquisition["voltage"]),
        "%.2f" % float(acquisition["spherical_aberration"]),
        "%.3f" % float(acquisition["amplitude_contrast"]),
        str(int(settings["box_size"])),
        "%.1f" % float(settings["min_resolution"]),
        "%.1f" % float(settings["max_resolution"]),
        "%.1f" % float(settings["min_defocus"]),
        "%.1f" % float(settings["max_defocus"]),
        "%.1f" % float(settings["defocus_step"]),
        "no",   # Do you know what astigmatism is present?
        "no",   # Slower, more exhaustive search?
        "no",   # Use a restraint on astigmatism?
        "no",   # Find additional phase shift?
        "no",   # Do you want to set expert options?
    ]

    run_with_stdin(
        [config.exe("ctffind")],
        answers,
        cwd=config_dir,
        log_file=os.path.join(config_dir, "%s_ctffind.log" % series.name),
        check=False,
        dry_run=dry_run,
    )

    if dry_run:
        return None
    if not os.path.exists(output_txt):
        warn("%s: CTFFIND4 produced no %s" % (series.name, output_txt))
        return None
    try:
        defocus = parse_ctffind_output(output_txt)
    except (ValueError, OSError) as exc:
        warn("%s: %s" % (series.name, exc))
        return None
    info("%s: defocus %.1f A" % (series.name, defocus))
    return defocus


def defocus_for_series(config, mdoc_paths, force=False, dry_run=False):
    """Estimate the defocus for several tilt series.

    Returns ``{tomo_name: defocus_in_angstrom}``, skipping series CTFFIND4
    could not handle.
    """
    require_executable(config.exe("ctffind"), "CTFFIND4 defocus estimate")
    ensure_dir(config.isonet_config_dir)

    results = {}
    for mdoc_path in mdoc_paths:
        series = parse_mdoc(mdoc_path)
        defocus = run_ctffind(config, series, force=force, dry_run=dry_run)
        if defocus is not None:
            results[series.name] = defocus
    return results
