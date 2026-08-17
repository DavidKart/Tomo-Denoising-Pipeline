"""The multi-page results figure.

One panel group per denoised tomogram, laid out like 3dmod's XYZ window:

      +-----------+
      |    XZ     |        XZ  : slice at the central Y
      +-----------+---+
      |           |   |
      |    XY     |ZY |     XY  : slice at the central Z (the main view)
      |           |   |     ZY  : slice at the central X
      +-----------+---+

Slices are drawn with the origin at the bottom left and Y increasing upwards,
which is how 3dmod displays them. Panels are captioned with the RESOLVE
resolution and signal ratio and sorted best resolution first, eight tomograms
per page (4 columns x 2 rows).

The figure is rebuilt whenever both halves exist - after quality_check when a
denoising result is already there, and after a prediction step when RESOLVE
results are already there.
"""

import glob
import os

import numpy as np

from .mdoc import match_tomo_name
from .utils import ensure_dir, info, warn

#: Where each denoiser leaves its predictions, relative to the processing dir.
DENOISER_OUTPUTS = {
    "cryoCARE": ["denoised_cryoCARE"],
    "isonet": [
        os.path.join("isonet", "corrected_tomos"),
        os.path.join("isonet", "isonet_maps"),
        os.path.join("isonet", "predict"),
    ],
}

TOMOGRAM_EXTENSIONS = (".mrc", ".rec", ".map")

ROWS_PER_PAGE = 2
COLUMNS_PER_PAGE = 4
PER_PAGE = ROWS_PER_PAGE * COLUMNS_PER_PAGE

#: Resolution the images inside the PDF are rasterised at. The vector PDF
#: backend otherwise falls back to matplotlib's default figure dpi (100),
#: which is soft enough to look noticeably blurry once printed or zoomed in.
FIGURE_DPI = 500

#: Gutter between the XY/XZ/ZY panes, as a fraction of the largest of the
#: three slice dimensions. A fixed pixel count (the previous approach) looks
#: fine for a small tomogram and vanishes entirely for a large one, because
#: the same handful of array pixels covers a shrinking fraction of the panel
#: as the array grows - a proportional gutter keeps the visible gap the same
#: size on the page regardless of the tomogram's resolution.
GUTTER_FRACTION = 0.012
MIN_GUTTER = 3


def report_path(config, denoiser):
    return os.path.join(config.processing_dir, "results_%s.pdf" % denoiser)


# ------------------------------------------------------------- finding volumes


def _candidate_files(directory):
    found = []
    for extension in TOMOGRAM_EXTENSIONS:
        found.extend(glob.glob(os.path.join(directory, "*" + extension)))
    return sorted(found)


def find_denoised(config, denoiser, tomo_names):
    """Map tomogram name -> denoised volume, for one denoiser.

    The denoisers wrap the tomogram name in affixes of their own, so the name
    is matched as a whole underscore-delimited run rather than as a plain
    substring - otherwise Position_1 would claim Position_10's file.
    """
    matches = {}
    for relative in DENOISER_OUTPUTS.get(denoiser, []):
        directory = os.path.join(config.processing_dir, relative)
        if not os.path.isdir(directory):
            continue
        for path in _candidate_files(directory):
            name = match_tomo_name(path, tomo_names)
            if name is not None and name not in matches:
                matches[name] = path
    return matches


def available_denoisers(config, tomo_names):
    """Which denoisers have usable output right now."""
    return {name: found
            for name in DENOISER_OUTPUTS
            for found in [find_denoised(config, name, tomo_names)]
            if found}


# -------------------------------------------------------------------- slicing


def central_slices(path):
    """Central XY, XZ and ZY slices of a tomogram.

    MRC data is indexed (z, y, x). The returned arrays are already oriented for
    display with ``origin="lower"``: XY is (y, x), XZ is (z, x), ZY is (y, z).
    """
    import mrcfile

    with mrcfile.mmap(path, mode="r", permissive=True) as volume:
        data = volume.data
        if data is None or data.ndim != 3:
            raise ValueError("%s is not a 3D volume" % path)
        nz, ny, nx = data.shape
        xy = np.asarray(data[nz // 2, :, :], dtype="float32")
        xz = np.asarray(data[:, ny // 2, :], dtype="float32")
        zy = np.asarray(data[:, :, nx // 2], dtype="float32").T
    return xy, xz, zy


def _contrast(image):
    """Percentile clipping, so a few hot pixels do not flatten the contrast."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


# --------------------------------------------------------------------- drawing


def compose_xyz(xy, xz, zy, gutter):
    """Lay the three planes out on one canvas, 3dmod XYZ style.

    Compositing into a single image rather than three subplots is what keeps
    the panes exactly adjacent and at one common scale: with square pixels,
    separate axes each letterbox their image independently and drift apart.

    The canvas is meant to be drawn with ``origin="lower"``, so row 0 is the
    bottom. Gutters are NaN and render as the axes background. *gutter* is in
    array pixels - callers scale it to the tomogram's own resolution (see
    ``GUTTER_FRACTION``) so the gap reads the same size on the page no matter
    how large the tomogram is.
    """
    ny, nx = xy.shape
    nz = xz.shape[0]

    height = ny + gutter + nz
    width = nx + gutter + nz
    canvas = np.full((height, width), np.nan, dtype="float32")

    canvas[0:ny, 0:nx] = xy                                   # main view
    canvas[ny + gutter:ny + gutter + nz, 0:nx] = xz           # above
    canvas[0:ny, nx + gutter:nx + gutter + nz] = zy           # to the right
    return canvas


def _draw_tomogram(figure, cell, path, measurement, name):
    """Draw one tomogram's three central slices into *cell*."""
    axis = figure.add_subplot(cell)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_visible(False)

    try:
        xy, xz, zy = central_slices(path)
    except Exception as exc:                              # noqa: BLE001
        axis.text(0.5, 0.5, "%s\ncould not be read\n%s" % (name, exc),
                  ha="center", va="center", fontsize=6)
        return

    ny, nx = xy.shape
    nz = xz.shape[0]
    gutter = max(MIN_GUTTER, int(round(GUTTER_FRACTION * max(nx, ny, nz))))

    canvas = compose_xyz(xy, xz, zy, gutter=gutter)
    low, high = _contrast(xy)

    colours = plt_colormap()
    axis.imshow(canvas, cmap=colours, vmin=low, vmax=high, origin="lower",
                interpolation="nearest", aspect="equal")

    height, width = canvas.shape
    for label, (x, y) in (("XY", (2, ny - 2)),
                          ("XZ", (2, height - 2)),
                          ("ZY", (nx + gutter + 2, ny - 2))):
        axis.text(x, y, label, fontsize=5, color="yellow",
                  ha="left", va="top")
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(-0.5, height - 0.5)

    # A thin border around each of the three panes individually - drawn last
    # so it sits on top of the image.
    from matplotlib.patches import Rectangle
    for x0, y0, w, h in (
        (0, 0, nx, ny),                              # XY
        (0, ny + gutter, nx, nz),                    # XZ
        (nx + gutter, 0, nz, ny),                    # ZY
    ):
        axis.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False,
                                 edgecolor="black", linewidth=0.6))

    if measurement is not None:
        caption = "%s\nResolution: %.1f Å   Signal ratio: %.2f" % (
            name, measurement.resolution, measurement.signal_ratio)
    else:
        caption = "%s\n(not measured)" % name
    axis.set_title(caption, fontsize=6.5, pad=3)


def plt_colormap():
    """Greyscale with NaN (the gutters) drawn as the page background."""
    import matplotlib

    try:                                   # matplotlib >= 3.5
        colours = matplotlib.colormaps["gray"].copy()
    except (AttributeError, KeyError):     # older releases
        colours = matplotlib.cm.get_cmap("gray").copy()
    colours.set_bad("white")
    return colours


def build_report(config, denoiser, volumes, measurements, log=None):
    """Write the multi-page PDF for one denoiser. Returns its path, or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec

    names = [n for n in volumes if n in measurements]
    unmeasured = [n for n in volumes if n not in measurements]
    if not names and not unmeasured:
        return None

    # Best resolution first; anything RESOLVE did not measure goes last.
    names.sort(key=lambda n: measurements[n].resolution)
    names.extend(sorted(unmeasured))

    path = report_path(config, denoiser)
    ensure_dir(os.path.dirname(path))
    info("building %s (%d tomograms)" % (path, len(names)))

    pages = (len(names) + PER_PAGE - 1) // PER_PAGE
    with PdfPages(path) as pdf:
        for page in range(pages):
            chunk = names[page * PER_PAGE:(page + 1) * PER_PAGE]
            figure = plt.figure(figsize=(11.69, 8.27),      # A4 landscape
                               dpi=FIGURE_DPI)
            figure.suptitle(
                "%s - denoised tomograms, best resolution first  "
                "(page %d of %d)" % (denoiser, page + 1, pages),
                fontsize=10,
            )
            grid = GridSpec(ROWS_PER_PAGE, COLUMNS_PER_PAGE, figure=figure,
                            wspace=0.12, hspace=0.28,
                            left=0.03, right=0.97, top=0.90, bottom=0.03)
            for index, name in enumerate(chunk):
                cell = grid[index // COLUMNS_PER_PAGE,
                            index % COLUMNS_PER_PAGE]
                _draw_tomogram(figure, cell, volumes[name],
                               measurements.get(name), name)
            pdf.savefig(figure, dpi=FIGURE_DPI)
            plt.close(figure)

    if log is not None:
        log.note("results figure: %s (%d tomograms, %d page(s))"
                 % (path, len(names), pages))
    return path


def build_all_reports(config, measurements, log=None):
    """Build a figure for every denoiser that has output. Returns the paths."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        warn("matplotlib is not installed - skipping the results figure. "
             "Install it with:  pip install matplotlib")
        return []

    if not measurements:
        return []

    produced = []
    for denoiser in sorted(DENOISER_OUTPUTS):
        volumes = find_denoised(config, denoiser, list(measurements))
        if not volumes:
            continue
        try:
            path = build_report(config, denoiser, volumes, measurements,
                                log=log)
        except Exception as exc:                          # noqa: BLE001
            warn("could not build the %s results figure: %s" % (denoiser, exc))
            if log is not None:
                log.note("results figure for %s FAILED: %s" % (denoiser, exc))
            continue
        if path:
            produced.append(path)
    return produced


def refresh_after_prediction(config, denoiser, log=None):
    """Rebuild the figure after a prediction, if RESOLVE results already exist."""
    from .resolve import read_summary

    measurements = read_summary(config)
    if not measurements:
        if log is not None:
            log.note("no RESOLVE results yet - run 'tomo_denoising "
                     "quality_check' to get the results figure.")
        return None

    volumes = find_denoised(config, denoiser, list(measurements))
    if not volumes:
        return None
    try:
        return build_report(config, denoiser, volumes, measurements, log=log)
    except Exception as exc:                              # noqa: BLE001
        warn("could not build the %s results figure: %s" % (denoiser, exc))
        return None
