"""Command line interface: ``tomo_denoising <step>``."""

import argparse
import os
import sys

from . import __version__
from .config import DEFAULT_CONFIG_NAME, ConfigError, load_config, write_template
from .utils import CommandError, info

DESCRIPTION = """\
Cryo-ET processing and denoising pipeline, built on WarpTools.

Steps, in the order they are normally run:

  create_config         write a commented tomo_denoising_config.yaml to edit
  motion_correct        WarpTools create_settings + fs_motion_and_ctf
  align_tiltseries      ts_import, ts_stack, tilt axis search, native
                        ts_aretomo3 (axis fixed, no search)
  reconstruct_tomograms ts_defocus_hand, ts_ctf, ts_reconstruct (full
                        tomogram, frame-split and tilt-split half maps)
  run_missalignment     MissAlignment-refined reconstruction (on by default;
                        --skip-missalignment in run_pipe disables it)
  quality_check         RESOLVE resolution estimate, training set, results figure
  train_isonet          prepare_star, denoise, mask, refine
  predict_isonet         apply the refined IsoNet network

  run_pipe               all of the above in one go

Every step writes a report to <processing_directory>/logs/ and, unless
overwrite is true, resumes rather than repeats finished work.
"""


def _add_common(parser, with_force=True):
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_NAME, metavar="FILE",
        help="configuration file (default: ./%s)" % DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "-n", "--dry-run", action="store_true",
        help="print every command that would be run, without running it")
    if with_force:
        parser.add_argument(
            "-f", "--force", action="store_true", default=None,
            help="redo work that is already finished, overriding "
                 "'overwrite' in the configuration")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tomo_denoising",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version="tomo_denoising %s" % __version__)
    sub = parser.add_subparsers(dest="command", metavar="<step>")

    create = sub.add_parser("create_config",
                            help="write a commented configuration file to edit")
    create.add_argument("-o", "--output", default=DEFAULT_CONFIG_NAME,
                        metavar="FILE",
                        help="where to write it (default: ./%s)"
                             % DEFAULT_CONFIG_NAME)
    create.add_argument("-f", "--force", action="store_true",
                        help="overwrite an existing file")

    _add_common(sub.add_parser(
        "motion_correct",
        help="WarpTools create_settings + fs_motion_and_ctf"))

    _add_common(sub.add_parser(
        "align_tiltseries",
        help="ts_import, ts_stack, tilt axis search, native ts_aretomo3"))

    _add_common(sub.add_parser(
        "reconstruct_tomograms",
        help="ts_defocus_hand, ts_ctf, ts_reconstruct -> half maps"))

    _add_common(sub.add_parser(
        "run_missalignment",
        help="MissAlignment-refined reconstruction (WarpTools + miss-alignment)"))

    _add_common(sub.add_parser(
        "quality_check",
        help="RESOLVE resolution estimate, training set and results figure"))

    _add_common(sub.add_parser(
        "train_isonet",
        help="prepare_star, denoise, make_mask and refine"))

    _add_common(sub.add_parser(
        "predict_isonet",
        help="denoise with the refined IsoNet network"))

    pipe = sub.add_parser("run_pipe",
                          help="run the whole pipeline from the configuration")
    _add_common(pipe)
    pipe.add_argument("--skip-quality", action="store_true",
                      help="skip the RESOLVE quality check; requires the "
                           "training mdoc directory to be populated already")
    pipe.add_argument("--skip-missalignment", action="store_true",
                      help="skip MissAlignment-refined reconstruction; it "
                           "runs by default")
    pipe.add_argument("--produce-intermediates", action="store_true",
                      help="also reconstruct the pre-MissAlignment "
                           "frame-split/tilt-split half maps in "
                           "reconstruct_tomograms, even when MissAlignment "
                           "will run right after and reconstruct its own "
                           "(normally skipped there)")

    return parser


def _create_config(args):
    try:
        path = write_template(args.output, force=args.force)
    except ConfigError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    print("Wrote %s" % path)
    print("")
    print("Edit it - at the very least paths.input_frames,")
    print("paths.processing_directory and the mdocs directories - then run:")
    print("")
    print("    tomo_denoising run_pipe")
    print("")
    print("or step by step, starting with:")
    print("")
    print("    tomo_denoising motion_correct")
    print("")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "create_config":
        return _create_config(args)

    config = load_config(args.config)
    info("configuration: %s" % config.source)
    info("processing directory: %s" % config.processing_dir)
    os.makedirs(config.processing_dir, exist_ok=True)

    force = getattr(args, "force", None)

    try:
        if args.command == "motion_correct":
            from . import warptools
            return warptools.motion_correct(config, force=force,
                                            dry_run=args.dry_run)

        if args.command == "align_tiltseries":
            from . import warptools
            return warptools.align_tiltseries(config, force=force,
                                              dry_run=args.dry_run)

        if args.command == "reconstruct_tomograms":
            from . import warptools
            return warptools.reconstruct_tomograms(config, force=force,
                                                    dry_run=args.dry_run)

        if args.command == "run_missalignment":
            from . import missalign
            return missalign.run(config, force=force, dry_run=args.dry_run)

        if args.command == "quality_check":
            from . import resolve
            return resolve.quality_check(config, force=force,
                                         dry_run=args.dry_run)

        if args.command == "train_isonet":
            from . import isonet
            return isonet.train(config, force=force, dry_run=args.dry_run)

        if args.command == "predict_isonet":
            from . import isonet
            return isonet.predict(config, force=force, dry_run=args.dry_run)

        if args.command == "run_pipe":
            from . import pipeline
            return pipeline.run_pipe(config, skip_quality=args.skip_quality,
                                     skip_missalignment=args.skip_missalignment,
                                     produce_intermediates=args.produce_intermediates,
                                     force=force, dry_run=args.dry_run)
    except CommandError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
