"""Command line interface: ``tomo_denoising <step>``."""

import argparse
import os
import sys

from . import __version__
from .config import DEFAULT_CONFIG_NAME, ConfigError, load_config, write_template
from .utils import CommandError, info

DESCRIPTION = """\
Cryo-ET processing and denoising pipeline.

Steps, in the order they are normally run:

  create_config      write a commented tomo_denoising_config.yaml to edit
  create_halfMaps    raw frames -> tomograms + even/odd half maps
  train_cryoCARE     train a cryoCARE denoiser on the training tomograms
  predict_cryoCARE   apply the cryoCARE denoiser to the prediction tomograms
  train_isonet       CTFFIND4 defocus, prepare_star, denoise, mask, refine
  predict_isonet     apply the refined IsoNet network
"""


def _add_config_argument(parser):
    parser.add_argument(
        "-c", "--config",
        default=DEFAULT_CONFIG_NAME,
        metavar="FILE",
        help="configuration file (default: ./%s)" % DEFAULT_CONFIG_NAME,
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="print every command that would be run, without running it",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="tomo_denoising",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version",
                        version="tomo_denoising %s" % __version__)
    sub = parser.add_subparsers(dest="command", metavar="<step>")

    create = sub.add_parser(
        "create_config",
        help="write a commented configuration file to edit",
    )
    create.add_argument("-o", "--output", default=DEFAULT_CONFIG_NAME,
                        metavar="FILE",
                        help="where to write it (default: ./%s)"
                             % DEFAULT_CONFIG_NAME)
    create.add_argument("-f", "--force", action="store_true",
                        help="overwrite an existing file")

    halfmaps = sub.add_parser(
        "create_halfMaps",
        help="MotionCor3 + newstack + AreTomo2 -> tomograms and half maps",
    )
    _add_config_argument(halfmaps)
    halfmaps.add_argument("-f", "--force", action="store_true",
                          help="reprocess tilt series whose half maps exist")

    train_cc = sub.add_parser("train_cryoCARE", help="train cryoCARE")
    _add_config_argument(train_cc)

    predict_cc = sub.add_parser("predict_cryoCARE",
                                help="denoise with the trained cryoCARE model")
    _add_config_argument(predict_cc)

    train_in = sub.add_parser(
        "train_isonet",
        help="CTFFIND4 defocus, prepare_star, denoise, make_mask and refine",
    )
    _add_config_argument(train_in)
    train_in.add_argument("--force-ctf", action="store_true",
                          help="re-run CTFFIND4 even if a _PS.txt exists")

    predict_in = sub.add_parser("predict_isonet",
                                help="denoise with the refined IsoNet network")
    _add_config_argument(predict_in)
    predict_in.add_argument("--force-ctf", action="store_true",
                            help="re-run CTFFIND4 even if a _PS.txt exists")

    return parser


def _create_config(args):
    try:
        path = write_template(args.output, force=args.force)
    except ConfigError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    print("Wrote %s" % path)
    print("")
    print("Edit it - at the very least paths.input_frames and")
    print("paths.processing_directory - then run:")
    print("")
    print("    tomo_denoising create_halfMaps")
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

    try:
        if args.command == "create_halfMaps":
            from . import halfmaps
            return halfmaps.create_half_maps(
                config, force=args.force, dry_run=args.dry_run)

        if args.command == "train_cryoCARE":
            from . import cryocare
            return cryocare.train(config, dry_run=args.dry_run)

        if args.command == "predict_cryoCARE":
            from . import cryocare
            return cryocare.predict(config, dry_run=args.dry_run)

        if args.command == "train_isonet":
            from . import isonet
            return isonet.train(config, force_ctf=args.force_ctf,
                                dry_run=args.dry_run)

        if args.command == "predict_isonet":
            from . import isonet
            return isonet.predict(config, force_ctf=args.force_ctf,
                                  dry_run=args.dry_run)
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
