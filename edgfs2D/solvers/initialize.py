# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
from pathlib import Path

import torch

from edgfs2D.utils.dictionary import Dictionary
from edgfs2D.utils.util import split_vargs


def initialize():
    # do not track gradients globally
    torch.set_grad_enabled(False)

    ap = ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", help="sub-command help")

    # Run command
    ap_run = sp.add_parser("run", help="run --help")
    ap_run.add_argument("inp", type=FileType("r"), help="input file")
    ap_run.add_argument("mesh", type=FileType("r"), help="input mesh file")
    ap_run.add_argument(
        "-v",
        nargs=2,
        action="append",
        default=[],
        help="substitute variables. Example: -v basis-tri::degree 2",
    )
    ap_run.set_defaults(process_run=True)

    # Restart command
    ap_restart = sp.add_parser("restart", help="restart --help")
    ap_restart.add_argument("inp", type=FileType("r"), help="input file")
    ap_restart.add_argument("mesh", type=FileType("r"), help="input mesh file")
    ap_restart.add_argument("soln", type=Path, help="input soln file")
    ap_restart.add_argument(
        "-v",
        nargs=2,
        action="append",
        default=[],
        help="substitute variables. Example: -v basis-tri::degree 2",
    )
    ap_restart.set_defaults(process_restart=True)

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, "process_run") or hasattr(args, "process_restart"):
        inp = Dictionary.load(args.inp, defaults=split_vargs(args.v))
        return inp, args
    else:
        ap.print_help()
        exit(0)
