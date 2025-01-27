# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType
from pathlib import Path

from edgfs2D.utils.dictionary import Dictionary
from edgfs2D.utils.util import split_vargs


def initialize():
    ap = ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", help="sub-command help")

    # Post process command
    ap_pp = sp.add_parser("pp", help="restart --help")
    ap_pp.add_argument("inp", type=FileType("r"), help="input file")
    ap_pp.add_argument("mesh", type=FileType("r"), help="input mesh file")
    ap_pp.add_argument("soln", type=Path, help="input solution file")
    ap_pp.add_argument(
        "-v",
        nargs=2,
        action="append",
        default=[],
        help="substitute variables. Example: -v basis-tri::degree 2",
    )
    ap_pp.add_argument(
        "-p",
        nargs=1,
        type=str,
        default="vtu",
        help="post processor name. Example: -v vtu",
    )
    ap_pp.set_defaults(process_pp=True)

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the post process method
    if hasattr(args, "process_pp"):
        inp = Dictionary.load(args.inp, defaults=split_vargs(args.v))
        return inp, args
    else:
        ap.print_help()
        exit(0)
