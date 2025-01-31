# -*- coding: utf-8 -*-

"""
Post process solution
"""
from argparse import REMAINDER, ArgumentParser, FileType
from pathlib import Path

import torch

from edgfs2D.post_process import get_post_processor


def main():
    # do not track gradients globally
    torch.set_grad_enabled(False)

    ap = ArgumentParser()

    # Post process command
    ap.add_argument(
        "p",
        type=str,
        help="post processor name. Example: vtu",
    )

    # Add a special argument to capture everything after '--'
    ap.add_argument(
        "plugin_args",
        nargs=REMAINDER,
        help="Arguments to be forwarded to the plugin",
    )

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the post process method
    if hasattr(args, "p"):
        pp = get_post_processor(args.p, args.plugin_args)
        pp.execute()
    else:
        ap.print_help()
        exit(0)


def __main__():
    main()
