# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser, FileType

from edgfs2D.utils.dictionary import Dictionary


def initialize():
    ap = ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', help='sub-command help')

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('inp', type=FileType('r'), help='input file')
    ap_run.add_argument('mesh', type=FileType('r'), help='input mesh file')
    ap_run.add_argument('-v', nargs=2, action='append', default=[],
        help='substitute variables. Example: -v mesh::Nv 32')
    ap_run.set_defaults(process_run=True)

    # Restart command
    ap_restart = sp.add_parser('restart', help='restart --help')
    ap_restart.add_argument('inp', type=FileType('r'), help='input file')
    ap_restart.add_argument('mesh', type=FileType('r'), help='input mesh file')
    ap_restart.add_argument('dist', nargs='+', action='append',
        type=FileType('r'), default=[], help='sorted list of distributions')
    ap_restart.add_argument('-v', nargs=2, action='append', default=[],
        help='substitute variables. Example: -v mesh::Nv 32')
    ap_restart.set_defaults(process_restart=True)

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process_run') or hasattr(args, 'process_restart'):
        vars = {}
        for key, value in args.v:
            keyscope = key.split('::')
            child = None
            parent = vars
            while keyscope:
                child = keyscope.pop(0)
                if not parent or child not in parent:
                    parent[child] = dict()
                if(keyscope):
                    parent = parent[child]
                else:
                    parent[child] = value
        inp = Dictionary.load(args.inp, defaults=vars)
        return inp, args
    else:
        ap.print_help()