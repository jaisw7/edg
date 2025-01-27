# -*- coding: utf-8 -*-
import itertools as it
import re

import numpy as np

_npeval_syms = {
    "__builtins__": None,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "asin": np.arcsin,
    "cos": np.cos,
    "acos": np.arccos,
    "tan": np.tan,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "abs": np.abs,
    "pow": np.power,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
    "pi": np.pi,
    "linspace": np.linspace,
    "logspace": np.logspace,
}


def npeval(expr, locals):
    # Disallow direct exponentiation
    if "^" in expr or "**" in expr:
        raise ValueError("Direct exponentiation is not supported; use pow")

    # Ensure the expression does not contain invalid characters
    if not re.match(r"[A-Za-z0-9 \t\n\r.,+\-*/%()<>=]+$", expr):
        raise ValueError("Invalid characters in expression")

    # Disallow access to object attributes
    objs = "|".join(it.chain(_npeval_syms, locals))
    if re.search(r"(%s|\))\s*\." % objs, expr):
        raise ValueError("Invalid expression")

    return eval(expr, _npeval_syms, locals)


def ndrange(*args):
    return it.product(*map(range, args))


def subclasses(cls, just_leaf=False):
    sc = cls.__subclasses__()
    ssc = [g for s in sc for g in subclasses(s, just_leaf)]

    return [s for s in sc if not just_leaf or not s.__subclasses__()] + ssc


def subclass_where(cls, **kwargs):
    k, v = next(iter(kwargs.items()))

    for s in subclasses(cls):
        if hasattr(s, k) and getattr(s, k) == v:
            return s

    raise KeyError(
        "No subclasses of {0} with cls.{1} == '{2}'".format(cls.__name__, k, v)
    )
