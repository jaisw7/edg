import io
import json
import os
import re
from configparser import NoOptionError, NoSectionError, SafeConfigParser

import numpy as np

from edgfs2D.utils.util import np_map, torch_cmap, torch_map

cfgsect = "config"


def _ensure_float(m):
    m = m.group(0)
    return m if any(c in m for c in ".eE") else m + "."


class Dictionary(object):
    def __init__(self, inistr=None, defaults={}):
        self._cp = cp = SafeConfigParser(inline_comment_prefixes=[";"])
        cp.optionxform = str

        if inistr:
            cp.read_string(inistr)

        if defaults:
            cp.read_dict(defaults)

        self._dtypename = self.lookupordefault(cfgsect, "precision", "double")
        self._device = self.lookupordefault(cfgsect, "device", "cpu")
        self._dtype = np_map[self._dtypename]
        self._ttype = torch_map[self._dtypename]
        self._cttype = torch_cmap[self._dtypename]

    @staticmethod
    def load(file, defaults={}):
        if isinstance(file, str):
            file = open(file)

        return Dictionary(file.read(), defaults=defaults)

    def get_section(self, section):
        items = {}
        if self.has_section(section):
            items.update({section: dict(self._cp.items(section))})
        items.update({cfgsect: dict(self._cp.items(cfgsect))})
        return SubDictionary(section, defaults=items)

    def has_section(self, section):
        return self._cp.has_section(section)

    def has_option(self, section, option):
        return self._cp.has_option(section, option)

    def lookup(self, section, option, vars=None):
        val = self._cp.get(section, option, vars=vars)
        return os.path.expandvars(val)

    def lookupordefault(self, section, option, default, vars=None):
        try:
            T = type(default)
            val = T(self._cp.get(section, option, vars=vars))
        except:
            val = default
        return val

    def lookuppath(self, section, option, default, vars=None, abs=False):
        path = self.lookupordefault(section, option, default, vars)
        path = os.path.expanduser(path)

        if abs:
            path = os.path.abspath(path)

        return path

    def lookupexpr(self, section, option, subs={}):
        expr = self.lookup(section, option)

        # Ensure the expression does not contain invalid characters
        if not re.match(r"[A-Za-z0-9 \t\n\r.,+\-*/%()<>=\{\}\[\]\$]+$", expr):
            raise ValueError("Invalid characters in expression")

        # Substitute variables
        if subs:
            expr = re.sub(
                r"\b({0})\b".format("|".join(subs)),
                lambda m: subs[m.group(1)],
                expr,
            )

        # Convert integers to floats
        expr = re.sub(
            r"\b((\d+\.?\d*)|(\.\d+))([eE][+-]?\d+)?(?!\s*])",
            _ensure_float,
            expr,
        )

        # Encase in parenthesis
        return "({0})".format(expr)

    def lookupfloat(self, section, option):
        return self._dtype(self.lookup(section, option))

    def lookupfloats(self, section, options):
        return map(lambda op: self.lookupfloat(section, op), options)

    def lookupint(self, section, option):
        return int(self.lookup(section, option))

    def lookupints(self, section, options):
        return map(lambda op: self.lookupint(section, op), options)

    def lookup_list(self, section, option, dtype):
        return np.array(
            list(map(dtype, json.loads(self.lookup(section, option))))
        )

    def lookupfloat_list(self, section, option):
        return self.lookup_list(section, option, self._dtype)

    def lookupint_list(self, section, option):
        return self.lookup_list(section, option, int)

    def __str__(self):
        buf = io.StringIO()
        self._cp.write(buf)
        return buf.getvalue()

    def sections(self):
        return self._cp.sections()

    def section_values(self, section, type):
        iv = []
        for k, v in self._cp.items(section):
            try:
                try:
                    v.index("[")
                    iv.append((k, self.lookup_list(section, k, type)))
                except ValueError:
                    iv.append((k, type(v)))
            except ValueError:
                pass
        return dict(iv)

    # Global configurations that are required in all systems
    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def ttype(self):
        return self._ttype

    @property
    def cttype(self):
        return self._cttype


class SubDictionary:
    def __init__(self, section, defaults={}):
        self.dict = Dictionary(defaults=defaults)
        self._section = section

    def has_section(self, *args, **kwargs):
        return self.dict.has_section(*args, **kwargs)

    def has_option(self, *args, **kwargs):
        return self.dict.has_option(self._section, *args, **kwargs)

    def lookup(self, *args, **kwargs):
        return self.dict.lookup(self._section, *args, **kwargs)

    def lookupordefault(self, *args, **kwargs):
        return self.dict.lookupordefault(self._section, *args, **kwargs)

    def lookuppath(self, *args, **kwargs):
        return self.dict.lookuppath(self._section, *args, **kwargs)

    def lookupexpr(self, *args, **kwargs):
        return self.dict.lookupexpr(self._section, *args, **kwargs)

    def lookupfloat(self, *args):
        return self.dict.lookupfloat(self._section, *args)

    def lookupfloats(self, *args):
        return self.dict.lookupfloats(self._section, *args)

    def lookupint(self, *args):
        return self.dict.lookupint(self._section, *args)

    def lookupints(self, *args):
        return self.dict.lookupints(self._section, *args)

    def lookup_list(self, *args):
        return self.dict.lookup_list(self._section, *args)

    def lookupfloat_list(self, *args):
        return self.dict.lookupfloat_list(self._section, *args)

    def lookupint_list(self, *args):
        return self.dict.lookup_list(self._section, *args)

    @property
    def dtype(self):
        return self.dict._dtype

    @property
    def ttype(self):
        return self.dict._ttype

    @property
    def device(self):
        return self.dict._device

    @property
    def cttype(self):
        return self.dict._cttype
