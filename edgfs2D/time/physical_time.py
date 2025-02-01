from argparse import Namespace
from pathlib import Path

import numpy as np

from edgfs2D.fields.readers.h5 import H5FieldReader
from edgfs2D.plugins import get_plugin_for_solver, get_plugin_sections
from edgfs2D.solvers.base import BaseSolver
from edgfs2D.utils.dictionary import Dictionary

tsect = "time"
ttype = np.float64


class PhysicalTime(object):
    """Defines physical time assoicated with simulations"""

    def __init__(self, global_cfg: Dictionary, args: Namespace):
        self._global_cfg = global_cfg
        self._args = args
        self._cfg = self._global_cfg.get_section(tsect)

        # start time of the simulation
        self._tstart = ttype(self._define_start_time())

        # end time of the simulation
        self._tend = ttype(self._cfg.lookupfloat("tend"))

        # time-step of the simulation
        self._dt = ttype(self._cfg.lookupfloat("dt"))
        if self._dt <= 0:
            raise ValueError("timestep should be positive")

        if self._tend <= self._tstart:
            self._nsteps = 0
        else:
            self._nsteps = np.ceil((self._tend - self._tstart) / self._dt)
            self._dt = (self._tend - self._tstart) / self._nsteps

        self._time = self._tstart
        self._step = 0

        # plugins
        self._plugins = []

    @property
    def is_restart(self):
        return hasattr(self._args, "process_restart")

    def run(self):
        return self._step < self._nsteps

    def increment(self):
        self._step += 1
        self._time += self._dt
        self.call_plugins()

    @property
    def args(self):
        return self._args

    @property
    def global_cfg(self):
        return self._global_cfg

    @property
    def time(self):
        return self._time

    @property
    def dt(self):
        return self._dt

    @property
    def step(self):
        return self._step

    def _define_start_time(self):
        if self.is_restart:
            path = Path(self.args.soln)
            return H5FieldReader(path).read_metadata("time")
        else:
            return self._cfg.lookupfloat("tstart")

    def load_plugins(self, instance: BaseSolver):
        for m in get_plugin_sections(self._global_cfg):
            name = m.group(1)
            self._plugins.append(
                get_plugin_for_solver(
                    self._global_cfg, type(instance).__name__, name, instance
                )
            )

    def call_plugins(self):
        for plugin in self._plugins:
            plugin.__call__()

    def should_output(self, step):
        return step > 0 and (self._step % step == 0)
