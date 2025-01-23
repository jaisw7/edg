# -*- coding: utf-8 -*-

import os

import h5py as h5py
import numpy as np
from dgfs1D.nputil import get_comm_rank_root

"""Write distribution for single-species systems"""
class DGFSDistributionStd():
    
    def __init__(self, tcurr, shape, cfg, cfgsect, extn='.h5'):

        self.K, self.Ne, self.Nv = shape

        # Construct the solution writer
        self.basedir = cfg.lookuppath(cfgsect, 'basedir', '.', abs=True)
        self.basename = cfg.lookup(cfgsect, 'basename')
        if not self.basename.endswith(extn):
            self.basename += extn

        _, rank, _ = get_comm_rank_root()
        self.basename += "_rank=" + str(rank)

        # Output time step and next output time
        self.dt_out = cfg.lookupfloat(cfgsect, 'dt-out')
        self.tout_next = tcurr + self.dt_out

    def __call__(self, dt, tcurr, coeff):
        if abs(self.tout_next - tcurr) > 0.5*dt:
            return

        # Write out the file
        fname = self.basename.format(t=tcurr)
        solnfname = os.path.join(self.basedir, fname)
        
        with h5py.File(solnfname, 'w') as h5f:
            dst = h5f.create_dataset('coeff', data=coeff.get())
            dst.attrs['time'] = tcurr
            dst.attrs['K'] = self.K
            dst.attrs['Ne'] = self.Ne
            dst.attrs['Nv'] = self.Nv

        # Compute the next output time
        self.tout_next = tcurr + self.dt_out