# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from itertools import chain

import numpy as np

from edgfs2D.utils.util import fuzzysort


class BaseReader(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _to_raw(self):
        pass

    def get_mesh(self):
        mesh = self._to_raw()

        # Add metadata
        mesh["mesh_uuid"] = np.array(str(mesh["mesh_uuid"]), dtype="S")

        return mesh


class NodalMeshAssembler(object):
    # Dimensionality of each element type
    _petype_ndim = {
        "line": 1,
        "tri": 2,
    }

    # Face numberings for each element type
    _petype_fnums = {
        "line": {"point": [0, 1]},
        "tri": {"line": [0, 1, 2]},
    }

    # Number of nodes in the first-order representation an element
    _petype_focount = {"point": 1, "line": 2, "tri": 3}

    def __init__(self, nodepts, elenodes, pents, maps):
        self._nodepts = nodepts
        self._elenodes = elenodes
        self._felespent, self._bfacespents, self._pfacespents = pents
        self._etype_map, self._petype_fnmap, self._nodemaps = maps

    def _to_first_order(self, elemap):
        foelemap = {}
        for (etype, epent), eles in elemap.items():
            petype = self._etype_map[etype][0]

            # Number of nodes in the first-order representation
            focount = self._petype_focount[petype]

            foelemap[petype, epent] = eles[:, :focount]

        return foelemap

    def _split_fluid(self, elemap):
        selemap = defaultdict(dict)

        for (petype, epent), eles in elemap.items():
            selemap[epent][petype] = eles

        return selemap.pop(self._felespent), selemap

    def _foface_info(self, petype, pftype, foeles):
        # Face numbers of faces of this type on this element
        fnums = self._petype_fnums[petype][pftype]

        # First-order nodes associated with this face
        fnmap = self._petype_fnmap[petype][pftype]

        # Connectivity; (petype, eidx, fidx, flags)
        con = [(petype, i, j, 0) for i in range(len(foeles)) for j in fnums]

        # Nodes
        nodes = np.sort(foeles[:, fnmap]).reshape(len(con), -1)

        return con, nodes

    def _extract_faces(self, foeles):
        fofaces = defaultdict(list)
        for petype, eles in foeles.items():
            for pftype in self._petype_fnums[petype]:
                fofinf = self._foface_info(petype, pftype, eles)
                fofaces[pftype].append(fofinf)

        return fofaces

    def _pair_fluid_faces(self, ffofaces):
        pairs = defaultdict(list)
        resid = {}

        for pftype, faces in ffofaces.items():
            for f, n in chain.from_iterable(zip(f, n) for f, n in faces):
                sn = tuple(n)

                # See if the nodes are in resid
                if sn in resid:
                    pairs[pftype].append([resid.pop(sn), f])
                # Otherwise add them to the unpaired dict
                else:
                    resid[sn] = f

        return pairs, resid

    def _pair_periodic_fluid_faces(self, bpart, resid):
        pfaces = defaultdict(list)
        nodepts = self._nodepts

        for lpent, rpent in self._pfacespents.values():
            for pftype in bpart[lpent]:
                lfnodes = bpart[lpent][pftype]
                rfnodes = bpart[rpent][pftype]

                lfpts = np.array([[nodepts[n] for n in fn] for fn in lfnodes])
                rfpts = np.array([[nodepts[n] for n in fn] for fn in rfnodes])

                lfidx = fuzzysort(lfpts.mean(axis=1).T, range(len(lfnodes)))
                rfidx = fuzzysort(rfpts.mean(axis=1).T, range(len(rfnodes)))

                for lfn, rfn in zip(lfnodes[lfidx], rfnodes[rfidx]):
                    lf = resid.pop(tuple(sorted(lfn)))
                    rf = resid.pop(tuple(sorted(rfn)))

                    pfaces[pftype].append([lf, rf])

        return pfaces

    def _ident_boundary_faces(self, bpart, resid):
        bfaces = defaultdict(list)

        bpents = set(self._bfacespents.values())

        for epent, fnodes in bpart.items():
            if epent in bpents:
                for fn in chain.from_iterable(fnodes.values()):
                    bfaces[epent].append(resid.pop(tuple(sorted(fn))))

        return bfaces

    def get_connectivity(self):
        # For connectivity a first-order representation is sufficient
        eles = self._to_first_order(self._elenodes)

        # Split into fluid and boundary parts
        fpart, bpart = self._split_fluid(eles)

        # Extract the faces of the fluid elements
        ffaces = self._extract_faces(fpart)

        # Pair the fluid-fluid faces
        fpairs, resid = self._pair_fluid_faces(ffaces)

        # Identify periodic boundary face pairs
        pfpairs = self._pair_periodic_fluid_faces(bpart, resid)

        # Identify the fixed boundary faces
        bf = self._ident_boundary_faces(bpart, resid)

        if any(resid.values()):
            raise ValueError("Unpaired faces in mesh")

        # Flattern the face-pair dicts
        pairs = chain(
            chain.from_iterable(fpairs.values()),
            chain.from_iterable(pfpairs.values()),
        )

        # Generate the internal connectivity array
        con = list(pairs)

        # Generate boundary condition connectivity arrays
        bcon = {}
        for pbcrgn, pent in self._bfacespents.items():
            bcon[pbcrgn] = bf[pent]

        # Output
        ret = {"con_p0": np.array(con, dtype="S4,i4,i1,i1").T}

        for k, v in bcon.items():
            ret["bcon_{0}_p0".format(k)] = np.array(v, dtype="S4,i4,i1,i1")

        return ret

    def get_shape_points(self):
        spts = {}

        # Global node map (node index to coords)
        nodepts = self._nodepts

        for etype, pent in self._elenodes:
            if pent != self._felespent:
                continue

            # Elements and type information
            eles = self._elenodes[etype, pent]
            petype, nnodes = self._etype_map[etype]

            # Go from Gmsh to internal node ordering
            peles = eles[:, self._nodemaps.from_internal[petype, nnodes]]

            # Obtain the dimensionality of the element type
            ndim = self._petype_ndim[petype]

            # Build the array
            arr = np.array([[nodepts[i] for i in nn] for nn in peles])
            arr = arr.swapaxes(0, 1)
            arr = arr[..., :ndim]

            spts["spt_{0}_p0".format(petype)] = arr

        return spts
