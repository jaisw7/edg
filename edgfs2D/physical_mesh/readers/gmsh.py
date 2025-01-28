# -*- coding: utf-8 -*-

import hashlib
import re
from collections import defaultdict

import numpy as np

from edgfs2D.physical_mesh.readers import BaseReader, NodalMeshAssembler
from edgfs2D.physical_mesh.readers.nodemaps import GmshNodeMaps


def msh_section(mshit, section):
    endln = "$End{}\n".format(section)
    endix = int(next(mshit)) - 1

    for i, l in enumerate(mshit):
        if l == endln:
            raise ValueError("Unexpected end of section $" + section)

        yield l.strip()

        if i == endix:
            break
    else:
        raise ValueError("Unexpected EOF")

    if next(mshit) != endln:
        raise ValueError("Expected $End" + section)


class GmshReader(BaseReader):
    # Supported file types and extensions
    name = "gmsh"
    extn = [".msh"]

    # Gmsh element types to FRFS type (petype) and node counts
    _etype_map = {
        1: ("line", 2),
        8: ("line", 3),
        26: ("line", 4),
        27: ("line", 5),
        2: ("tri", 3),
        9: ("tri", 6),
        21: ("tri", 10),
        23: ("tri", 15),
        15: ("point", 1),
    }

    # First-order node numbers associated with each element face
    _petype_fnmap = {
        "line": {"point": [[0], [1]]},
        "tri": {"line": [[0, 1], [1, 2], [2, 0]]},
    }

    # Mappings between the node ordering of FRFS and that of Gmsh
    _nodemaps = GmshNodeMaps

    def __init__(self, msh):
        # compute uuid
        hasher = hashlib.new("sha256")
        with open(msh.name, "rb") as f:
            while chunk := f.read():
                hasher.update(chunk)
        self._uuid = hasher.hexdigest()

        # Get an iterator over the lines of the mesh
        mshit = iter(msh)

        # Required section readers
        sect_map = {
            "MeshFormat": self._read_mesh_format,
            "Nodes": self._read_nodes,
            "Elements": self._read_eles,
            "PhysicalNames": self._read_phys_names,
            "Entities": self._read_entities,
        }

        for l in filter(lambda l: l != "\n", mshit):
            # Ensure we have encountered a section
            if not l.startswith("$"):
                raise ValueError("Expected a mesh section")

            # Strip the '$' and '\n' to get the section name
            sect = l[1:-1]

            # If the section is known then read it
            if sect in sect_map:
                sect_map[sect](mshit)
            else:
                endsect = "$End{}\n".format(sect)

                for el in mshit:
                    if el == endsect:
                        break
                else:
                    raise ValueError("Expected $End" + sect)

        # Account for any starting node offsets
        for k, v in self._elenodes.items():
            v -= self._nodeoff

    def _read_mesh_format(self, mshit):
        ver, ftype, dsize = next(mshit).split()

        if ver != "4.1":
            raise ValueError("Invalid mesh version")
        if ftype != "0":
            raise ValueError("Invalid file type")
        if dsize != "8":
            raise ValueError("Invalid data size")

        if next(mshit) != "$EndMeshFormat\n":
            raise ValueError("Expected $EndMeshFormat")

    def _read_phys_names(self, msh):
        # Physical entities can be divided up into:
        #  - fluid elements ('the mesh')
        #  - boundary faces
        #  - periodic faces
        self._felespent = None
        self._bfacespents = {}
        self._pfacespents = defaultdict(list)

        # Seen physical names
        seen = set()

        # Extract the physical names
        for l in msh_section(msh, "PhysicalNames"):
            m = re.match(r'(\d+) (\d+) "((?:[^"\\]|\\.)*)"$', l)
            if not m:
                raise ValueError("Malformed physical entity")

            pent, name = int(m.group(2)), m.group(3).lower()

            # Ensure we have not seen this name before
            if name in seen:
                raise ValueError("Duplicate physical name: {}".format(name))

            # Fluid elements
            if name == "fluid":
                self._felespent = pent
            # Periodic boundary faces
            elif name.startswith("periodic"):
                p = re.match(r"periodic[ _-]([a-z0-9]+)[ _-](l|r)$", name)
                if not p:
                    raise ValueError("Invalid periodic boundary condition")

                self._pfacespents[p.group(1)].append(pent)
            # Other boundary faces
            else:
                self._bfacespents[name] = pent

            seen.add(name)

        if self._felespent is None:
            raise ValueError("No fluid elements in mesh")

        if any(len(pf) != 2 for pf in self._pfacespents.values()):
            raise ValueError("Unpaired periodic boundary in mesh")

    def _read_entities(self, mshit):
        self._tagpents = tagpents = {}

        # Obtain the entity counts
        npts, *ents = (int(i) for i in next(mshit).split())

        # Skip over the point entities
        for i in range(npts):
            next(mshit)

        # Iterate through the curves, surfaces, and volume entities
        for ndim, nent in enumerate(ents, start=1):
            for j in range(nent):
                ent = next(mshit).split()
                etag, enphys = int(ent[0]), int(ent[7])

                if enphys == 0:
                    continue
                elif enphys == 1:
                    tagpents[ndim, etag] = abs(int(ent[8]))
                else:
                    raise ValueError("Invalid physical tag count for entity")

        if next(mshit) != "$EndEntities\n":
            raise ValueError("Expected $EndEntities")

    def _read_nodes(self, mshit):
        # Entity count, node count, minimum and maximum node numbers
        ne, nn, ixl, ixu = (int(i) for i in next(mshit).split())

        self._nodepts = nodepts = np.empty((ixu - ixl + 1, 3))
        nodepts.fill(np.nan)

        for i in range(ne):
            nen = int(next(mshit).split()[-1])
            nix = [int(next(mshit)) for _ in range(nen)]

            for j in nix:
                nodepts[j - ixl] = next(mshit).split()

        # Save the starting node offset
        self._nodeoff = ixl

        if next(mshit) != "$EndNodes\n":
            raise ValueError("Expected $EndNodes")

    def _read_eles(self, mshit):
        elenodes = defaultdict(list)

        # Block and total element count
        nb, ne = (int(i) for i in next(mshit).split()[:2])

        for i in range(nb):
            edim, etag, etype, ecount = (int(j) for j in next(mshit).split())

            if etype not in self._etype_map:
                raise ValueError(f"Unsupported element type {etype}")

            # Determine the number of nodes associated with each element
            nnodes = self._etype_map[etype][1]

            # Lookup the physical entity type
            epent = self._tagpents[edim, etag]

            # Allocate space for, and read in, these elements
            enodes = np.empty((ecount, nnodes), dtype=np.int64)
            for j in range(ecount):
                enodes[j] = next(mshit).split()[1:]

            elenodes[etype, epent].append(enodes)

        if ne != sum(len(vv) for v in elenodes.values() for vv in v):
            raise ValueError("Invalid element count")

        if next(mshit) != "$EndElements\n":
            raise ValueError("Expected $EndElements")

        self._elenodes = {k: np.vstack(v) for k, v in elenodes.items()}

    def _to_raw(self):
        # Assemble a nodal mesh
        maps = self._etype_map, self._petype_fnmap, self._nodemaps
        pents = self._felespent, self._bfacespents, self._pfacespents
        mesh = NodalMeshAssembler(self._nodepts, self._elenodes, pents, maps)

        rawm = {}
        rawm.update(mesh.get_connectivity())
        rawm.update(mesh.get_shape_points())
        rawm.update({"mesh_uuid": self._uuid})
        return rawm
