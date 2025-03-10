#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import re
import subprocess
import sys

from setuptools import setup

# Python version
if sys.version_info[:2] < (3, 10):
    logging.error("edgfs2D requires Python 3.10 or newer")
    sys.exit(-1)

# EDGFS version
vfile = open("_version.py").read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)
else:
    logging.error("Unable to find a version string in _version.py")

# Modules
modules = [
    "edgfs2D",
    "edgfs2D.basis",
    "edgfs2D.basis.tri",
    "edgfs2D.boundary_conditions",
    "edgfs2D.distribution_mesh",
    "edgfs2D.entropy_fluxes",
    "edgfs2D.fields",
    "edgfs2D.fields.readers",
    "edgfs2D.fields.writers",
    "edgfs2D.fluxes",
    "edgfs2D.initial_conditions",
    "edgfs2D.integrators",
    "edgfs2D.limiters",
    "edgfs2D.physical_mesh",
    "edgfs2D.physical_mesh.readers",
    "edgfs2D.plugins",
    "edgfs2D.post_process",
    "edgfs2D.proto",
    "edgfs2D.quadratures",
    "edgfs2D.scattering",
    "edgfs2D.solvers",
    "edgfs2D.solvers.advection",
    "edgfs2D.solvers.fast_spectral",
    "edgfs2D.solvers.fast_spectral.formulations",
    "edgfs2D.sphericaldesign",
    "edgfs2D.sphericaldesign.symmetric",
    "edgfs2D.time",
    "edgfs2D.utils",
    "edgfs2D.velocity_mesh",
]

# Data
package_data = {
    "edgfs2D.proto": ["*.proto"],
    "edgfs2D.basis": ["tri/*.pb"],
    "edgfs2D.sphericaldesign": ["symmetric/*.txt"],
    # "edgfs2D.std.kernels": ["*.mako"],
    # "edgfs2D.std.kernels.bcs": ["*.mako"],
    # "edgfs2D.std.kernels.scattering": ["*.mako"],
}

# Hard dependencies
install_requires = [
    "h5py >= 2.6",
    "numpy >= 1.8",
    "torch >= 2.5.1",
    "protobuf>=4.0.0",
    "isort>=5.13.2",
    "loguru>=0.7.3",
]

# Soft dependencies
extras_require = {"all": ["pyvista >= 0.44", "autoflake >= 2.3"]}

# Scripts
console_scripts = [
    "edgfsAdv2D = edgfs2D.solvers.advection.main:__main__",
    "edgfsFs2D = edgfs2D.solvers.fast_spectral.main:__main__",
    "edgfsPost2D = edgfs2D.post_process.main:__main__",
]

# Info
classifiers = [
    "License :: GNU GPL v2",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.3",
    "Topic :: Scientific/Engineering",
]

description = (
    "Entropy Stable Discontinuous Galerkin Fast Spectral in Two Dimensions"
)

long_description = """edgfs2D is an open-source minimalistic implementation of
Entropy-Stable Discontinuous Galerkin Fast Spectral methods in two dimension"""


def compile_proto():
    proto_dir = "edgfs2D/proto"
    output_dir = "."
    proto_files = [f for f in os.listdir(proto_dir) if f.endswith(".proto")]
    for proto_file in proto_files:
        input_path = os.path.join(proto_dir, proto_file)
        subprocess.run(
            ["protoc", f"--python_out={output_dir}", input_path], check=True
        )


compile_proto()

setup(
    name="edgfs2D",
    version=version,
    description=description,
    long_description=long_description,
    author="Shashank Jaiswal",
    author_email="jaisw7@gmail.com",
    url="http://www.github.com/jaisw7",
    license="GNU GPL v2",
    keywords="Applied Mathematics",
    packages=["edgfs2D"] + modules,
    package_data=package_data,
    entry_points={"console_scripts": console_scripts},
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
)
