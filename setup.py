#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import subprocess
import sys

from setuptools import setup

# Python version
if sys.version_info[:2] < (3, 3):
    print("edgfs2D requires Python 3.3 or newer")
    sys.exit(-1)

# DGFS version
vfile = open("_version.py").read()
vsrch = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", vfile, re.M)

if vsrch:
    version = vsrch.group(1)
else:
    print("Unable to find a version string in _version.py")

# Modules
modules = [
    "edgfs2D.boundary_conditions",
    "edgfs2D.basis",
    "edgfs2D.basis.tri",
    "edgfs2D.boundary_conditions",
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
    "edgfs2D.proto",
    "edgfs2D.quadratures",
    "edgfs2D.sphericaldesign",
    "edgfs2D.sphericaldesign.symmetric",
    "edgfs2D.solvers",
    "edgfs2D.solvers.advection",
    "edgfs2D.solvers.fast_spectral.std.kernels",
    "edgfs2D.solvers.fast_spectral.std.kernels.bcs",
    "edgfs2D.solvers.fast_spectral.std.kernels.scattering",
    "edgfs2D.solvers.fast_spectral.std.scattering",
    "edgfs2D.velocity_mesh",
    "edgfs2D.utils",
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
    "appdirs >= 1.4.0",
    "gimmik >= 2.0",
    "h5py >= 2.6",
    "mako >= 1.0.0",
    "numpy >= 1.8",
    "pytools >= 2016.2.1",
    "torch >= 2.5.1",
    "protobuf>=3.0.0",
    "isort>=5.13.2",
]

# Soft dependencies
extras_require = {}

# Scripts
console_scripts = [
    "edgfsAdv2D = edgfs2D.solvers.advection.advection:__main__",
    "edgfsFs2D = edgfs2D.solvers.fast_spectral.fast_spectral:__main__",
]

# Info
classifiers = [
    "License :: GNU GPL v2",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.3",
    "Topic :: Scientific/Engineering",
]

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
    description="Entropy Stable Discontinuous Galerkin Fast Spectral in Two Dimension",
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
