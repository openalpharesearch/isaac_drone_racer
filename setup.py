# Copyright (c) 2025, Kousheek Chakraborty
# Forked and maintained by Ai Robotics @ Berkeley
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""Setuptools configuration for the ``isaac_drone_racer`` package.

Reads package metadata from ``extension.toml`` so that version, author, and
description stay in a single source of truth.
"""

import os

import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "extension.toml"))
pkg = EXTENSION_TOML_DATA["package"]

setup(
    name="isaac_drone_racer",
    packages=find_packages(),
    author=pkg["author"],
    maintainer=pkg["maintainer"],
    url=pkg["repository"],
    version=pkg["version"],
    description=pkg["description"],
    keywords=pkg["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    entry_points={
        "console_scripts": [
            "isaac_ros_bridge_node = ros_bridge.interface_node:main",
        ],
    },
    zip_safe=False,
)
