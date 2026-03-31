# Copyright (c) 2025, Kousheek Chakraborty
# Forked and maintained by Ai Robotics @ Berkeley
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""CSV-based telemetry logger with automatic plot generation.

Provides :class:`CSVLogger` for streaming per-step scalar metrics to disk and
the helper :func:`log` for attaching metrics to an environment's extras dict.
"""

import csv
import os
from datetime import datetime

import torch

from utils.plotter import generate_plots


class CSVLogger:
    """Append-mode CSV logger that auto-generates plots on save.

    Each :meth:`log` call appends a single row.  Column headers are inferred
    from the first call and extended dynamically if new keys appear later.

    Attributes:
        file_path: Path to the active CSV file.
        keys: Current ordered list of column names.
        file_initialized: Whether the header row has been written.
    """

    def __init__(self, folder_path="."):
        """Create a new CSV log file in *folder_path*.

        Args:
            folder_path: Directory in which to create the timestamped CSV.

        Raises:
            FileNotFoundError: If *folder_path* does not exist.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(folder_path, f"log_{timestamp}.csv")
        self.keys = []  # Keeps track of column headers
        self.file_initialized = False

    def log(self, data_dict):
        """Append a row of scalar metrics to the CSV file.

        All tensor values must have shape ``(1,)`` -- they are converted to
        Python scalars before writing.

        Args:
            data_dict: Mapping of column names to single-element tensors.

        Raises:
            ValueError: If any value is not a tensor or has unexpected shape.
        """
        # Verify that all tensors have n = 1
        for key, tensor in data_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Value for key '{key}' must be a tensor.")
            if tensor.ndim != 1 or tensor.shape[0] != 1:
                raise ValueError(f"Tensor for key '{key}' must have shape (1,), but got {tensor.shape}.")

        # Flatten tensors to scalar values (since n = 1, we can extract the single value)
        flattened_data = {key: tensor.item() for key, tensor in data_dict.items()}

        # Initialize the CSV file if not already done
        if not self.file_initialized:
            self.keys = list(flattened_data.keys())
            with open(self.file_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.keys)
                writer.writeheader()
            self.file_initialized = True

        # Check for new keys and update the CSV header if necessary
        new_keys = [key for key in flattened_data.keys() if key not in self.keys]
        if new_keys:
            self.keys.extend(new_keys)
            # Rewrite the CSV file with the updated header
            with open(self.file_path) as file:
                rows = list(csv.DictReader(file))
            with open(self.file_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.keys)
                writer.writeheader()
                writer.writerows(rows)

        # Write the new row
        with open(self.file_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.keys)
            # Fill in missing keys with empty values
            row = {key: flattened_data.get(key, "") for key in self.keys}
            writer.writerow(row)

    def save(self):
        """Finalise the current log, generate plots, and start a new file.

        Raises:
            RuntimeError: If no data has been logged yet.
        """
        # Ensure the current file is saved (already handled by the log method)
        if not self.file_initialized:
            raise RuntimeError("No file has been initialized yet. Log some data first.")

        generate_plots(self.file_path)
        # Generate a new file name with a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(os.path.dirname(self.file_path), f"log_{timestamp}.csv")

        # Reset keys and reinitialize the file
        self.keys = []
        self.file_initialized = False


def log(env, keys, value):
    """Store per-step metric columns in ``env.extras['metrics']``.

    Args:
        env: Environment instance whose ``extras`` dict receives the data.
        keys: List of column name strings, one per value column.
        value: Tensor of shape ``(num_envs, len(keys))``.

    Raises:
        TypeError: If *keys* is not a list of strings.
        ValueError: If *keys* length doesn't match the second dimension
            of *value*.
    """
    if "metrics" not in env.extras:
        env.extras["metrics"] = {}

    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise TypeError("keys must be a list of strings.")

    if len(keys) != value.shape[1]:
        raise ValueError(f"Length of keys ({len(keys)}) must match the second dimension of value ({value.shape[1]}).")

    for i, key in enumerate(keys):
        env.extras["metrics"][key] = value[:, i]
