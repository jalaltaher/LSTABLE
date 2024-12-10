# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:35:07 2024

@author: Jalal
"""

import numpy as np
from . import CTS_distribution


def increments_CTS_generator(
    n_increments: int,
    Delta: float,
    alpha: float,
    P: float,
    Q: float,
    A: float,
    B: float,
    drift: float,
    c: float = 0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generates increments of a CTS process with drift.

    Parameters:
        n_increments (int): Number of increments to generate.
        Delta (float): Sampling rate.
        alpha (float): Parameter of the CTS process.
        P, Q, A, B (float): Parameters of the CTS process.
        drift (float): Drift coefficient.
        c (float, optional): Additional parameter, default is 0.
        verbose (bool, optional): Verbose output giving time remaining, default is False.

    Returns:
        np.ndarray: Array of shape (n_increments,) representing the increments.
    """
    increments_without_drift = CTS_generator_Bauemer_vectorial(
        alpha, Delta * P, Delta * Q, A, B, n_increments, c, verbose
    )
    return increments_without_drift + Delta * drift


def trajectory_CTS_generator(
    n_increments: int,
    n_trajectories: int,
    Delta: float,
    alpha: float,
    P: float,
    Q: float,
    A: float,
    B: float,
    drift: float,
    c: float = 0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generates trajectories of a CTS process with drift.

    Parameters:
        n_increments (int): Number of increments per trajectory.
        n_trajectories (int): Number of trajectories to generate.
        Delta (float): Sampling rate.
        alpha (float): Parameter of the CTS process.
        P, Q, A, B (float): Parameters of the CTS process.
        drift (float): Drift coefficient.
        c (float, optional): Additional parameter, default is 0.
        verbose (bool, optional): Verbose output giving time remaining, default is False.

    Returns:
        np.ndarray: Array of shape (n_trajectories, n_increments + 1)
        if `n_trajectories > 1`, else a 1D array of shape (n_increments + 1,).
    """
    increment_matrix = np.array(
        [
            increments_CTS_generator(n_increments, Delta, alpha, P, Q, A, B, drift, c, verbose)
            for _ in range(n_trajectories)
        ]
    )

    # Ensure the matrix starts with zeros (initial position)
    increment_matrix_with_0 = np.hstack([np.zeros((n_trajectories, 1)), increment_matrix])

    # Compute the cumulative sum along increments
    res = np.cumsum(increment_matrix_with_0, axis=1)

    # Return a 1D array if there's only one trajectory
    if n_trajectories == 1:
        return res[0]
    return res
