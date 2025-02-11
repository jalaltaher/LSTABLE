# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:43:55 2024

@author: Jalal
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from math import pi, log
from .Stable_distribution import stable_to_levy_parameter


def increment_stable_levy_process_generator(
    n: int, Delta: float, alpha: float, P: float, Q: float, drift: float, nb_sample: int
):
    """

    Generates increments of an alpha stable Levy process with triplet (drift,0,nu) where nu is the stable
    Levy measure with parameter alpha,P,Q

    Parameters
    ----------
    n : (int)
        Number of increments
    Delta : float
        Time step for the increments
    alpha : float
        stability index 0<alpha<2
    P : float
        Stable Levy measure parameter for positive jumps
    Q : float
        Stable Levy measure parameter for negative jumps
    drift : float
        Drift term
    nb_sample : int
        Number of trajectories

    Returns (numpy matrix)
    -------
    numpy matrix of size (nb_sample,n) each line corresponding the the increments of a trajectory

    """
    # Parameter conversion
    alpha, sigma, beta, mu = stable_to_levy_parameter(alpha, P, Q, drift)

    # Initial sampling
    sample_matrix = st.levy_stable(alpha, beta).rvs((nb_sample, n))  # Generate a matrix of S_alpha(1,beta,0)
    res = np.zeros((nb_sample, n))

    # Rescaling
    if alpha != 1:
        res = sigma * Delta ** (1 / alpha) * sample_matrix + Delta * mu
    else:
        res = sigma * Delta * sample_matrix + Delta * mu + (2 / pi) * sigma * Delta * log(sigma * Delta) * beta
    return res


def trajectory_stable_Levy_process_generator(
    n: int, Delta: float, alpha: float, P: float, Q: float, drift: float, nb_sample: int
):
    """
    Generates trajectories of an alpha stable Lévy process with triplet (drift,0,nu) and nu the Lévy measure
    is characterized bv alpha,P,Q.

    """
    # Increments generation
    increments_matrix = increment_stable_levy_process_generator(n, Delta, alpha, P, Q, drift, nb_sample)
    # Add a column of zeros as a first value at t=0
    increments_matrix_add0 = np.hstack([np.zeros((nb_sample, 1)), increments_matrix])

    return np.cumsum(increments_matrix_add0, axis=1)
