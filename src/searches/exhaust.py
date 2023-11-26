"""This module conducts an exhaustive search over 2x2 games with payoffs in
a chosen range for games which maximally distinguish the Nash, QRE, level-1(alpha),
and PCH strategies. ML Simulated play is used to "guess" the value of
the parameter necessary for computing predictions for QRE, level-1(alpha), and PCH.
"""

import data_sim.gen_pop as nn
from solvers import nash, qre, pch, level1
from distinguish.utils import disting
import numpy as np

# Constants
SIM_RUNS = 5 # runs to average over for simulated data
TOP_K = 5 # top k level to use for computing PCH


def exhaustive_search(search_space):
    '''Performs an exhaustive search over games passed directly in the 
    search_space list. Performs data simulation '''
    # For each matrix, estimate the parameters of the PCH, QRE, and 
    # level-1(alpha) models through repeated data simulation
    params = []
    for g in search_space:
        estimates = [[], [], [], []]
        for _ in range(SIM_RUNS):
            # Generate simulated data on the game
            sim_data = nn.sim_data(g, plot=False)

            # Generate predictions of the models using the simulated data
            _, (t, a_pch) = pch.pch_est(g, sim_data, alpha=True)
            _, a_l1 = level1.level1_est(g, sim_data)
            _, l = qre.qre_est(g, sim_data)

            # Add the estimates to the list of estimated values
            estimates[0].append(t)
            estimates[1].append(l)
            estimates[2].append(a_l1)
            estimates[3].append(a_pch)
        # Average the estimates
        params.append(tuple(np.mean(estimates[i]) for i in range(4)))

    # For each matrix, compute the distinguishability using the estimated 
    # parameters. Then find and return the matrix with maximal
    # distinguishability
    dists = []
    for i, g in enumerate(search_space):
        paramset = params[i]
        pch_p, pch_q = pch.pch(g, TOP_K, paramset[0], alpha=paramset[3])
        l1_p, l1_q = level1.level_1(g, paramset[2])
        qre_p, qre_q = qre.qre(g, paramset[1])
        nash_p, nash_q = nash.mixed_nash(g)
        dist = disting([(pch_p, pch_q), (qre_p, qre_q), (nash_p, nash_q),
                         (l1_p, l1_q)])
        dists.append(dist)
    max = np.argmax(dists)
    return dists[max], search_space[max], params[max]
