'''This module first conducts gradient based searching for the optimally
distinguished 2x2 game matrices for a subset of parameter space. It then 
conducts exhaustive searching for the optimally distinguished 2x2 game matrices
on the subset of distinguishable matrices identified and find the optimal 
matrix based on data simulation.'''

from searches import simple_search, exhaust
import numpy as np
from plot import *

if __name__ == "__main__":
    # Create the space of potentially optimal matrices
    opt_space = []

    # Search given constant param spaces for optimal matrices
    for tau in np.arange(0.5, 3.5, 0.5):
        for lamb in np.arange(1, 5):
            for alpha in np.arange(0.3, 0.7, 0.1):
                opt_space.append(simple_search.opt_search(10, (tau, alpha, lamb)))

    # Find the most distinguishable matrix in the space with data simulation
    dist, opt_mat, (t, l, a_l1, a_pch) = exhaust.exhaustive_search(opt_space)
    print("Most distinguishable matrix:")
    print(opt_mat)
    print("Distinguishability: {}".format(dist))
    print("PCH: t={}, a={}".format(t, a_pch))
    print("QRE: l={}".format(l))
    print("Level 1: a={}".format(a_l1))
    plot_predictions(opt_mat, 10, t, l, a_pch, a_l1)
