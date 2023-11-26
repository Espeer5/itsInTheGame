''' This module conducts a constrained optimization to determine a 2x2 game with payoffs in
a chosen range for a game which maximally distinguishes the Nash, QRE, level-1(alpha),
and PCH strategies. ML Simulated play is used to "guess" the value of
the parameter necessary for computing predictions for QRE, level-1(alpha), and PCH.

We utilize the module scipy.optimize to compute the optimization.
'''

import data_sim.gen_pop as nn
import game
from solvers import nash, qre, pch
from distinguish.utils import disting
import scipy.optimize as opt
import numpy as np

def dist_from_params(params):
    (row_payoffs, col_payoffs) = game.payoffs_from_params(params)
    g = game.GameBoard(row_payoffs, col_payoffs)
    if nash.pure_nash(g) == []: # check if it has pure nash just in case
            mn = nash.mixed_nash(g)
            if mn[0] > 0 and mn[0] < 1 and mn[1] > 0 and mn[1] < 1:
                sim_data = nn.sim_data(g, plot=False)

                # Generate predictions of the models using the simulated data
                (pch_p, pch_q), t = pch.pch_est(g, sim_data)
                (qre_p, qre_q), l = qre.qre_est(g, sim_data)
                nash_p, nash_q = nash.mixed_nash(g)

                dist = disting([(pch_p, pch_q), (qre_p, qre_q), (nash_p, nash_q)])
                print(dist)

                # Return 1/distinguishability to use for minimization
                return 1/dist
    return np.inf

def opt_search(upper_bound):
    '''Performs a search over games with parameters up to float upper_bound via optimization.
    
    Returns a game object with the "maximally distinguishable" game and the distinguishability
    of that game'''
    ub = upper_bound
    x0 = (ub,ub/2,ub/3,ub/4,ub/4,ub/3,ub/2,ub)
    param_bounds = [(0,ub), (0,ub), (0,ub), (0,ub), (1,ub), (1,ub), (1,ub), (1,ub)]
    result = opt.minimize(dist_from_params, x0, method=None, bounds=param_bounds, options={'maxiter':100}) # not rlly sure what to use for method??
    (row_payoffs, col_payoffs) = game.payoffs_from_params(result.x)
    opt_game = game.GameBoard(row_payoffs, col_payoffs)
    return opt_game
