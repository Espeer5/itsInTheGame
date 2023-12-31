"""Runs gradient based search methods with constant parameter values to find 
maximally distinguished matrices for ranges of parameters. Intended to produce
spaces of matrices to search over using data simulation."""

import game
from solvers import nash, qre, pch, level1
from distinguish.utils import disting
import scipy.optimize as opt
import plot

# Constants
TOP_K = 10

def e_determinate(params, tau, alpha, lamb):
    """Computes the distinguishability of a game given its parameters. Should 
    be maximized to find the most distinguishable games."""
    (row_payoffs, col_payoffs) = game.payoffs_from_params(params)
    g = game.GameBoard(row_payoffs, col_payoffs)

    # Generate predictions of the models using the simulated data
    pch_p, pch_q = pch.pch(g, TOP_K, tau, alpha=alpha)
    l1_p, l1_q = level1.level_1(g, alpha)
    qre_p, qre_q = qre.qre(g, lamb)
    nash_p, nash_q = nash.mixed_nash(g)

    dist = disting([(pch_p, pch_q), (qre_p, qre_q), (nash_p, nash_q), 
                    (l1_p, l1_q)])

    # Return 1/distinguishability to use for minimization
    return -dist


def opt_search(upper_bound, consts):
    """Performs a search over games with parameters up to float upper_bound via 
    optimization.
    
    Returns a game object with the "maximally distinguishable" game and the 
    distinguishability of that game"""
    ub = upper_bound
    x0 = (ub, ub/2, ub/3, ub/4, ub/4, ub/3, ub/2, ub)
    param_bounds = [(0, ub), (0, ub), (0, ub), (0, ub), (1, ub), (1, ub),
                     (1, ub), (1, ub)]
    result = opt.minimize(e_determinate, x0, args=consts, method=None,
                           bounds=param_bounds)
    (row_payoffs, col_payoffs) = game.payoffs_from_params(result.x)
    opt_game = game.GameBoard(row_payoffs, col_payoffs)
    return opt_game
    