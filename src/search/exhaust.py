''' This module conducts an exhaustive search over 2x2 games with payoffs in
a chosen range for games which maximally distinguish the Nash, QRE, level-1(alpha),
and PCH strategies. ML Simulated play is used to "guess" the value of
the parameter necessary for computing predictions for QRE, level-1(alpha), and PCH.

Games above a certain threshold of distinguishability are chosen as candidates for further
investigation.
'''

import data_sim.gen_pop as nn
import game
from solvers import nash, qre, pch
from distinguish.utils import disting
import itertools

def exhaustive_search(upper_bound, min_dist):
    '''Performs an exhaustive search over games with parameters up to int upper_bound.
    Don't use upper_bound > 3 unless you want to be sitting here for hundreds of hours.
    
    Returns list of "candidate" games with distinguishability exceeding float min_dist'''
    # generate game board
    # sim data on board
    # est params (and p*,q*) from data
    # compute distinguishability
    # add to candidate list or reject
    # return list of candidate game objects
    candidates = []
    ub = upper_bound+1
    iters = [range(ub), range(ub), range(ub), range(ub),
             range(1,ub), range(1,ub), range(1,ub), range(1,ub)]
    games = list(itertools.product(*iters))
    for i in range(len(games)):
        
        # Create game
        (row_payoffs, col_payoffs) = game.payoffs_from_params(games[i])
        temp_game = game.GameBoard(row_payoffs, col_payoffs)
        if nash.pure_nash(temp_game) == []: # check if it has pure nash just in case
            mn = nash.mixed_nash(temp_game)
            if mn[0] > 0 and mn[0] < 1 and mn[1] > 0 and mn[1] < 1:        
                # Simulate play
                sim_data = nn.sim_data(temp_game, plot=False)

                # Generate predictions of the models using the simulated data
                (pch_p, pch_q), t = pch.pch_est(temp_game, sim_data)
                (qre_p, qre_q), l = qre.qre_est(temp_game, sim_data)
                nash_p, nash_q = nash.mixed_nash(temp_game)

                # Compute distinguishability
                dist = disting([(pch_p, pch_q), (qre_p, qre_q), (nash_p, nash_q)])
                if dist >= min_dist:
                    print(dist)
                    candidates.append(temp_game)
        
    return candidates    