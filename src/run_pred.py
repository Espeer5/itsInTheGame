"""This module constructs a 2x2 game board object from a given payoff matrix, 
then uses neural networks to create simulated data on the board. It then uses
that simulated data to estimate the free parameters of the PCH and QRE models.
It then generates the predictions of each model and plots them against each
other."""

import numpy as np
import game
from solvers import nash, qre, pch
import data_sim.gen_pop as nn
import matplotlib.pyplot as plt
from distinguish.utils import disting
import time

if __name__ == "__main__":
    # Create a game board (currently random) with mixed nash prediction
    board_found = False
    while not board_found:
        row = np.random.randint(0, 5, size=(2, 2))
        col = np.random.randint(0, 5, size=(2, 2))
        g = game.GameBoard(row, col)
        if nash.pure_nash(g) == []:
            mn = nash.mixed_nash(g)
            if mn[0] > 0 and mn[0] < 1 and mn[1] > 0 and mn[1] < 1:
                board_found = True
    tick = time.time()
    # Generate simulated data on the game board
    sim_data = nn.sim_data(g, plot=True)

    # Generate predictions of the models using the simulated data
    (pch_p, pch_q), t = pch.pch_est(g, sim_data)
    (qre_p, qre_q), l = qre.qre_est(g, sim_data)
    nash_p, nash_q = nash.mixed_nash(g)
    tock = time.time()
    print(tock - tick)

    # Plot the predictions of the models
    plt.plot(pch_p, pch_q, 'ro', label='PCH t={}'.format(round(t, 4)))
    plt.plot(qre_p, qre_q, 'bo', label='QRE lambda={}'.format(round(l, 4)))
    plt.plot(nash_p, nash_q, 'go', label='Nash Equilibrium')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('p')
    plt.ylabel('q')
    plt.title('Model Predictions')
    plt.legend(loc='upper right')
    plt.savefig("mixed_models.png")
    plt.clf()

    print(g)
    print("NN Play: {}".format(sim_data))
    print("Nash: {}".format(nash.mixed_nash(g)))
    print("Distinguishability: {}".format(disting([(pch_p, pch_q), 
                                                   (qre_p, qre_q), 
                                                   (nash_p, nash_q)])))