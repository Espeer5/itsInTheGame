"""This module provides some plotting utilities for vizualizing the results 
of different game theoretic models on 2x2 matrices."""

import matplotlib.pyplot as plt
from solvers import nash, qre, pch, level1


def plot_predictions(game, top_k, tau, lamb, alpha):
    """Plot the predictions of the Nash, PCH and QRE models for a given game and
    poisson parameter tau"""
    ne_p, ne_q = nash.mixed_nash(game)
    pch_p, pch_q = pch.pch(game, top_k, tau, alpha=alpha)
    qre_p, qre_q = qre.qre(game, lamb)
    l1_p, l1_q = level1.level_1(game, alpha)
    plt.plot(pch_p, pch_q, 'ro', label='PCH, t={}'.format(tau))
    plt.plot(qre_p, qre_q, 'bo', label='QRE Arc')
    plt.plot(ne_p, ne_q, 'go', label='Nash Equilibrium')
    plt.plot(l1_p, l1_q, 'ko', label='Level 1')
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.xlabel('p')
    plt.ylabel('q')
    plt.title('Model Predictions')
    plt.legend(loc='upper right')
    plt.savefig("mixed_models.png")
    plt.cla()
