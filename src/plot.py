"""This module provides some plotting utilities for vizualizing the results 
of different game theoretic models on 2x2 matrices."""

import matplotlib.pyplot as plt

import game
import solvers.nash as nash
import solvers.qre as qre
import solvers.pch as pch


def plot_predictions(game, tau):
    """Plot the predictions of the Nash, PCH and QRE models for a given game and
    poisson parameter tau"""
    ne_p, ne_q = nash.mixed_nash(game)
    pch_p, pch_q = pch.pch(game, 10, tau)
    qre_p, qre_q = qre.qre_curve(game, 10, 0.1)
    plt.plot(pch_p, pch_q, 'r*', label='PCH, t={}'.format(tau))
    plt.plot(qre_p, qre_q, 'b', label='QRE Arc')
    plt.plot(ne_p, ne_q, 'go', label='Nash Equilibrium')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('p')
    plt.ylabel('q')
    plt.title('Model Predictions')
    plt.legend(loc='upper right')
    plt.savefig("mixed_models.png")


if __name__ == "__main__":
    row = [[8, 0], [0, 1]]
    col = [[0, 1], [1, 0]]
    g = game.GameBoard(row, col)
    plot_predictions(g, 2)