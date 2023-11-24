"""This solver is a level one model in which players throughout the population 
play as a best response to a level 0 distribution. Here level 0 is assigned to 
be (.5, .5). The best response is given by the same as the first step in Poisson
CH solver.
"""

import solvers.utils

def level_1(game, alpha=None):
    """Returns the p and q values for a given game as a best response to level 
    0 (random mixing) play."""

    # If risk aversion parameter alpha given, transform game matrix
    if alpha:
        game = solvers.utils.tranform_game(game, alpha)

    p_0 = [(0.5, 0.5)]
    q_0 = [(0.5, 0.5)]

    # Compute row probabilities for level 1
    E_R_0 = sum([game.get_row_matrix()[0][c] * q_0[0][c] for c in range(0, 2)])
    E_R_1 = sum([game.get_row_matrix()[1][c] * q_0[0][c] for c in range(0, 2)])
    if E_R_0 > E_R_1:
        p = (1, 0)
    elif E_R_0 < E_R_1:
        p = (0, 1)
    else:
        p = (0.5, 0.5)

    # Compute column probabilities for level 1
    E_C_0 = sum([game.get_col_matrix()[r][0] * p_0[0][0] for r in range(0, 2)])
    E_C_1 = sum([game.get_col_matrix()[r][1] * p_0[0][1] for r in range(0, 2)])
    if E_C_0 > E_C_1:
        q = (1, 0)
    elif E_C_0 < E_C_1:
        q = (0, 1)
    else:
        q = (0.5, 0.5)

    return p, q
