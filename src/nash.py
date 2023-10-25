""" This module computes the Nash equilibrium solution for a 2x2 Game 
object. The Nash solution is either pure or mixed strategy, in both cases 
represented as the probability of playing each strategy as a tuple of 
(row player p(row = 1), col player p(col = 1))."""

# Compute all pure nash equilibria solutions for a 2x2 game
# Returns [] if no pure nash equilibria exist
def pure_nash(game):
    """Compute all pure nash equilibria solutions for a 2x2 game
    Returns [] if no pure nash equilibria exist
    """
    r_maxes = [max(game.get_row_matrix()[r]) for r in range(2)]
    c_maxes = [max([game.get_col_matrix()[r][c] 
                    for r in range(2)]) for c in range(2)]
    equilibria = []
    for i in range(2):
        for j in range(2):
            scores = game[i, j]
            if scores[0] == r_maxes[i] and scores[1] == c_maxes[j]:
                equilibria.append((i,j))
    return equilibria


def mixed_nash(game):
    """ Ensures there is no pure nash equilibrium, then computes and returns 
    the mixed Nash equilibrium"""

    # Ensure there is no pure Nash equilibrium strategy
    try:
        assert(pure_nash(game) == [])
    except AssertionError:
        raise ValueError("Game has pure nash equilibria")
    
    # Compute the mixed nash equilibrium
    row_pay = game.get_row_matrix()
    col_pay = game.get_col_matrix()
    p = ((col_pay[1][1] - col_pay[1][0]) / 
        (col_pay[0][0] + col_pay[1][1] - col_pay[0][1] - col_pay[1][0]))
    q = ((row_pay[1][1] - row_pay[0][1]) /
        (row_pay[0][0] + row_pay[1][1] - row_pay[0][1] - row_pay[1][0]))
    return (p, q)
    