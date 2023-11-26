""" This module contains an object representing a 2x2 game board. The object 
constains independent payoff matrices for the row and columm players giving 
their payoffs for each strategy.
"""

import numpy as np


def pad(str, length):
    """Pad a string with spaces to the given length as evenly as possible. If 
    the difference between the length of the string and the desired length is
    odd, the extra space will be added to the right side of the string."""
    diff = length - len(str)
    if diff == 0:
        return str
    elif diff == 1:
        return str + " "
    elif diff == 2:
        return " " + str + " "
    elif diff % 2 == 0:
        return " " * (diff // 2) + str + " " * (diff // 2)
    else:
        return " " * (diff // 2) + str + " " * (diff // 2 + 1)


class GameBoard:
    """ This class represents a 2x2 game board. It contains independent payoff
    matrices for the row and column players giving their payoffs for each"""

    def __init__(self, row_payoff, col_payoff):
        """Initialize the game board with the payoff matrices for the row and
        column players"""
        self.row_player = row_payoff
        self.col_player = col_payoff

    def __getitem__(self, key):
        """Retrieve the payoff tuple given the row and column
        strategies via indexing"""
        return (self.row_player[key[0]][key[1]], self.col_player[key[0]][key[1]])
    
    def get_row_matrix(self):
        """Retrieve the row player's payoff matrix"""
        return self.row_player
    
    def get_col_matrix(self):
        """Retrieve the column player's payoff matrix"""
        return self.col_player
    
    def row_payoff(self, row_strat, col_strat):
        """Retrieve the payoff for the row player given the row and column"""
        return self.row_player[row_strat][col_strat]
    
    def col_payoff(self, row_strat, col_strat):
        """Retrieve the payoff for the column player given the row and column"""
        return self.col_player[row_strat][col_strat]
    
    def __str__(self):
        """Return a string representation of the game board"""
        rcs = [f'{self.row_player[0][0]:.2f}, {self.col_player[0][0]:.2f}',
               f'{self.row_player[0][1]:.2f}, {self.col_player[0][1]:.2f}',
               f'{self.row_player[1][0]:.2f}, {self.col_player[1][0]:.2f}',
               f'{self.row_player[1][1]:.2f}, {self.col_player[1][1]:.2f}']
        conlen = max([len(rc) for rc in rcs])
        return ("|" + "-" * (conlen + 4) + "|" + "-" * (conlen + 4) + "|\n" +
                "|" + " " * (conlen + 4) + "|" + " " * (conlen + 4) + "|\n" +
                "|" + "  " + pad(rcs[0], conlen) + "  |  " + 
                  pad(rcs[1], conlen) + "  |\n" +
                "|" + " " * (conlen + 4) + "|" + " " * (conlen + 4) + "|\n" +
                "|" + "-" * (conlen + 4) + "|" + "-" * (conlen + 4) + "|\n" +
                "|" + " " * (conlen + 4) + "|" + " " * (conlen + 4) + "|\n" +
                "|" + "  " + pad(rcs[2], conlen) + "  |  " + 
                  pad(rcs[3], conlen) + "  |\n" +
                "|" + " " * (conlen + 4) + "|" + " " * (conlen + 4) + "|\n" +
                "|" + "-" * (conlen + 4) + "|" + "-" * (conlen + 4) + "|\n")
    

def payoffs_from_params(params):
    '''Parameterization of 2x2 board taken from Selten Chmura.
    params is a tuple of floats (al,ar,bu,bd,cl,cr,du,dd).
    a's and b's >= 0, c's and d's > 0.

    Returns (row_payoffs, col_payoffs) each element a 2x2 np array.

    A game board constructed in this way should ensure a uniquely determined,
    completely mixed Nash equilibrium.'''
    row_payoffs = np.array([[params[0] + params[4], params[1]],
                            [params[0], params[1] + params[5]]]).reshape(2,2)
    col_payoffs = np.array([[params[2], params[2] + params[6],
                             params[3] + params[7], params[3]]]).reshape(2,2)
    return (row_payoffs, col_payoffs)


def rand_game():
    '''Returns a random 2x2 game board with a unique mixed Nash equilibrium.
    using the parameterization from Selten Chmura. Useful for testing purposes'''
    params = np.random.rand(8)
    params[4:] += 1
    params[2:4] += 1
    return GameBoard(*payoffs_from_params(params))
