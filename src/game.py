""" This module contains an object representing a 2x2 game board. The object 
constains independent payoff matrices for the row and columm players giving 
their payoffs for each strategy.
"""

class GameBoard:
    """ This class represents a 2x2 game board. It contains independent payoff
    matrices for the row and column players giving their payoffs for each"""

    # Initialize a 2x2 game board of all zeros
    def __init__(self, row_payoff, col_payoff):
        self.row_player = row_payoff
        self.col_player = col_payoff
    
    # Retrieve the row player's payoff matrix
    def get_row_matrix(self):
        return self.row_player
    
    # Retrieve the column player's payoff matrix
    def get_col_matrix(self):
        return self.col_player
    
    # Retrieve the payoff for the row player given the row and column strategies
    def row_payoff(self, row_strat, col_strat):
        return self.row_player[row_strat][col_strat]
    
    # Retrieve the payoff for the col player given the row and column strategies
    def col_payoff(self, row_strat, col_strat):
        return self.col_player[row_strat][col_strat]
