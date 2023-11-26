"""Utility methods for solvers"""

import math
import numpy as np
from game import GameBoard


def euclid_error(model, data):
    """euclidean distance between two points on the (p, q) plane"""
    return math.sqrt((model[1] - data[1])**2 + (model[0] - data[0])**2)


def transform_game(game, alpha):
    """Transforms the game matrix by alpha according to Fudenberg, Liang 2019"""
    row_matrix = game.get_row_matrix()
    col_matrix = game.get_col_matrix()
    a_row = np.array([[row_matrix[0][0] ** alpha, row_matrix[0][1] ** alpha],
                      [row_matrix[1][0] ** alpha, row_matrix[1][1] ** alpha]])
    a_col = np.array([[col_matrix[0][0] ** alpha, col_matrix[0][1] ** alpha],
                      [col_matrix[1][0] ** alpha, col_matrix[1][1] ** alpha]])
    return GameBoard(a_row, a_col)
    