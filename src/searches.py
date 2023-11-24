'''This module conducts an exhaustive search and a search via optimization
for the optimally distinguishable game board.'''

from search import exhaust, opt_search
import numpy as np
import game
from solvers import nash, qre, pch
import data_sim.gen_pop as nn
import matplotlib.pyplot as plt
from distinguish.utils import disting

if __name__ == "__main__":
    g = opt_search.opt_search(10)
    print(g)
    exhaust.exhaustive_search(3, 0.25)