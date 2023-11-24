'''This module conducts an exhaustive search and a search via optimization
for the optimally distinguishable game board.'''

from search import exhaust, opt_search

if __name__ == "__main__":
    exhaust.exhaustive_search(2, 0.15)