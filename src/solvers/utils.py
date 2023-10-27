"""Utility methods for solvers"""

import math


def euclid_error(model, data):
    """euclidean distance between two points on the (p, q) plane"""
    return math.sqrt((model[1] - data[1])**2 + (model[0] - data[0])**2)