"""This module constains utility methods needed for measuring the
distinguishability of the different points in euclidean space based 
on the harmonic mean of their euclidean distances"""


def euc_dist(p1, p2):
    """Computes the euclidean distance between two points"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**(1/2)


def H_mean(vals):
    '''Computes the harmonic mean of a number of values.'''
    return len(vals) / sum([1 / v for v in vals])
    

def disting(points):
    """Computes the distinguishability of a set of points given by 
    the harmonic mean of the euclidean distances between each point"""
    return H_mean([euc_dist(p1, p2) for i, p1 in enumerate(points) for p2 in
                   points[i + 1:]])
