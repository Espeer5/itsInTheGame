"""This module constains utility methods needed for measuring the
distinguishability of the three models on a single game matrix.
Distinguishability is here the average distance of any single model's 
prediction from the centroid of the three model predictions, which 
accounts not only for the pairwise distance between the models, but how 
dispersed and non-central all 3 models are from each other."""


def centroid(points):
    """Computes the centroid of any number of points"""
    x_s = 0
    y_s = 0
    for p in points:
        x_s += p[0]
        y_s += p[1]
    return (x_s / len(points), y_s / len(points))


def euc_dist(p1, p2):
    """Computes the euclidean distance between two points"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**(1/2)

def H_mean(points):
    '''Computes the harmonic mean of a number of points.
    Harmonic mean is dominated by the minimum of its arguments,
    which will help with maximally distinguishing all four points.'''
    x = 0
    y = 0
    for p in points:
        x += 1/p[0]
        y += 1/p[1]
    return (len(points) / x, len(points) / y)


def disting(points):
    """Computes the distinguishability of a set of points given by 
    the average distance from the centroid of the points"""
    cent = centroid(points)
    dists = []
    for p in points:
        dists.append(euc_dist(p, cent))
    return sum(dists) / len(dists)