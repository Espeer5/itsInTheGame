''' This module conducts a constrained optimization to determine a 2x2 game with payoffs in
a chosen range for a game which maximally distinguishes the Nash, QRE, level-1(alpha),
and PCH strategies. ML Simulated play is used to "guess" the value of
the parameter necessary for computing predictions for QRE, level-1(alpha), and PCH.

We utilize the module scipy.optimize to compute the optimization.
'''