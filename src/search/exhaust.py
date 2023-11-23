''' This module conducts an exhaustive search over 2x2 games with payoffs in
a chosen range for games which maximally distinguish the Nash, QRE, level-1(alpha),
and PCH strategies. ML Simulated play is used to "guess" the value of
the parameter necessary for computing predictions for QRE, level-1(alpha), and PCH.

Games above a certain threshold of distinguishability are chosen as candidates for further
investigation.

'''