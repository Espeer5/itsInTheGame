"""This module contains the code neededto generate a population of small neural 
networks trained toplay 2x2 games. This population may then be used to simulate 
2x2 game data and estimate reasonable parameter values for the free parameters
of the game theory models."""

import torch
import game
import random as rand
import numpy as np

ITERS = 100

class GameAgent(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GameAgent, self).__init__()
        self.hidden_layers = rand.randint(1, 5)
        self.fcl_s = torch.nn.ModuleList()
        self.fcl_s.append(torch.nn.Linear(input_dim, 8))
        for _ in range(self.hidden_layers):
            self.fcl_s.append(torch.nn.Linear(8,8))
        self.fcl_s.append(torch.nn.Linear(8, output_dim))

    def forward(self, x):
        for layer in self.fcl_s:
            x = layer(x)
        return x
    

def game_to_input(game):
    """Convert a game object to a tensor for input to the neural network"""
    collect = []
    for i in range(2):
        for j in range(2):
            for t in range(2):
                collect.append(game[i, j][t])
    return torch.tensor(collect, dtype=torch.float32)


def get_r_exp(row_p, col_p, game):
    """Computes expectations scores for the row agent playing against the column
    agent with their specified probabilities and the given game matrix"""
    Rp_0_0 = row_p * col_p * game[0, 0][0]
    Rp_0_1 = row_p * (1 - col_p) * game[0, 1][0]
    Rp_1_0 = (1 - row_p) * col_p * game[1, 0][0]
    Rp_1_1 = (1 - row_p) * (1 - col_p) * game[1, 1][0]
    return Rp_0_0 + Rp_0_1 + Rp_1_0 + Rp_1_1


def get_c_exp(row_p, col_p, game):
    """Computes expectations scores for the column agent playing against the row
    agent with their specified probabilities and the given game matrix"""
    Cp_0_0 = row_p * col_p * game[0, 0][1]
    Cp_0_1 = row_p * (1 - col_p) * game[0, 1][1]
    Cp_1_0 = (1 - row_p) * col_p * game[1, 0][1]
    Cp_1_1 = (1 - row_p) * (1 - col_p) * game[1, 1][1]
    return Cp_0_0 + Cp_0_1 + Cp_1_0 + Cp_1_1


def train():

    # Create a network for each of the row and column players
    RowAgent = GameAgent(8,1)
    ColAgent = GameAgent(8,1)
    
    # Create an optimizer for each network
    RowOpt = torch.optim.Adam(RowAgent.parameters(), lr=0.001)
    ColOpt = torch.optim.Adam(ColAgent.parameters(), lr=0.001)

    # Run the training loop
    for _ in range(ITERS):

        row = np.array([[rand.randint(0, 10) for _ in range(2)] for _ in range(2)])
        col = np.array([[rand.randint(0, 10) for _ in range(2)] for _ in range(2)])
        g = game.GameBoard(row, col)

        ins = game_to_input(g)
        row_p = RowAgent(ins)
        print(row_p)
        col_p = ColAgent(ins)

        # Compute the loss
        row_l = -get_r_exp(row_p, col_p, g)
        col_l = -get_c_exp(row_p, col_p, g)

        # Backpropagate the loss
        RowOpt.zero_grad()
        ColOpt.zero_grad()
        row_l.backward(retain_graph=True)
        col_l.backward()
        RowOpt.step()
        ColOpt.step()

        print(g)
        print("p={}".format(row_p))
        print("q={}".format(col_p))
    
    return RowAgent, ColAgent


if __name__ == "__main__":
    r, c = train()
    print(r)
    print(c)




