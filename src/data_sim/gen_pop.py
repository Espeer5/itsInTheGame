"""This module contains the code neededto generate a population of small neural 
networks trained toplay 2x2 games. This population may then be used to simulate 
2x2 game data and estimate reasonable parameter values for the free parameters
of the game theory models."""

import torch
import copy

ITERS = 500

class GameAgent(torch.nn.Module):
    """A neural network which learns to be either the row or column player 
    for a given game matrix"""

    def __init__(self, input_dim, output_dim):
        """Create a neural network to learn a 2x2"""
        super(GameAgent, self).__init__()
        self.hidden_layers = 8
        print("Hidden Layers: {}".format(self.hidden_layers))
        self.fcl_s = torch.nn.ModuleList()
        self.fcl_s.append(torch.nn.Linear(input_dim, 8))
        for _ in range(self.hidden_layers):
            self.fcl_s.append(torch.nn.Linear(8,8))
        self.fcl_s.append(torch.nn.Linear(8, output_dim))

    def forward(self, x):
        """Forward pass of the neural network on an input tensor x"""
        m = torch.nn.Sigmoid()
        for layer in self.fcl_s:
            x = m(layer(x))
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
    Rp_0_0 = row_p.mul(col_p).mul(game[0, 0][0])
    Rp_0_1 = row_p.mul((col_p.mul(-1)).add(1)).mul(game[0, 1][0])
    Rp_1_0 = ((row_p.mul(-1)).add(1)).mul(col_p).mul(game[1, 0][0]) 
    Rp_1_1 = ((row_p.mul(-1)).add(1)).mul((col_p.mul(-1)).add(1)).mul(game[1, 1][0])
    return Rp_0_0 + Rp_0_1 + Rp_1_0 + Rp_1_1


def get_c_exp(row_p, col_p, game):
    """Computes expectations scores for the column agent playing against the row
    agent with their specified probabilities and the given game matrix"""
    Cp_0_0 = row_p.mul(col_p).mul(game[0, 0][1])
    Cp_0_1 = row_p.mul((col_p.mul(-1)).add(1)).mul(game[0, 1][1])
    Cp_1_0 = ((row_p.mul(-1)).add(1)).mul(col_p).mul(game[1, 0][1]) 
    Cp_1_1 = ((row_p.mul(-1)).add(1)).mul((col_p.mul(-1)).add(1)).mul(game[1, 1][1])
    return Cp_0_0 + Cp_0_1 + Cp_1_0 + Cp_1_1


def train(g):
    """Trains a row network and column network by playing them against each 
    other in a 2x2 game for ITER rounds"""

    # Create a network for each of the row and column players
    RowAgent = GameAgent(8,1)
    ColAgent = GameAgent(8,1)
    
    # Create an optimizer for each network
    RowOpt = torch.optim.Adam(RowAgent.parameters(), lr=0.001)
    ColOpt = torch.optim.Adam(ColAgent.parameters(), lr=0.001)

    # Run the training loop
    for i in range(ITERS):

        ins_1 = game_to_input(g)
        ins_2 = copy.deepcopy(ins_1)
        row_p = RowAgent(ins_1)
        col_p = ColAgent(ins_2)

        # Backpropagate the loss in an alternating fashion
        if i % 2 == 0:
            row_l = -get_r_exp(row_p, col_p, g)
            RowOpt.zero_grad()
            row_l.backward()
            RowOpt.step()
        else:
            col_l = -get_c_exp(row_p, col_p, g)
            ColOpt.zero_grad()
            col_l.backward()
            ColOpt.step()
    
    return RowAgent, ColAgent


def sim_data(game):
    """Return simulated game play data for a given game matrix by training 
    neural networks, and extracting and returning their play frequency"""
    r, c = train(game)
    ins = game_to_input(game)
    row_p = r(ins)
    col_p = c(ins)
    return row_p.item(), col_p.item()
    