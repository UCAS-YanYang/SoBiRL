import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn

from RanMDP import RandMDP

torch.set_default_dtype(torch.float64)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="RanMDP",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=6e-3,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=3,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="M_SoBiRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")


    # Algorithm specific arguments
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.5,
        help="the discount factor gamma")
    parser.add_argument("--state-space", type=float, default=10,
        help="state space dimension")
    parser.add_argument("--action-space", type=float, default=5,
        help="action space dimension")

    # hyperparameters
    parser.add_argument("--eps-tol", type=float, default=1e-5,
        help="control the tolerance of INGAD")
    parser.add_argument("--alpha", type=float, default=0.1,
        help="quadratic coefficient")
    parser.add_argument("--tau", type=float, default=0.01,
        help="entropy coefficient")
    
    args = parser.parse_args()

    return args





if __name__ == "__main__":
    args = parse_args()
    env = RandMDP(args.state_space, args.action_space, args)
    env.M_SoBiRL()


