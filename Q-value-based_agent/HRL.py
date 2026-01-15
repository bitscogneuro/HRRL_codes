"""Homeostatic reinforcement learning.

Described in the paper:
Continuous Homeostatic Reinforcement Learning
for Self-Regulated Autonomous Agents.

Authors: Hugo Laurençon, Charbel-Raphaël Ségerie,
Johann Lussange, Boris S. Gutkin.
"""
import os

import numba
from numba import jit

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

import sys
import argparse
import torch

# Set print options to suppress scientific notation


from typing import Literal

import torch

from utils import set_all_seeds
from hyperparam import Hyperparam
from environment import Environment
from agent import Agent
from actions import Actions
from nets import Net_J, Net_f
from plots import Plots
from algorithm import Algorithm
from algorithm_test import Testing_Class

import os

print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def parseArgs(argv):
    parser = argparse.ArgumentParser(description='Run the simulation of the agent.')

    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the reproductibility of experiments.')

    parser.add_argument('--difficulty', type=str, default="EASY",
                        help='Difficulty of the environment. EASY or MEDIUM.')

    parser.add_argument('--num_iter', type=int, default=5000,
                        help='Number of iterations to run the simulation for.')

    parser.add_argument('--freq_print', type=int, default=1,
                        help='Print details of the current action every freq_print steps.')

    parser.add_argument('--freq_plot', type=int, default=5000,
                        help='Make one plot every freq_plot steps.')

    parser.add_argument('--freq_save_weights', type=int, default=5000,
                        help='Save neural networks weights every freq_save_weights steps.')

    return parser.parse_args(argv)


def main(argv):
    args = parseArgs(argv)

    set_all_seeds(args.seed)

    hyperparam = Hyperparam(level=args.difficulty)
    hyperparam.cst_algo.N_iter = args.num_iter
    hyperparam.cst_algo.N_print = args.freq_print
    hyperparam.cst_algo.cycle_plot = args.freq_plot
    hyperparam.cst_algo.N_save_weights = args.freq_save_weights

    env = Environment(hyperparam)
    agent = Agent(hyperparam)
    actions = Actions(hyperparam)
    net_J = Net_J(hyperparam)
    net_f = Net_f(hyperparam)
    plots = Plots(hyperparam)

    algo = Algorithm(hyperparam, env, agent, actions, net_J, net_f, plots)
    algo.simulation()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
