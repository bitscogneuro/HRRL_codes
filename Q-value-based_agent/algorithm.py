import time

import pandas

from hyperparam import Hyperparam
from environment import Environment
from agent import Agent, HomogeneousZeta
from actions import Actions
from nets import Net_J, Net_f
from plots import Plots

from typing import List

import numpy as np
import torch

torch.set_printoptions(precision=8, linewidth=1000, sci_mode=False)

# Set print options to suppress scientific notation
# Set NumPy print options
np.set_printoptions(suppress=True, precision=10, linewidth=1000)


class Algorithm:
    def __init__(self, hyperparam: Hyperparam, env: Environment, agent: Agent,
                 actions: Actions, net_J: Net_J, net_f: Net_f, plots: Plots):

        # CLASSES #########################################
        self.hp = hyperparam
        self.env = env
        self.agent = agent
        self.actions = actions
        self.net_J = net_J
        self.net_f = net_f
        self.plots = plots

        # UTILS ############################################
        self.optimizer_J = torch.optim.Adam(
            self.net_J.parameters(), lr=self.hp.cst_algo.learning_rate_J)
        self.optimizer_f = torch.optim.Adam(
            self.net_f.parameters(), lr=self.hp.cst_algo.learning_rate_f)

        # TODO: ATTENTION AU DEEP COPY, si on met directement self.agent.zeta
        # au lieu de zeta_init (avec ou sans clone), ça ne marche pas
        zeta_init = HomogeneousZeta(self.hp)
        zeta_init.agent_internal_state = self.agent.zeta.agent_internal_state  # .clone()
        self.historic_zeta = []
        self.historic_zeta2 = []
        self.historic_zeta3 = []
        self.historic_zeta4 = []
        self.q_scores = [0.0] * 10
        self.historic_actions = []
        self.historic_losses = []  # will contain a list of 2d [L_f, L_J]
        self.historic_drive = []  # will contain a list of 3d instant drive, delta, discounted
        self.action_counts = np.zeros(10, dtype=int)

    def evaluate_action(self, indexes_of_posible_actions: list[int]):

        best_score = np.inf
        chosen_action = indexes_of_posible_actions[0]
        q_value_of_action = [0.0] * len(indexes_of_posible_actions)

        for index, action in enumerate(indexes_of_posible_actions):

            rewards_sum_upto_this_time = self.q_scores[action]
            number_of_occurences_of_action = 0

            if len(self.historic_actions) > 0:
                number_of_occurences_of_action = self.historic_actions.count(action)

            if number_of_occurences_of_action == 0:
                q_value_of_action[index] = 0
            else:
                print("\n Sum of rewards", rewards_sum_upto_this_time,
                      "number of occurences for action", action, "is ", number_of_occurences_of_action)
                q_value_of_action[index] = rewards_sum_upto_this_time / number_of_occurences_of_action

            print("\n Q-value of action ", action, "is ", q_value_of_action[index])

        # Find the minimum value
        min_value = min(q_value_of_action)
        print("\n minimum Q-value", min_value)

        # Find all indices of the minimum value
        min_indices = [i for i, x in enumerate(q_value_of_action) if x == min_value]
        print("\n min indices", min_indices)

        if len(min_indices) > 1:
            rand_index = np.random.choice(min_indices)
            chosen_action = indexes_of_posible_actions[rand_index]
            print("\n randomly chosen index", rand_index)
        else:
            chosen_action = indexes_of_posible_actions[min_indices[0]]

        print("\n chosen action", chosen_action)

        return chosen_action

    def simulation_one_step(self, k: int):
        """Simulate one step.
        """

        self.net_f.double()
        self.net_J.double()
        self.net_f.eval()

        _zeta = self.agent.zeta.agent_internal_state

        possible_actions = [cstr(self.agent, self.env) for cstr in self.actions.df.loc[:, "constraints"].tolist()]
        indexes_possible_actions = [i for i in range(
            self.hp.cst_actions.n_actions) if possible_actions[i]]
        print(f"possible_actions {possible_actions}")

        index_default_action = self.actions.df.loc[:, "name"] == self.hp.cst_actions.default_action
        action = self.actions.df.index[index_default_action][0]

        # action = np.random.choice(indexes_possible_actions)
        #

        initial_epsilon = 0.3
        final_epsilon = 0.05
        decay_steps = 5000

        epsilon = max(final_epsilon, initial_epsilon - (k / decay_steps) * (initial_epsilon - final_epsilon))
        print(f"epsilon {epsilon}")

        if len(indexes_possible_actions) > 0:
            if np.random.random() <= self.hp.cst_algo.eps:
                action = np.random.choice(indexes_possible_actions)

            else:
                action = self.evaluate_action(indexes_possible_actions)

        print(f"Chosen action {action}")

        # True next state
        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.agent_internal_state = self.actions.df.loc[action, "new_state"](self.agent,
                                                                                 self.env).agent_internal_state
        _new_zeta = new_zeta.agent_internal_state.double()

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.agent_internal_state = _new_zeta

        instant_drive = self.agent.drive(new_zeta)
        self.q_scores[action] = self.q_scores[action] + instant_drive

        _zeta.requires_grad = False

        print(f"Action {action}")
        print(f"Input zeta \t {_zeta.detach().numpy()}\n")
        print(f"True prediction\t {_new_zeta.detach().numpy()}\n")

        self.agent.zeta.agent_internal_state = _new_zeta

        for index in self.hp.cst_agent.features_to_index["homeostatic"]:
            if self.agent.zeta.agent_internal_state[index] + self.agent.x_star.agent_internal_state[
                index] < self.hp.cst_agent.min_resource:
                self.agent.zeta.agent_internal_state[index] = -self.agent.x_star.agent_internal_state[
                    index] + self.hp.cst_agent.min_resource

        # loss = np.array([Loss_f.detach().numpy(), Loss_J.detach().numpy()])
        drive = np.array([instant_drive.detach().item()])
        self.historic_zeta.append(_zeta.detach().numpy())
        self.historic_actions.append(action)

        # save historic
        self.historic_drive.append(drive)

    def simulation(self):
        start_time = time.time()  # Record start time

        for k in range(self.hp.cst_algo.N_iter):

            self.simulation_one_step(k)
            # print("net_f:", [pg['lr'] for pg in self.optimizer_f.param_groups])

            if ((k % self.hp.cst_algo.N_print) == 0) or (k == self.hp.cst_algo.N_iter - 1):
                print("Iteration:", k, "/", self.hp.cst_algo.N_iter - 1)


            df1 = pandas.DataFrame(self.historic_actions, columns=['Action'])
            df2 = pandas.DataFrame(self.historic_zeta,
                                   columns=['c_cr1', 'c_cr2', 'c_mf1', 'c_sf1', 'c_x', 'c_y'])
            df3 = pandas.DataFrame(self.historic_losses, columns=['loss_f', 'loss_j'])
            df4 = pandas.DataFrame(self.historic_drive, columns=['instant'])
            df5 = pandas.DataFrame(self.historic_zeta2,
                                   columns=['p_cr1', 'p_cr2', 'p_mf1', 'p_sf1', 'p_x', 'p_y'])
            df6 = pandas.DataFrame(self.historic_zeta3,
                                   columns=['t1', 't2', 'tm', 'ts', 'tx', 'ty'])
            df7 = pandas.DataFrame(self.historic_zeta4,
                                   columns=['p1', 'p2', 'pm', 'ps', 'px', 'py'])
            combined_df2 = pandas.concat([df2, df1, df3, df4, df5, df6, df7], axis=1)
            combined_df2.to_csv(
                f"data/qvalue.csv")

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time

            print(f"Elapsed time: {elapsed_time:.4f} seconds")
