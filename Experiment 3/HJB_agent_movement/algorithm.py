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

        #
        # TODO: ATTENTION AU DEEP COPY, si on met directement self.agent.zeta
        # au lieu de zeta_init (avec ou sans clone), ça ne marche pas
        zeta_init = HomogeneousZeta(self.hp)
        zeta_init.agent_internal_state = self.agent.zeta.agent_internal_state  # .clone()
        self.historic_zeta = []
        self.historic_zeta2 = []
        self.historic_zeta3 = []
        self.historic_zeta4 = []

        self.historic_actions = []
        self.historic_losses = []  # will contain a list of 2d [L_f, L_J]
        self.historic_drive = []  # will contain a list of 3d instant drive, delta, discounted
        self.action_counts = np.zeros(10, dtype=int)
        self.grads = []
        self.resource_positions = []

    def update_resource_positions(self, t):
        # Updated time periods for slower movement
        T1 = 500  # Resource 1 round trip = 100 steps (50 forward, 50 back)
        T2 = 750  # Resource 2 round trip = 150 steps (75 forward, 75 back)

        # ---- Resource 1: moves along x from 0.5 to 1.5, y stays 4.25 ----
        t1 = t % T1
        if t1 < T1 // 2:
            self.hp.cst_env.resources[0].x = 0.5 + (t1 / (T1 // 2)) * (1.5 - 0.5)
        else:
            self.hp.cst_env.resources[0].x = 1.5 - ((t1 - T1 // 2) / (T1 // 2)) * (1.5 - 0.5)
        self.hp.cst_env.resources[0].y = 4.25

        # ---- Resource 2: moves along y from 1.5 to 2.5, x stays 3.5 ----
        t2 = t % T2
        if t2 < T2 // 2:
            self.hp.cst_env.resources[1].y = 1.5 + (t2 / (T2 // 2)) * (2.5 - 1.5)
        else:
            self.hp.cst_env.resources[1].y = 2.5 - ((t2 - T2 // 2) / (T2 // 2)) * (2.5 - 1.5)
        self.hp.cst_env.resources[1].x = 3.5

    def evaluate_action(self, action: int):

        self.net_f.double()
        self.net_J.double()

        _zeta_tensor = self.agent.zeta.agent_internal_state

        zeta_u = torch.cat(
            [_zeta_tensor, torch.zeros(self.hp.cst_actions.n_actions)])
        index_control = len(_zeta_tensor) + action
        zeta_u[index_control] = 1

        _zeta_tensor = _zeta_tensor.clone().detach().requires_grad_()

        self.net_J.eval()
        self.net_f.eval()

        true = HomogeneousZeta(self.hp)
        true.agent_internal_state = self.actions.df.loc[action, "new_state"](self.agent,
                                                                             self.env).agent_internal_state
        _new_zeta = true.agent_internal_state.double()

        f = self.net_f.forward(zeta_u).detach()

        f[4] = _new_zeta[4] - _zeta_tensor[4]
        f[5] = _new_zeta[5] - _zeta_tensor[5]

        print(f"Predicted f is {f}")

        new_zeta_tensor = _zeta_tensor + self.hp.cst_algo.time_step * f

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.agent_internal_state = new_zeta_tensor

        instant_reward = self.agent.drive(new_zeta)

        grad_ = torch.autograd.grad(self.net_J(_zeta_tensor), _zeta_tensor)[0]

        if (self.env.is_near_resource(self.agent.zeta.x, self.agent.zeta.y, 0)) or (
                self.env.is_near_resource(self.agent.zeta.x, self.agent.zeta.y, 1)):
            print("Agent near resource")
            grad_filtered = grad_.clone()
            grad_filtered[4:] = 0.0
        else:
            grad_filtered = grad_  # use full gradient

        print(f"grad_ {grad_}")

        future_reward = torch.dot(grad_filtered, f)
        future_reward = future_reward.detach()

        score = instant_reward + future_reward

        _zeta_tensor.requires_grad = False
        # BIZARRE
        self.agent.zeta.agent_internal_state = _zeta_tensor

        print(f"Action {action}")
        print(f"Input zeta \t {_zeta_tensor.detach().numpy()}")
        print(f"Predicted zeta\t {new_zeta.agent_internal_state.detach().numpy()}")
        print(f"True prediction\t {true.agent_internal_state.detach().numpy()}\n")
        print(f"instant drive {instant_reward}\n"
              f"future drive {future_reward}\n"
              f"total drive {score}\n")

        self.grads.append(grad_filtered.detach().numpy())

        return score.detach().numpy()

    def simulation_one_step(self, k: int):
        """Simulate one step.
        """

        self.net_f.double()
        self.net_J.double()
        self.net_J.eval()
        self.net_f.eval()

        _zeta = self.agent.zeta.agent_internal_state

        possible_actions = [cstr(self.agent, self.env) for cstr in self.actions.df.loc[:, "constraints"].tolist()]
        indexes_possible_actions = [i for i in range(
            self.hp.cst_actions.n_actions) if possible_actions[i]]
        print(f"possible_actions {possible_actions}")

        index_default_action = self.actions.df.loc[:, "name"] == self.hp.cst_actions.default_action
        action = self.actions.df.index[index_default_action][0]

        # action = np.random.choice(indexes_possible_actions)

        if np.random.random() <= self.hp.cst_algo.eps:
            action = np.random.choice(indexes_possible_actions)
        else:
            best_score = np.Inf
            for act in indexes_possible_actions:
                score = self.evaluate_action(act)
                if score < best_score:
                    best_score = score
                    action = act

        print(f"Chosen action {action}")

        zeta_u = torch.cat(
            [_zeta, torch.zeros(self.hp.cst_actions.n_actions)])
        zeta_u[len(_zeta) + action] = 1

        zeta_u = zeta_u.double()
        # print(f"{zeta_u.dtype} zeta {zeta_u.detach().numpy()}")

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.agent_internal_state = self.actions.df.loc[action, "new_state"](self.agent,
                                                                                 self.env).agent_internal_state
        _new_zeta = new_zeta.agent_internal_state.double()

        f = self.net_f.forward(zeta_u)
        f[4] = _new_zeta[4] - _zeta[4]
        f[5] = _new_zeta[5] - _zeta[5]

        predicted_new_zeta = _zeta + self.hp.cst_algo.time_step * f

        print(f"predicted new zeta {predicted_new_zeta}")

        true_delta = _new_zeta - _zeta
        pred_delta = f
        print(f"True delta for action {action} is {true_delta}")

        diff = _new_zeta - predicted_new_zeta
        print(f"difference = {diff}")

        Loss_f = torch.sum(diff * diff)

        print(f"Loss f is {Loss_f}")

        _zeta = _zeta.clone().detach().requires_grad_()

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.agent_internal_state = predicted_new_zeta

        instant_drive = self.agent.drive(new_zeta)

        grad_ = torch.autograd.grad(self.net_J(_zeta),
                                    _zeta)[0]
        grad_ = grad_.double()

        true_delta = true_delta.double()

        delta_deviation = torch.dot(grad_, f)

        discounted_deviation = - torch.log(torch.tensor(self.hp.cst_algo.gamma)) * \
                               self.net_J.forward(_zeta)

        print(f"discounted = {discounted_deviation}")
        Loss_J = torch.square(
            instant_drive + delta_deviation - discounted_deviation)

        _zeta.requires_grad = False

        print(f"Input zeta \t {_zeta.detach().numpy()}")
        # print(f"True delta \t {true_delta.detach().numpy()}")
        # print(f"Pred delta \t {f.detach().numpy()}\n")
        print(f"Output zeta\t {predicted_new_zeta.detach().numpy()}")
        print(f"True zeta\t {_new_zeta.detach().numpy()}\n")

        self.agent.zeta.agent_internal_state = predicted_new_zeta

        for index in self.hp.cst_agent.features_to_index["homeostatic"]:
            if self.agent.zeta.agent_internal_state[index] + self.agent.x_star.agent_internal_state[
                index] < self.hp.cst_agent.min_resource:
                self.agent.zeta.agent_internal_state[index] = -self.agent.x_star.agent_internal_state[
                    index] + self.hp.cst_agent.min_resource

        loss = np.array([Loss_f.detach().numpy(), Loss_J.detach().numpy()[0]])

        # loss = np.array([Loss_f.detach().numpy(), Loss_J.detach().numpy()])
        drive = np.array(
            [instant_drive.detach().item(), delta_deviation.detach().item(), discounted_deviation.detach().item()])

        # save historic
        self.historic_drive.append(drive)
        self.historic_zeta.append(_zeta.detach().numpy())
        self.historic_zeta2.append(predicted_new_zeta.detach().numpy())
        self.historic_zeta3.append(true_delta.detach().numpy())
        self.historic_zeta4.append(f.detach().numpy())
        self.historic_actions.append(action)
        self.historic_losses.append(loss)

    def simulation(self):
        start_time = time.time()  # Record start time

        for k in range(self.hp.cst_algo.N_iter):

            self.update_resource_positions(k)
            print(f"Position R1 = {self.hp.cst_env.resources[0].x, self.hp.cst_env.resources[0].y}")
            print(f"Position R2 = {self.hp.cst_env.resources[1].x, self.hp.cst_env.resources[1].y}")

            positions = np.array([self.hp.cst_env.resources[0].x, self.hp.cst_env.resources[0].y,
                                 self.hp.cst_env.resources[1].x, self.hp.cst_env.resources[1].y])

            self.resource_positions.append(positions)

            self.simulation_one_step(k)
            # print("net_f:", [pg['lr'] for pg in self.optimizer_f.param_groups])

            if ((k % self.hp.cst_algo.N_print) == 0) or (k == self.hp.cst_algo.N_iter - 1):
                print("\nIteration:", k, "/", self.hp.cst_algo.N_iter - 1)

                print("Zeta before action:", self.historic_zeta[-1])
                action = self.historic_actions[-1]
                print("Action:", action, self.actions.df.loc[action, "name"])
                print("")
            print(f"-----------------------------------------------------------------------------")

            df1 = pandas.DataFrame(self.historic_actions, columns=['Action'])
            df2 = pandas.DataFrame(self.historic_zeta,
                                   columns=['c_cr1', 'c_cr2', 'c_mf1', 'c_sf1', 'c_x', 'c_y'])
            df3 = pandas.DataFrame(self.historic_losses, columns=['loss_f', 'loss_j'])
            df4 = pandas.DataFrame(self.historic_drive, columns=['instant', 'delta', 'discounted'])
            df5 = pandas.DataFrame(self.historic_zeta2,
                                   columns=['p_cr1', 'p_cr2', 'p_mf1', 'p_sf1', 'p_x', 'p_y'])
            df6 = pandas.DataFrame(self.historic_zeta3,
                                   columns=['t1', 't2', 'tm', 'ts', 'tx', 'ty'])
            df7 = pandas.DataFrame(self.historic_zeta4,
                                   columns=['p1', 'p2', 'pm', 'ps', 'px', 'py'])
            df9 = pandas.DataFrame(self.resource_positions, columns = ['x1', 'y1', 'x2', 'y2'])
            combined_df2 = pandas.concat([df2, df1, df3, df4, df5, df6, df7, df9], axis=1)
            combined_df2.to_csv(
                f"/Users/ibrain/Downloads/Cognitive Neuroscience Work/boris gutkin/Dec 2025 Revision/Oscillation Plots/case_movement.csv")

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time

            df8 = pandas.DataFrame(self.grads, columns=['g1', 'g2', 'g3', 'g4', 'g5', 'g6'])

            grads_file = pandas.concat([df8], axis=1)
            grads_file.to_csv("grads.csv")

            print(f"Elapsed time: {elapsed_time:.4f} seconds")
