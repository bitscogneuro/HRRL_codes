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

        self.historic_actions = []
        self.historic_losses = []  # will contain a list of 2d [L_f, L_J]
        self.historic_drive = []  # will contain a list of 3d instant drive, delta, discounted
        self.action_counts = np.zeros(10, dtype=int)

    def evaluate_action(self, action: int):

        self.net_f.double()
        self.net_J.double()

        _zeta_tensor = self.agent.zeta.agent_internal_state
        zeta_u = torch.cat(
            [_zeta_tensor, torch.zeros(self.hp.cst_actions.n_actions)])
        index_control = len(_zeta_tensor) + action
        zeta_u[index_control] = 1

        for param in self.net_f.parameters():
            param.requires_grad = False
        for param in self.net_J.parameters():
            param.requires_grad = False

        _zeta_tensor.requires_grad = True

        self.net_J.eval()
        self.net_f.eval()

        f = self.net_f.forward(zeta_u)

        # Predicted next state
        # Predict next homeostatic state
        new_zeta_tensor = _zeta_tensor + self.hp.cst_algo.time_step * f

        # Now compute new x, y positions deterministically
        # Example: for "walk right", x = x + 0.1
        current_x = _zeta_tensor[4]
        current_y = _zeta_tensor[5]

        new_x, new_y = self.actions.df.loc[action, "new_state"](self.agent, self.env).agent_internal_state[4:6]

        # Combine everything into full predicted zeta
        new_zeta_tensor[4] = new_x
        new_zeta_tensor[5] = new_y

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.agent_internal_state = new_zeta_tensor

        instant_reward = self.agent.drive(new_zeta)

        grad_ = torch.autograd.grad(self.net_J(_zeta_tensor), _zeta_tensor)[0]

        future_reward = torch.dot(grad_, f)
        future_reward = future_reward.detach()

        score = instant_reward + future_reward

        _zeta_tensor.requires_grad = False
        # BIZARRE
        self.agent.zeta.agent_internal_state = _zeta_tensor

        for param in self.net_f.parameters():
            param.requires_grad = True
        for param in self.net_J.parameters():
            param.requires_grad = True

        self.net_J.train()
        self.net_f.train()

        true = HomogeneousZeta(self.hp)
        true.agent_internal_state = self.actions.df.loc[action, "new_state"](self.agent,
                                                                             self.env).agent_internal_state
        print(f"Action {action}")
        print(f"Input zeta \t {_zeta_tensor.detach().numpy()}")
        print(f"Predicted zeta\t {new_zeta.agent_internal_state.detach().numpy()}")
        print(f"True prediction\t {true.agent_internal_state.detach().numpy()}\n")
        print(f"instant drive {instant_reward}\n"
              f"future drive {future_reward}\n"
              f"total drive {score}\n")
        return score.detach().numpy()

    def simulation_one_step(self, k: int):
        """Simulate one step.
        """

        self.net_f.double()
        self.net_J.double()

        _zeta = self.agent.zeta.agent_internal_state

        possible_actions = [cstr(self.agent, self.env) for cstr in self.actions.df.loc[:, "constraints"].tolist()]
        indexes_possible_actions = [i for i in range(
            self.hp.cst_actions.n_actions) if possible_actions[i]]
        print(f"possible_actions {possible_actions}")

        index_default_action = self.actions.df.loc[:, "name"] == self.hp.cst_actions.default_action
        action = self.actions.df.index[index_default_action][0]

        action = np.random.choice(indexes_possible_actions)
        #
        # if np.random.random() <= self.hp.cst_algo.eps:
        #     action = np.random.choice(indexes_possible_actions)
        # else:
        #     best_score = np.Inf
        #     for act in indexes_possible_actions:
        #         score = self.evaluate_action(act)
        #         if score < best_score:
        #             best_score = score
        #             action = act

        print(f"Chosen action {action}")

        zeta_u = torch.cat(
            [_zeta, torch.zeros(self.hp.cst_actions.n_actions)])
        zeta_u[len(_zeta) + action] = 1

        zeta_u = zeta_u.double()
        print(f"{zeta_u.dtype} zeta {zeta_u.detach().numpy()}")

        current_x = _zeta[4]
        current_y = _zeta[5]

        new_x, new_y = self.actions.df.loc[action, "new_state"](self.agent, self.env).agent_internal_state[4:6]

        # Get predicted delta from net_f
        f = self.net_f.forward(zeta_u)
        f[4] = new_x - current_x
        f[5] = new_y - current_y

        predicted_new_zeta = _zeta + self.hp.cst_algo.time_step * f

        # True next state
        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.agent_internal_state = self.actions.df.loc[action, "new_state"](self.agent, self.env).agent_internal_state
        _new_zeta = new_zeta.agent_internal_state.double()

        # Compute difference only on first 4 dimensions (homeostatic)
        diff = (_new_zeta[:4] - predicted_new_zeta[:4])

        true_delta = _new_zeta - _zeta

        # Apply weights only to homeostatic part
        weights = torch.tensor([100, 100, 100, 100], dtype=torch.float64, device=diff.device)

        # Get action-specific loss scaling coefficient
        coeff = self.actions.df.loc[action, "coefficient_loss"]

        # print(weights * diff * diff )

        # Compute weighted loss

        epsilon = 1e-6

        sign_penalty = torch.where(
            torch.abs(true_delta) < epsilon,
            torch.abs(f),
            torch.relu(-f * true_delta)
        )

        loss_term = torch.sum(weights * diff * diff)

        sign_loss_term = 10000 * torch.sum(sign_penalty)

        Loss_f =  loss_term + sign_loss_term

        print(f"Loss is {Loss_f} = {loss_term}, {sign_loss_term}")


        # Backpropagation
        self.optimizer_f.zero_grad()
        Loss_f.backward()
        self.optimizer_f.step()

        _zeta.requires_grad = True

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.agent_internal_state = _new_zeta

        instant_drive = self.agent.drive(new_zeta)
        delta_deviation = torch.dot(torch.autograd.grad(self.net_J(_zeta),
                                                        _zeta)[0], f)

        discounted_deviation = - torch.log(torch.tensor(self.hp.cst_algo.gamma)) * \
                               self.net_J.forward(_zeta)

        deviation_fn = self.net_J.forward(_zeta)
        Loss_J = torch.square(
            instant_drive + delta_deviation - discounted_deviation)

        print(f"instant: {instant_drive} delta {delta_deviation} discounted {discounted_deviation}")

        _zeta.requires_grad = False

        # Backpropagation
        # self.optimizer_J.zero_grad()
        # Loss_J.backward()
        # self.optimizer_J.step()

        print(f"Action {action}")
        print(f"Input zeta \t {_zeta.detach().numpy()}\n")
        print(f"True delta \t {true_delta.detach().numpy()}")
        print(f"Pred delta \t {f.detach().numpy()}\n")
        print(f"Predicted zeta\t {predicted_new_zeta.detach().numpy()}")
        print(f"True prediction\t {_new_zeta.detach().numpy()}\n")

        self.agent.zeta.agent_internal_state = _new_zeta

        for index in self.hp.cst_agent.features_to_index["homeostatic"]:
            if self.agent.zeta.agent_internal_state[index] + self.agent.x_star.agent_internal_state[
                index] < self.hp.cst_agent.min_resource:
                self.agent.zeta.agent_internal_state[index] = -self.agent.x_star.agent_internal_state[
                    index] + self.hp.cst_agent.min_resource

        loss = np.array([loss_term.detach().numpy(), sign_loss_term.detach().numpy(), Loss_J.detach().numpy()[0]])

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

            self.simulation_one_step(k)
            # print("net_f:", [pg['lr'] for pg in self.optimizer_f.param_groups])

            if ((k % self.hp.cst_algo.N_print) == 0) or (k == self.hp.cst_algo.N_iter - 1):
                print("Iteration:", k, "/", self.hp.cst_algo.N_iter - 1)

                print("Zeta before action:", self.historic_zeta[-1])
                action = self.historic_actions[-1]
                print("Action:", action, self.actions.df.loc[action, "name"])
                print("")
            print(f"-----------------------------------------------------------------------------")

            if ((k % self.hp.cst_algo.N_save_weights) == 0) or (k == self.hp.cst_algo.N_iter - 1):
                torch.save(self.net_f.state_dict(), 'weights/weights_net_f')
                # torch.save(self.net_J.state_dict(), 'weights/weights_net_J')

            df1 = pandas.DataFrame(self.historic_actions, columns=['Action'])
            df2 = pandas.DataFrame(self.historic_zeta,
                                   columns=['c_cr1', 'c_cr2', 'c_mf1', 'c_sf1', 'c_x', 'c_y'])
            df3 = pandas.DataFrame(self.historic_losses, columns=['loss_term', 'sign_loss', 'loss_j'])
            df4 = pandas.DataFrame(self.historic_drive, columns=['instant', 'delta', 'discounted'])
            df5 = pandas.DataFrame(self.historic_zeta2,
                                   columns=['p_cr1', 'p_cr2', 'p_mf1', 'p_sf1', 'p_x', 'p_y'])
            df6 = pandas.DataFrame(self.historic_zeta3,
                                   columns=['t1', 't2', 'tm', 'ts', 'tx', 'ty'])
            df7 = pandas.DataFrame(self.historic_zeta4,
                                   columns=['p1', 'p2', 'pm', 'ps', 'px', 'py'])
            combined_df2 = pandas.concat([df2, df1, df3, df4, df5, df6, df7], axis=1)
            combined_df2.to_csv(
                f"data/hjb.csv")

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time

            print(f"Elapsed time: {elapsed_time:.4f} seconds")
