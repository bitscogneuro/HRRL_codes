import math

import pandas

from hyperparam import Hyperparam
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

TensorTorch = type(torch.Tensor().to(device))


class HomogeneousZeta:
    """Homogeneous to the state (internal + external) of the agent."""

    def __init__(self, hyperparam: Hyperparam) -> None:
        self.hp = hyperparam
        self.agent_internal_state = torch.zeros(self.hp.cst_agent.zeta_shape).double()

    def get_resource(self, res: int):
        assert res < self.hp.difficulty.n_resources
        return float(self.agent_internal_state[self.hp.cst_agent.features_to_index[f"resource_{res}"]])

    def set_resource(self, res: int, val: float):
        assert res < self.hp.difficulty.n_resources
        # self.agent_internal_state[self.hp.cst_agent.features_to_index[f"resource_{res}"]] = val
        self.agent_internal_state[self.hp.cst_agent.features_to_index[f"resource_{res}"]] = torch.tensor(val,
                                                                                                         dtype=torch.float64)

    @property
    def muscular_fatigue(self) -> float:
        return float(self.agent_internal_state[self.hp.cst_agent.features_to_index["muscular_fatigue"]])

    @muscular_fatigue.setter
    def muscular_fatigue(self, val: float) -> None:
        self.agent_internal_state[self.hp.cst_agent.features_to_index["muscular_fatigue"]] = val

    @property
    def sleep_fatigue(self) -> float:
        return float(self.agent_internal_state[self.hp.cst_agent.features_to_index["sleep_fatigue"]])

    @sleep_fatigue.setter
    def sleep_fatigue(self, val: float) -> None:
        self.agent_internal_state[self.hp.cst_agent.features_to_index["sleep_fatigue"]] = val

    @property
    def x(self) -> float:
        return float(self.agent_internal_state[self.hp.cst_agent.features_to_index["x"]])

    @x.setter
    def x(self, val: float) -> None:
        self.agent_internal_state[self.hp.cst_agent.features_to_index["x"]] = val

    @property
    def y(self) -> float:
        return float(self.agent_internal_state[self.hp.cst_agent.features_to_index["y"]])

    @y.setter
    def y(self, val: float) -> None:
        self.agent_internal_state[self.hp.cst_agent.features_to_index["y"]] = val

    @property
    def homeostatic(self) -> TensorTorch:
        return self.agent_internal_state[self.hp.cst_agent.features_to_index["homeostatic"]]

    @homeostatic.setter
    def homeostatic(self, val: TensorTorch) -> None:
        self.agent_internal_state[self.hp.cst_agent.features_to_index["homeostatic"]] = val

    @property
    def non_homeostatic(self) -> TensorTorch:
        return self.agent_internal_state[self.hp.cst_agent.features_to_index["non_homeostatic"]]

    @non_homeostatic.setter
    def non_homeostatic(self, val: TensorTorch) -> None:
        self.agent_internal_state[self.hp.cst_agent.features_to_index["non_homeostatic"]] = val


class Agent:
    def __init__(self, hyperparam: Hyperparam):
        """Initialize the Agent.
        """
        self.hp = hyperparam

        self.zeta = HomogeneousZeta(self.hp)
        # Setting initial position

        self.zeta.x = self.hp.cst_agent.default_pos_x
        self.zeta.y = self.hp.cst_agent.default_pos_y
        self.zeta.muscular_fatigue = self.hp.cst_agent.min_resource
        self.zeta.sleep_fatigue = self.hp.cst_agent.min_resource

        # self.zeta.set_resource(0, -0.905)
        # self.zeta.set_resource(1, - 1.895)
        # self.zeta.muscular_fatigue = 0.1066920993
        # self.zeta.sleep_fatigue = 0.6471827661
        # self.zeta.x = 3.4
        # self.zeta.y = 1.5

        def set_val_homeo(dic_val):
            homo_zeta = HomogeneousZeta(self.hp)
            for res in range(self.hp.difficulty.n_resources):
                homo_zeta.set_resource(res, dic_val[f"resource_{res}"])
            homo_zeta.muscular_fatigue = dic_val["muscular_fatigue"]
            homo_zeta.sleep_fatigue = dic_val["sleep_fatigue"]
            return homo_zeta

        self.x_star = set_val_homeo(self.hp.cst_agent.val_x_star)

        self.coeff_eq_diff = set_val_homeo(self.hp.cst_agent.val_coeff_eq_diff)

    def drive(self, zeta: HomogeneousZeta, epsilon: float = 0.001) -> float:
        drive_delta = torch.sum((zeta.homeostatic - self.x_star.homeostatic) ** 2)
        return drive_delta

    def dynamics(self, zeta: HomogeneousZeta, u: HomogeneousZeta) -> HomogeneousZeta:
        """
        Return the Agent's dynamics which is represented by the f function.
        """
        f = HomogeneousZeta(self.hp)
        f.homeostatic = (self.coeff_eq_diff.homeostatic + u.homeostatic) * \
                        (zeta.homeostatic + self.x_star.homeostatic)
        # print("f.homeostatic in dynamics function", f.homeostatic, "\n")

        f.non_homeostatic = u.non_homeostatic
        return f

    def euler_method(self, zeta: HomogeneousZeta, u: HomogeneousZeta) -> HomogeneousZeta:
        """Euler method for tiny time steps.
        """
        # print("zeta.homeo", zeta.agent_internal_state, u.homeostatic)
        new_zeta = HomogeneousZeta(self.hp)
        delta_zeta = self.hp.cst_algo.time_step * self.dynamics(zeta, u).agent_internal_state
        new_zeta.agent_internal_state = zeta.agent_internal_state + delta_zeta
        return new_zeta

    def integrate_multiple_steps(self,
                                 duration: float,
                                 zeta: HomogeneousZeta,
                                 control: HomogeneousZeta) -> HomogeneousZeta:
        """We integrate rigorously with an exponential over 
        long time period the differential equation.
        This function is usefull in the case of big actions, 
        such as going direclty to one of the resource.
        """
        x = zeta.homeostatic + self.x_star.homeostatic
        # print("zeta.homeostatic", zeta.homeostatic,"\nself.x_star.homeostatic", self.x_star.homeostatic)
        # print("--------------------\n")
        # print(
        #     f"\nintegrate multiple steps :  zeta.homeostatic = {zeta.homeostatic}, \nself.x_star.homeostatic = {self.x_star.homeostatic}, \nx = {x}")

        rate = self.coeff_eq_diff.homeostatic + control.homeostatic

        # print(
        #     f"\ncoeff_eq_diff.homeo = {self.coeff_eq_diff.homeostatic}, \ncontrol.homeostatic {control.homeostatic}, \nrate = {rate}")

        new_x = x * torch.exp(rate * duration)
        # print(f"\nnew x = {new_x}")

        new_zeta_homeo = new_x - self.x_star.homeostatic
        # print(f"\nnew_zeta_homeo = {new_zeta_homeo}")

        new_zeta = HomogeneousZeta(self.hp)
        new_zeta.homeostatic = new_zeta_homeo
        new_zeta.non_homeostatic = zeta.non_homeostatic.clone()
        return new_zeta

    # def integrate_multiple_steps_sleep(self,
    #                                    duration: float,
    #                                    zeta: HomogeneousZeta,
    #                                    control: HomogeneousZeta) -> HomogeneousZeta:
    #     """We integrate rigorously with an exponential over
    #     long time period the differential equation.
    #     This function is usefull in the case of big actions,
    #     such as going direclty to one of the resource.
    #     """
    #     x = zeta.homeostatic + self.x_star.homeostatic
    #     # print("integrate multiple steps", x, zeta.homeostatic, self.x_star.homeostatic)
    #     # coeff_eq_diff_homeostatic = torch.tensor([-0.0001, -0.0001, -0.05, -0.05]) # previously muscle loss =-0.008
    #     rate = self.coeff_eq_diff.homeostatic + control.homeostatic
    #     # print(rate, self.coeff_eq_diff.homeostatic, control.homeostatic)
    #     new_x = x * torch.exp(rate * duration)
    #
    #     new_zeta_homeo = new_x - self.x_star.homeostatic
    #
    #     new_zeta = HomogeneousZeta(self.hp)
    #     new_zeta.homeostatic = new_zeta_homeo
    #     new_zeta.non_homeostatic = zeta.non_homeostatic.clone()
    #     return new_zeta
