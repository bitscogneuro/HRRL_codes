
# CTCS-HRRL : Continuous-Time Continuous-Space Homeostatically-Regulated Reinforcement Learning
This repository contains the implementation and experimental framework for the CTCS-HRRL agent. The project investigates how biologically grounded, interoception-driven reinforcement learning can enable autonomous agents to maintain 
internal physiological regulation while interacting with dynamic environments. 

The work focuses on learning homeostatic and allostatic control laws in continuous-time settings using Hamilton-Jacobi-Bellman (HJB) based optimization.

## Purpose of the Experiments
The experiments are designed to evaluate whether a single unified control law can drive robust, biologically plausible behavior across different environmental and physiological conditions. Specifically, we test whether the agent can:
* Maintain internal homeostasis under continuous physiological depletion,

* Learn to navigate environments to acquire resources based on internal needs,

* Adapt behavior under varying environmental dynamics,

* Demonstrate allostatic regulation (continuous regulation underlying homeostasis).


## Experimental Overview

### Experiment 1 - Baseline Foraging and Regulation
**Objective:**
This experiment assessed the agent’s ability to adapt to small spatial shifts without retraining.

*Environment*:
* Two essential resources distributed in a 10 x 10 grid.
* Continuous physiological depletion.
* Discrete embodied actions (movement, consumption, sleep)

*Comparison agents*:
* Random agent
* Greedy agent
* CTCS-HRRL / HJB agent

**Experiment 1 Implementation Logic**:

Two stationary resources were placed at fixed locations — R1 at (0.5, 4.25) and R2 at (3.5, 1.5) with full availability throughout training.

*Testing environment:* A local spatial perturbation was introduced, with R1 moved to (1.0, 4.25) and R2 to (3.5,2.0). 

### Experiment 2 - Probabilistic Resource Dynamics
**Objective:**
Test the agent's ability to cope with increased scarcity and uncertainty using the same trained policy.

**Experiment 2 Implementation Logic**:

Two stationary resources were available 90% of the time during training, introducing stochastic uncertainty.

*Testing environment:* Resource availability was reduced to 80%.


### Experiment 3 - Oscillating Resources
**Objective:**
Test the agent's ability to adapt to faster temporal dynamics using the same learned model.

**Experiment 3 Implementation Logic**:

Two moving resources with full availability oscillated between two fixed positions during training. The oscillation period was set to T = 1000 steps for both resources. R1 oscillated horizontally, while R2 oscillated vertically.

*Testing environment:* The oscillation speed was increased to T = 500 for R1 and T = 750 for R2.


### Experiment Files:

Each experiment uses a common Transition Function which describes the state-transition for the agent. Thus, this file is common and required for all experiments, as it returns the learned neural model for the transition function (netf).

Each experiment's files are under the corresponding folder name. Each folder contains following sub-folders:

* _Deviation Function_: Corresponding to each experiment a new deviation function is learned as it estimates the long-term cost associated with a given state across the three different experimental situations. 
* _HJB agent_: Implements the CTCS-HRRL agent which takes decision (action selection) based on Deviation function based long-term cost of a state.
* _Random agent_: Implements a baseline agent which takes decision (action selection) randomly.
* _Greedy agent_: Implements another comparator agent which takes decision (action selection) based on the immediate drive of the next state.


## Comparison of CTCS-HRRL agent with Q-Value-Based Agent

In addition to the comparison of the CTCS-HRRL agent with Random and Greedy agent in each experiment, we also compared it with the Q-value-based agent.

To compare the CTCS-HRRL agent with a standard model-free reinforcement learning (RL) baseline, we used a Q-value-based agent. In this approach, the estimated value of an action is computed as the mean of the observed instant drives obtained when that action was previously selected.

Formally, the Q-value for any action *a* at time *t* is defined as:

$$
Q_t(a) = \frac{\sum \text{instant drive when } a \text{ taken prior to } t}
              {\text{number of times } a \text{ taken prior to } t}
$$



If the denominator is zero, a default value (e.g., zero) is assigned to *Qt(a)*. As more data are obtained, *Qt(a)* converges to the true value *q*(a), making it a sample-average estimate of the expected drive associated with action *a*.

Action selection follows a greedy strategy in which the action with the lowest estimated instant drive is selected. If multiple actions share the same minimal value, ties are broken randomly. Formally:

At := argminₐ Qt(a)  

An ε-greedy version of this rule is used, with ε = 0.05 to allow occasional random exploration.

This model-free agent updates Q-values directly from experience and does not require learning an explicit model of the environment. State transitions are obtained from ground-truth environment dynamics. This baseline therefore enables a direct comparison with the model-based CTCS-HRRL, which explicitly learns and exploits environment dynamics.

For the Q-value-based agent, we have a folder titled, "Q-Value_based Agent". It uses the Transition Function common to all experiments and selects action based on Q-value computed within the .py ().







