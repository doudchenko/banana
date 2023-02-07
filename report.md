# Report on the Banana Navigation problem

## TL;DR
We use a Deep Q-Network (DQN) that is largely based on the [solution](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution) to one of the problems discussed in the Udacity deep RL nanodegree program.[^1]

[^1]: As a matter of fact, a working solution can be obtained using almost identical code with very minor modifications and no changes to the hyperparameters.

The snippet below shows the performance of a trained agent:\
<img src="trained_agent.gif" width="600"/>

## Details of the implementation
### Deep Q-Network
We use a [dueling DQN approach](https://arxiv.org/abs/1511.06581) with two hidden layers having 64 units each, one layer corresponding to state levels and one to advantage values. The figure below shows the network architecture:\
<img src="dqn_graph.png" width="400"/>

### Prioritized experience replay
We also use [prioritized experience replay](https://arxiv.org/abs/1511.05952) to upsample the experience tuples that lead to the largest—in absolute value—temporal difference (TD) errors, $\delta_i$. The exact approach we use is the following:
1. Given the TD errors, $\delta_i$, compute the sampling probabilities:
$$P(i) = \frac{p_i^a}{\sum_j p_j^a},$$
where $p_i = \|\delta_i\| + e$ and $a$ and $e$ are hyperparameters.
2. The update rule for the network weights, $w$, is modified as:
$$\Delta w = \alpha\left(\frac{1}{n}\cdot\frac{1}{P(i)}\right)^b\delta_i\nabla_w\hat{q}(S_i,A_i,w),$$
where $n$ is the batch size and $b$ is a parameter which is increased gradually from $b_0$ to $b_1 by being divided by $r_b<1$ each episode.

### Double DQN
We implement a [double DQN](https://arxiv.org/abs/1509.06461) so that the TD errors are computed as:
$$\delta_i = R + \gamma\hat{q}\left(S', \arg\max_\nolimits{A'}\\,\hat{q}(S',A',\tilde{w}),w\right) - \hat{q}(S,A,\tilde{w}),$$
where $\gamma$ is the discount factor, $w$ is the "target" set of weights and $\tilde{w}$ is the "local" set of weights.

## Hyperparameters
There are a number of hyperparameters used in this implementation:
| Parameter       | Value       | Description                                                         |
| --------------- | ----------- | ------------------------------------------------------------------- |
| $\tau$          | $0.001$     | Parameter for the soft weight update rule.                          |
| $\alpha$        | $0.0005$    | Learning rate.                                                      |
| $e$             | $0.1$       | Value added to $\delta_i$'s.                                        |
| $a$             | $0.5$       | Power factor for sampling probabilities.                            |
| $b_0$           | $0.1$       | Starting value for the power parameter used for importance weights. |
| $b_1$           | $1.0$       | Ending value for the power parameter used for importance weights.   |
| $r_b$           | $0.99$      | Multiplicative rate used for increasing $b$.                        |
| $\varepsilon_0$ | $1.0$       | Starting value for the greedy-policy probability parameter.         |
| $\varepsilon_1$ | $0.01$      | Ending value for the greedy-policy probability parameter.           |
| $r_\varepsilon$ | $0.995$     | Multiplicative rate used for decreasing $\varepsilon$.              |

## Results
The agent is trained (achieves the average score above `13` over 100 consecutive episodes).
