# Unifying-algorithms-for-multi-step-reinforcement-learning
A state-dependent version of the Q(&sigma;) algorithm

We perform some numerical experiments with the Q(&sigma;) algorithm from De Asis et al.(2018), "Multi-step reinforcement learning: A unifying
algorithm" (https://arxiv.org/abs/1703.01327). The state-dependent version for the dynamics of the degree of sampling &sigma; is studied. 

We let &sigma; be a function of a number of visits to a state: for less-explored states, the degree of sampling is higher than for better-explored states. To accomplish this, we can decrease &sigma;(s) by a given factor after each visit to a state s. This approach generalizes the dynamic version of Q(&sigma;) algorithm by De Asis et al.(2018) in which the degree of sampling is reduced for all states simultaneously after each episode.

The experiments considered are the 19-state Random Walk task (prediction task) and the Stochastic Windy Gridworld (control task). See De Asis et al.(2018) and Sutton and Barto (2018) (http://incompleteideas.net/book/the-book.html). Parameters are taken from De Asis et al.(2018) and Dumke 2017 (https://arxiv.org/abs/1711.01569).

As measured by the RMS error in the value function estimation task and the number of steps to cross the gridworld, the proposed state-dependent version of the algorithm (red line) outperforms the dynamic algorithm of De Asis et al.(2018) with episode-dependent &sigma; (blue line) for most of the modelling settings.

![Figure 1](https://github.com/NikolayGudkov/Advances-in-unifying-algorithms-for-multi-step-reinforcement-learning/blob/main/Random_Walk.png)

![Figure 2](https://github.com/NikolayGudkov/Advances-in-unifying-algorithms-for-multi-step-reinforcement-learning/blob/main/Stochastic_Windy_Gridworld.png)

We also notice that the standard errors for the estimates obtained in the two experiments using the state-dependent algorithm (0.021 and 5.91) are lower than those acquired using the episode-dependent algorithm (0.036 and 9.08).
