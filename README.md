# Unifying-algorithms-for-multi-step-reinforcement-learning
A state-dependent extension to Q(&sigma;) algorithm

We perform some numerical experiments with the Q(&sigma;) algorithm from De Asis et al.(2018), "Multi-step reinforcement learning: A unifying
algorithm" (https://arxiv.org/abs/1703.01327). The state-dependent version for the dynamics of the degree of sampling is studied. 

We let &sigma; be a function of a number of visits to a state: for less-explored states, the degree of sampling is higher than for better-explored states. To accomplish this, we can decrease &sigma;(s) by a given factor after each visit to a state s. This approach generalizes the dynamic version of Q(&sigma;) algorithm by De Asis et al.(2018) in which the degree of sampling is reduced for all states simultaneously after each episode.

The experiments considered are the 19-state Random Walk task (prediction task) and the Stochastic Windy Gridworld (control task). See De Asis et al.(2018) and Sutton and Barto (2018) (http://incompleteideas.net/book/the-book.html). Parameters are taken from De Asis et al.(2018) and Dumke 2017 (https://arxiv.org/abs/1711.01569).
