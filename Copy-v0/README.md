Solution to the Copy-v0 environment with a Monte Carlo 1st-visit algorithm.
My first attempt at this was with a 1 step Q-learning agent, but it failed to learn the problem within a reasonable amount of time.

I considered a "success" to be when the policy outputted by the agent would lead to 100 tapes in a row being copied perfectly.
Each episode, the optimal policy is extracted from the agent and tested against 100 episodes of the environment. At 100% accuracy the experiment ends and the problem is solved.
The agent's solution time varies from 50-2000 episodes to solve, depending on the environment seed.

The agent is based off the one described in Sutton and Barto's textbook, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) and the rlglue software and design patterns are from Brian Tanner and Adam White's [RL-Glue project](https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0).

