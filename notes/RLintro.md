# Notes
## June 27, 2019

A collection of notes derived from the book, *Reinforcement Learning* by Richard Sutton and Andrew Barto. Available at htpp://incompleteideas.net/book/the-book.html.

### Some terminology
```
S+ = the state space of the entire problem, including the terminal state
S = the state space of the entire problem, not including the terminal state
argmaxa(f(a)) = returns the a that yields the the largest value from f(a)
maxa(f(a)) = f(argmaxa(f(a)))
A[s] = the possible actions that can be taken in state s
S[(s,a)] = the possible states that can result from taking action a in state s
p(s',r|s,a) = the probability of arriving in state s' and receiving reward r when taking action a in state s
pi[s] = the ideal action to take in state s given policy pi
Q[s,a] or Q[s] = the value of state s or the value of taking action a in state s
gamma = the discount rate. typically 0.9 if discounted, 1 if undiscounted, 0 if agent is myopic. The larger the discount,
the more the agent is concerned with future rewards. This should be configured in respect to whether or not the agents
current decisions affect its future state(s).
epsilon = random chance of agent performance an exploration (random) movement. Typically 0.1 for epsilon-greedy action
selection
alpha = stepsize, amount to adjust value function by when learning new information. typically 0.5
```

1. # Action Selection

Most RL methods require some form of policy or action-value based action selection method.

* **Greedy Selection:** Choosing the best action
```
A = argmaxa(Q(a))
```
* **Epsilon-greedy Selection:** Simple exploration with epsilon-probability.
```
i = random value between 0 and 1
if i > epsilon:
	A = random action
else:
	A = argmaxa(Q(a))
```
* **Uppder Confidence Bound (UCB):** Takes into account the proximitiy of the estimate to being maximal and the uncertanty in the estimates.
Does not perfome well on large state spaces.
```
A(t) = a s.t. ( Q(t)(a) + c * sqrt( ln(t) / N(t)(a) ) ) is maximized
Where
	c > 0 is the degree of exploration
	N(t)(a) is the number of times that action a has been selected prior to time t.
		If N(t)(a) = 0, then a is considered to be a maximizing action.
```

2. # Performance Measures

Methods for comparing the performance of different parameters and algorithms.

* **Optimal Action %**: Requires knowledge of the workins of the environment and whether the action was optimal. Plot % over steps.
* **Average Reward:** Simply plot the average reward over steps. Good for comparing specific implementation of agent in specific implementation of environment.
* **Average Reward wrt parameter:** Plot the average reward over first n=1000 steps against input parameter(s) (epsilon, alpha, c, Q(0), etc) on a logarithmic
scale. Good for comparing learning algorithms' general effectiveness and finding the best parameter value.
* **Mean Square Error:** Plot the mean square error (averaged over n=100 runs) of the value of a single state (`error = actual - estimate`) over the number of 
episodes run before acheiving the estimate, with the episodes on a logarithmic scale. Good for Monte Carlo method, where you can form an estimate of a single 
state without forming an estimate of the others.

# 3. Algorithms

Reinforcement learning algorithms covered by the book.

* **Dynamic Programming / Value Iteration:** Updating state values by sweeping through all states. Computationally expensive, especially on large state spaces.
Requires knowledge of all state spaces.
```
Parameters:
	a small threshold theta > 0
	Initialize V[s] for all s in S+ arbitrarily, except that V[terminal] = 0

Loop:
	delta = 0
	Loop for each s in S+:
		v = V[s]
		V[s] = max([sum([probability(s', r | s, a)*(r + gamma*V[s']) for each (s',r) in potential resulting stats from action a in state s]) for actions in possible actions in s])
		delta = max(delta, v - V[s])
until delta < theta

Output a detemrinistic policy pi, s.t.
	pi[s] = a s.t. (sum([probability(s',r|s,a)*(r+gamma*V[s'] for each (s',r) in potential states from taking action a in state s]) for possible actions in state s) is maximized
```

* **On-policy Monte Carlo First Visit control:** Use an episilon-soft policy to explore, and use the generated episode
to update said policy.
```
Parameters:
	Q = [arbitrary value for s in S for a in A]
	Returns = []
	pi = an arbitrary epsilon-soft policy
1. Generate an episode using pi
2. Generate the returns for each pair (s,a) in episode
	G[t] = R[t+1] + gamma*G[t+1]
3. For each pair (s,a) in the episode:
	G = the return that follows the first occurrence of (s,a)
	Append G to Returns[(s,a)]
	Q[(s,a)] = average(Returns[(s,a)])
4. For each s in the episode
	A* = argmaxa(Q[(s,a)])
	For each a in A[s]:
		if a == A*:
			pi[(a|s)] = 1 - epsilon + epsilon/len(A[s])
		else:
			pi[(a|s)] = epsilon/len(A[s])

```

* **SARSA:** On-policy TD control. Learns a sub-optimal policy but has good online performance due to application of
optimal value-function
```
Parameters:
	0 < alpha <= 1
	epsilon > 0
	Q = [arbitrary value for s in S+ for a in A[s]]
	Q[(terminal, any)] = 0

For each episode
	Init S
	Choose A from S using policy derived from Q (e.g. epsilon-greedy)
	Loop for each step S of episode, until S is terminal:
		Take action A, observe R, S'
		Choose A' from S' using policy derived from Q (e.g. epsilon-greedy)
		Q[(S,A)] = Q[(S,A)] + alpha*(R + gamma*Q[(S', A')] - Q[(S,A)])
		S = S'
		A = A'
```

* **Q-Learning:** Off-policy TD control. Learns the optimal policy but has poor online performance due to application
of epsilon-greedy policy.
```
Note:
	maxaQ[(S,a)] = a s.t. a in A[S] and Q[(S,a)] yields the highest value
Parameters:
	0 < alpha <= 1
	epsilon > 0
	Q = [arbitrary value for s in S+ for a in A[s]]
	Q[(terminal, any)] = 0

For each episode
	Init S
	Loop for each step S of episode, until S is terminal:
		Choose A from S using policy derived from Q (e.g. epsilon-greedy)
		Take action A, observe R, S'
		Q[(S,A)] = Q[(S,A)] + alpha*(R + gamma*maxaQ[(S', a)] - Q[(S,A)])
		S = S'
```

