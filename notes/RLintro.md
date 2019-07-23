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
alpha = stepsize, amount to adjust value function by when learning new information. typically 0.1 or 0.5 for tabular methods,
0.0001/(# tilings) for approximation methods.
```

# Dimensions of an algorithm
* Width: learning from a single posibility (TD methods) to learning from all posibilities (Dynamic programming)
* Depth: updating after an entire episode (MC) to updating after a single step (TD-0)
* on/off-policy: using the same algorithm for control and learning or controlling with one to learn another
* Return: Episodic vs continuing (MC vs Sarsa for example) and un/discounted
* action values vs state values vs afterstate values
* Model vs only environment
* Location of updates (update most recent? update last? update randomly from model?)

---

# Tabular Methods: calculating the perfect action for every single existing state-action pair individually

---

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
to update said policy. Updates policy at the end of an episode.
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
optimal value-function. Updates policy on each timestep.
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
of epsilon-greedy policy. Updates policy on each timestep.
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

A greedy policy generated from Q can then approximate the optimal policy, pi*
```

* **Expected Sarsa:** Q-Learning, but taking into account how likely each action is under the current policy. Eliminates
variance of randomly selecting `a` from `maxaQ[(S', a)]`, leading to better performance than Q-Learning and Sarsa
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
		Q[(S,A)] = Q[(S,A)] + alpha*(R + gamma*sum([pi(a|S')*Q[(S',a)] for a in A[s]]) - Q[(S,A)])
		S = S'

A greedy policy generated from Q can then approximate the optimal policy, pi*
```

* **N-step methods:** There are n-step methods that update the policy every n-steps. The can be implemented using 
eligibility traces, so we'll come back to that.

* **Tabular Dyna-Q:** Q-Learning but with extra training on a model for extra training between timesteps. The model is
created from environment experience. Example numbers of training steps (`n`) include 0, 5 and 50. On the assumption that
the model can be accessed much faster than the environment, a singly clumsy exploratory episode can yield a clean 
optimized route through the explored path. In some cases, a changing environment is quickly incorporated into an incorrect
model as the agent attempts to exploit the environment via exploitations learned from the model and updates the model
when its expectations are not met.
```
Parameters
	Q = [arbitrary value for s in S for a in A[s]]
	Model = [arbitrary value for s in S for a in A[s]]

Loop forever:
	S = current (non-terminal state)
	A = epsilon-greed(S,Q)
	Take action A, observe reward R and state S'
	Q[(S,A)] = Q[(S,A)] + alpha*(R + gamma*maxaQ[(S',a)] - Q[(S,A)])
	Model[(S,A)] = R, S' # assuming deterministic environment
	Loop n times:
		S = random previously encountered state
		A = random action previously taken in S
		R, S' = Model[(S,A)]
		Q[(S,A)] = Q[(S,A)] + alpha*(R + gamma*maxaQ[(S',a)] - Q[(S,A)])
```

---

# Approximate Solution Methods: Grouping states together and selecting actions based on the group

---

# Coarse coding

**Binary features:**

Imagine an environment where the states are points in 2d space. If we cover the space with overlapping circles,
we can describe the state by which circles it is inside of. Each circle is a `feature` and can have the value 1 or 0,
representing whether or not the state is inside of it. We can have much fewer circles than possible points on the space.

**The weight vector:**

Each circle has a weight as well. All the weights combined form the vector **w**. When we train at a state, the weights of 
all the circles enveloping the state are affected. Imagine we have a second point/state, somewhere else on the space. The
more circles that are touching both points, the more learning on one state affects the value function of the other state.

# Tile coding

Each feature becomes a tiling, a single grid with multiple partitions that covers the entire state space. Each partition is
considered a *tile*, and the entire feature is considered a *tiling*. Each feature is no long binary (in or out), but can 
contain a value indicating which tile (cell) of the feature the state resides in. We can add more features, and offset them
(diagonally) by a fraction of the tile width. Any state will then exists in 1 tile of each tiling. We can then define the
feature vector, **x**(s), with one component for each feature, indicating which tile of feature/tiling the state is in.

The step size (alpha) can be chosen as 1/((# tilings)*(how far the estimate should move inversly towards the target). e.g.
1/n = estimate moves directly to target, 1/10n = estimate moves 1 tenth of the way to the target.

Asymmetrically offset (circlish) tiling is preferred over uniformly offset (Squarish, e.g. each tile is up and to the right
of the previous) tiling because the gives a more homogenous extrapolation of the state. I.e. a change in the value of the 
state ripples around to the nearby states in a circle, rather than a straight line, which ignores nearby states perpendicular
to the line and affects far away states that are on the line.

We can also hash a tile into a set of just a few mini-tiles - areas that cover a small state space. This reduces the memory
requirment for the tile and the tiling to which it is part of, as though the task may exists in a large state space, the task
itself may only access a small part.

p.222 (9.6 Selecting Step-Size Parameters Manually)
