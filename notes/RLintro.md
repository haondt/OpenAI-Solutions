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

3. # Algorithms

Reinforcement learning algorithms covered by the book.

* **Dynamic Programming / Value Iteration:** Updating state values by sweeping through all states. Computationally expensive, especially on large state spaces.
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

* **Off-policy Monte Carlo control:** Using a soft policy to explore, while estimating the optimal policy. Does not bootstrap, i.e. doesn't use estimates of previous states to 
estimate current state value.
```
Parameters:
	Q = [arbitrary value for s in S for a in A]
	C = [abritrary value for s in S for a in A]
	pi[s] = argmaxa(Q[(s,a)])

For each episode E:
	b = any soft policy
	Generate an episode using b (b = S[0], A[0], R[1], ...,S[T-1],A[T-1],R[T])
	G = 0
	W = 1
	
	For each step of episode, t in [T-1, T-2, ..., 0]:
		G = gamma*G+R[t+1]
		C[(S[t], A[t])] += W
		Q[(S[t], A[t])] += (W/C[(S[t], A[t])])*(G - Q[(S[t], A[t])])
		pi[S[t]] = argmaxa(Q(S[t],a)) # with ties broken consistently
		if A[t] != pi[S[t]]:
			break
		W *= 1/b[A[t] given S[t]]
```

* **n-Step Sarsa:** Using the rewards from the previous n steps to update the value of the current state. Uses an epsilon-greedy policy to estimate Q ~ Q\*.
```
# Initialization
Q = {(s,a):arbitrary value for s in S for a in A}
pi = epsilon greedy w.r.t. Q, or a fixed given policy

# stepsize
0 < alpha <= 1
epsilon > 0
n > 0, type(n) == int

