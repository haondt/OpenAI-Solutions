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
alpha = stepsize, amount to adjust value function by when learning new information.
A constant stepsize (e.g 0.1) can track a non-stationary problem, as the agent
continues to value the every state equally. A changing step size (eg 1/n) will
allow the state values to converge. alpha is typically 0.1 or 0.5 for tabular methods,
0.0001/(# tilings) for approximation methods.
T = the terminal (last) time step
t = the time step
τ = the time step being updated in n-step cases
lambda = trace-decay parameter (e.g. 0.9)
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

The value of the state is then represented by v\_hat(s,**w**) ~ v\_pi(s). **w**
has one component for each feature, and there are typically much fewer features
than states (think float-based states) and changing one weight affects many
states. Similarly we also get q\_hat(s,a,**w**) ~ q\*(s,a).

# Tile coding

We can cover the state spaces with tilings, a grid essentially. Each tiling has
one tile for each feature (e.g. an 8x8 grid is a tiling for 64 features). Each
feature has a receptive field for each tiling. All features have the same number
of fields, corresponding with the number of tilings. Any state in the state
space will overlap with with 1 tile (feature) in each of the tilings (fields).
It can be different features for each field, depending on how the tiles overlap.
The feature vector **x**(s) has one component for each tile in each tiling. Thus
the length of **x**(s) is the height of the tilings x the width of the tilings x
the number of tilings, or the number of features x the number of fields per
feature and the number of active features will always be the number of fields
(one active feature per field). Each component is a 1 or 0 indicating if that
particular feature/field combo is activated by the state.

The step size (alpha) can be chosen as 1/((# tilings)\*(how far the estimate should move inversly towards the target). e.g.
1/n = estimate moves directly to target, 1/10n = estimate moves 1 tenth of the way to the target.

Asymmetrically offset (circlish) tiling is preferred over uniformly offset (Squarish, e.g. each tile is up and to the right
of the previous) tiling because the gives a more homogenous extrapolation of the state. I.e. a change in the value of the 
state ripples around to the nearby states in a circle, rather than a straight line, which ignores nearby states perpendicular
to the line and affects far away states that are on the line.

We can also hash a tile into a set of just a few mini-tiles - areas that cover a small state space. This reduces the memory
requirment for the tile and the tiling to which it is part of, as though the task may exists in a large state space, the task
itself may only access a small part.

# Step Size

* Good rule of thumb:

alpha = (T x E[**x**^transpose **x**])^-1

# Semi-gradient descent

* Gradient-descent: representing the state space as a weight vector (**w**) and
adjusting it after each example by a small amount in the direciton that would
most reduce the error, that is, the direction in which the error descends most
rapidly.

* Bootstrapping methods take into account the effect of changing the weight
vector **w** on the estimate, but ignore its effect on the target. They
include only part of the gradient are therefore semi-gradient.

# Average reward

* Toss out episodic vs discounted (continuing) tasks, bring in average reward
* `r(pi)` = average reward or average rate of reward

# Differential return

* Return is definned by difference between reward and average reward
```
G[t] = R[t+1] - r(pi) + R[t+2] - r(pi) + R[t+3] - r[pi] + ...
```
* value functions (v, q) and td errors are also defined using said difference

# Differential semi-gradient sarsa(0) with average reward
```
Parameters 
	# num features
	d = the number of features and size of w
	# differentiable action-value function
	Q = [arbitrary value for s in S for A in A for i in d]
	# step sizes
	alpha > 0
	beta > 0
	# weights
	w = [0 for i in d]
	# average reward estimate
	R_bar = 0

Init S, A
for each step:
	Take A, observe R, S'
	choose A' from Q(S,*,w) -> episilon greedy
	error = R - R_bar + Q(S',A',w) - q(S,A,w)
	R = R_bar + beta*error
	w = w + alpha*error*nabla_Q(S,A,w)
	S = S'
	A = A'
```

# Differential semi-gradient n-step Sarsa
```
Parameters
	# num features
	d = num features/len(w)
	# differentiable action-value function
	Q = [arbitrary value for s in S for a in A[s] for i in d]
	# weights
	w = [0 for _ in d]
	# Average reward estimate
	R_bar = 0
	# stepsize
	alpha > 0
	beta > 0
	# number of steps
	n is int, n >= 0
	"All store and access opertations (S[t], A[t], R[t]_) can take their index mod n+1" (?)

Init A[0], S[0]

Loop for each step t = 0,1,2,...:
	take action A[t], observe R[t+1], S[t+1]
	Select A[t+1] ~ argmaxa(pi(*,S[t+1]) or epsilon-greedy wrt Q[S[t+1], *, w)
	tau = t - n + 1 # tau is the time whose estimate is being updated
	if tau >= 0:
		estimate = sum(from i = tau+1 to tau+n: (R[i] - R_bar) + Q(S[tau+n],
			A[tau+n],w) - Q(S[tau], A[tau], w)
		R_bar = R + beta*estimate
		w = w + alpha*estimate*nabla_Q(S[t], A[t], w)
```

# No idea where to put this

```
[a,b,c]^transpose =  a
                   [ b ]
                     c

G[t:t+n] = n-step return from t+1 to t+n
G[t:h] = n-step return from t+1 to h

# undiscounted MC (all step) return (episodic)
G[T] = 0
G[t] = R[t+1] + R[t+2] + ... + R[T]
# discounted 1-step return (continuing)
G[t] = R[t+1] + gamma*R[t+2] + gamma**2 * R[t+3] + ...
     = sum([gamma**k * R[t+k+1] for k in range(0, infinity)])
     = R[t+1] + gamma*G[t+1]
     = sum([gamma**(k-t-1) * R[k] for k in range(t+1, T+1)]) # T can be inifinite or gamma can be 1 but not both
	 = G[t:t+1]
	 = R[t+1] + gamma*V[t](S[t+1])
# discounted n-step return (continuing)
G[t:h] = R[t+1] + gamma*G[t+1:h]
G[t:t+n] = R[t+1] + gamma*R[t+2] + ... + gamma**(n-1) * R[t+n] +
	gamma**n * Q[t+n-1](S[t+n], A[t+n]), n >= 1, 0 <= t < T=n

```
nabla\_f(**w**) = [ df(**w**)/dw\_1, df(**w**)/dw\_2 , ... ]^transpose
df(**w**)/dw\_x = derivative of f(**w**) wrt w\_x.
nablaf(**w**) = gradient of f wrt **w**.

In linear methods, we have **x**(s) = \[x\[i]\[s] for i in tiles per tiling/# features]
in this case,

nablav\_hat[(s,**w**)] = **x**(s)
nablaa\_hat(s,a,**w**) = **x**[(s,a)]
q\_hat(s,a,**w**) = **w**^transpose **x**(s,a) = `sum([w[i]*x[i][(s,a)] for i in [1,d]]`
  * Where `d` is the # of features aka the size of **w**

# Eligibility traces
* TD(lambda)
  * Lambda = 0 -> TD(0)
  * Lambda = 1 -> MC
* Offers computational advantages over n-step TD for unifying TD(0) to MC
* *Eligibility trace*, **z**, where len(z) = len(w) = d
* When a component of w is used to calculate an estimate, the corresponding
component of z is bumped up and begins to fade away.
* Learning occurs inside that component of w <=> a nonzero TD error occurs
before the trace falls back to zero.
* Lambda = trace-decay parameter in [0,1] which determines how fast it falls
* N-step = "forward views"
* eligibility trace = "backward views"
```
# episodic lambda-return
G(Lambda)[t] = (1-lambda)*(sum(from n=1 to T-t-1: lambda**(n-1)*G[t:t+n])
	+ lambda**(T-t-1)*G[t]
```

# Dutch traces
```
# auxilary vectors
a[0] = w[0]
a[t] = a[t-1] - alpha * x[t]*transpose(x[t])*a[t-1]
z[0] = x[0]
z[t] = z[t-1] + (1-alpha*transpose(z[t-1])*x[t])*x[t] + x[t]
# weight vector
w[t+1] = w[t] + alpha*(G[t] - transpose(w[t])*x[t])*x[t], 0 <= t < T (? eq. 12.13)
```

* The short: Online lambda-return is doable but too computationally expensive to
feasibly be run online. "True online TD(lambda)" is feasible but dutch traces
are even cheaper. (True -> because it is fast enough to be run online, which is
more "true" to the spirit of online learning)

# Sarsa lambda
```
# update rule
error[t] = R[t+1] + gamma*Q(S[t+1], A[t+1], w[t]) - Q(S[t], A[t], w[t])
w[t+1] = w[t] + alpha*error[t]*z[t]
# eligibility trace
z[-1] = [0 for _ in d]
z[t] = gamma*lambda*z[t-1] + nabla_Q(S[t], A[t], w[t]), asser(0<=t<=T)
```

Image from the book on p. 304 describes the differences well.
* Imagine an agent has finished an episode
* 1-step sarsa (sarsa(0)) updates the action-value of the action taken in the
	last step (right next to the goal)
* 10-step sarsa (n-step sarsa) updates **equally** the action-values of the last
10 actions from the goal.
* Sarsa(lambda) updates all of the action-values, with different degrees. The
one closest to the goal is changed the most and the change reduces with each
step.

* NOTE: the below can be modified to use dutch traces (as per exercise 12.6)..
figure that out

# Sarsa(lambda) with binary features and linear function approximation
```
Parameters
	some funciton F s.t. F(s,a) = set of indices of active features for (s, a)
	alpha > 0
	0 <= lambda <= 1
	d = num features/size of w
	w = [0 for _ in d]
	z = transpose([arbitrary value for _ in d])

Loop for each episode:
	init S
	choose A epsilon greedily according to Q(S,*,w)
	z = [0 for _ in d]
	Loop for each step of episode:
		Take A, observe R, S'
		error = R
		Loop for i in F(S,A):
			error = error - w[i]
			if (accumulating traces)
				z[i] = z[i] + 1
			elif(replacing traces)
				z[i] = 1
	
		If S' is terminal:
			w = w + alpha*error*z
			goto next episode

		Choose A' epsilon greedily from Q(S,*,w)
		Loop for i in F(S', A'):
			estimate = estimate + gamma*w[i]

		w = w + alpha*estimate*z
		z = gamma*lambda*z
		S = S'
		A = A'
```

# True online Sarsa(lambda) for estimating transpose(w)x ~ q\_pi or q\*
```
Parameters
	# feature function
	x = [[binary features for _ in d] for s in S+ for a in A]
	x(terminal, *) = [0 for _ in d]
	# step size
	alpha > 0
	# trace decay
	0 <= lambda <= 1
	# weights
	w = [0 for _ in d]

Loop for each episode:
	init S
	Choose A from epsilon greedy S,*,w
	x = x(S,A)
	z = [0 for _ in d]
	Q_old = 0
	Loop for each step of episode:
		Take action A, observe R, S'
		Choose A' epsilon greedy S',*,w
		x' = x(S',A')
		Q = transpose(w)x
		Q' = transpose(w)x'
		error = R = gamma*Q' - Q
		z = gamma*lambda*z + (1 -alpha*gamma*lambda*tranpose(z)*x)*x
		w = w + alpha*(error + Q - Q_old)*z - alpha*(Q-Q_old)*x
		Q_old = Q'
		x = x'
		A = A'
	until S' is terminal
```

# The termination function
* Having a different gamma (`gamma[t] = gamma(S[t])`) and lambda
	(`lambda[t] = lambda(S[t], A[t]) for each time step.
* General definition of return
```
G[t] = R[t+1] + gamma[t]*G[t+1]
     = R[t+1] + gamma[t]*R[t+2] + gamma[t+1]*gamma[t+2]*R[t+3] + ...
     = sum(k=t to infinity: (prod(i=t+1 to k: gamma[i]))*R[k+1])
```
* lambda-return
```
# state-based
G_lambdas[t] = R[t+1] + gamma[t+1] * ((1-lambda[t+1])*v(S[t+1],w[t])
	+ lambda[t+1]*G_lambdas[t+1])
# Sarsa
G_lambdaa[t] = R[t+1] + gamma[t+1] * ((1-lambda[t+1])*q(S[t+1],A[t+1],w[t])
	+ lambda[t+1]*G_lambdaa[t+1])
# Expected sarsa
G_lambdaa[t] = R[t+1] + gamma[t+1] * ((1-lambda[t+1])*V[t](S[t+1])
	+ lambda[t+1]*G_lambdas[t+1])
V[t](s) = sum(for all a: pi(a|s)*q(s,a,w[t]))
```
