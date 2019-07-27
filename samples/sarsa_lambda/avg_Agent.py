# Agent
# Based off the agent define in the RLGlue Software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0

# Modified by Noah Burghardt
import numpy as np
from tiles3 import tiles, IHT

# Agent class to represent a particular RL strategy
# MC first visit on-policy control
class Agent:

	# Set up parameters for agent
	# State-action pair values
	# Use a function defined below for Q instead of a dict
	#Q = None
	R_bar = 0

	# Action space, to be configured by the environment
	A = None

	# Agent configuration variables
	epsilon = 0.1

	# Index hash table / number of features
	hash_table_size = 2048
	# number of tilings / number of feature fields
	num_offset_tilings = 8
	# length of one side of tiling
	tiling_side_length = np.sqrt(hash_table_size)

	# weight vector
	w = [0]*hash_table_size
	iht = IHT(hash_table_size)

	# step size
	alpha = 0.1/num_offset_tilings
	beta = 0.1/num_offset_tilings

	tilecache = {}

	# min and max values for environment
	mins = []
	maxs = []

	last_action = None
	last_state = None


	# returns list of feature indices for the given state
	# [feature# for field 1, feature# for field2, ..., feature# for last field]
	# This is a list of all the features with value 1 (activated)
	def active_features(self, state, action):
		sap = (state,action)
		if (sap in self.tilecache):
			return self.tilecache[sap]
		else:
			scaleFactor = [self.tiling_side_length /
				(self.maxs[i]-self.mins[i])
				for i in range(len(self.maxs))]
			# Use distinct integer actions in tiling, resulting in a different
			# tiling for each action
			t = tiles(self.iht, self.num_offset_tilings,
				[state[i]*scaleFactor[i] for i in range(len(state))], [action])
			self.tilecache[sap] = t
			return t
	
	# Declare agent variables
	def __init__(self, actions, max_state, min_state):
		self.R_bar = 0
		self.A = actions
		self.mins = min_state
		self.maxs = max_state

	# Initialize agent variables
	# Run once, in experiments
	def agent_init(self):
		# Assume there are no states (new states can be valued lazily)
		self.R_bar = 0
		self.pi = {}
		self.returns = {}
		self.v = {}
		self.last_action = None
		self.last_state = None

	# Start agent
	# Runs at the beginning of an episode. The first method called when the experiment
	# starts, called after the environment starts
	# Args:
	#	state (state observation): The agent's current state
	# returns:
	#	The first action the agent takes
	def agent_start(self, state):
		# start episode
		self.last_state = state
		self.last_action = self.epGreedy(state)

		return self.last_action

	# A step taken by the agent
	# Args:
	#	reward (float): the reward received for taking the last action taken
	#	state (state observation): The agen's current state
	# returns:
	#	The action the agent is taking
	def agent_step(self, reward, state):
		Sprime = state
		R = reward
		Aprime = self.epGreedy(Sprime)
		S = self.last_state
		A = self.last_action
		w = self.w
	
		#print(S,A)
		#input(self.active_features(S,A))

		error = R - self.R_bar + self.Q(Sprime, Aprime, w) - self.Q(S,A,w)
		self.R_bar = self.R_bar + self.beta * error
		# w = w + alpha*error*nabla_q(S,A,w)
		# w = w + alpha*error*x(S,A)
		# since x(S,A)[feature] is 0 on inactive features, we just wont update
		# on inactive features, since the update reduces to w = w.
		# on active features x(S,A)[feature] is 1 and the update reduces to
		# w = w + alpha*error

		for feature in self.active_features(S,A):
			self.w[feature] += self.alpha*error

		self.last_state = Sprime
		self.last_action = Aprime
		return self.last_action

	# Run when the agent termintates
	# Args:
	#	reward (float): the reward the agent received for entering the terminal state
	def agent_end(self, reward):
		R = reward
		S = self.last_state
		A = self.last_action
		w = self.w

		# Compute error with the action-value of the terminal state being equal
		# to zero
		error = R - self.R_bar - self.Q(S,A,w)
		self.R_bar = self.R_bar + self.beta * error
		# w = w + alpha*error*nabla_q(S,A,w)
		# w = w + alpha*error*x(S,A)
		# since x(S,A)[feature] is 0 on inactive features, we just wont update
		# on inactive features, since the update reduces to w = w.
		# on active features x(S,A)[feature] is 1 and the update reduces to
		# w = w + alpha*error

		for feature in self.active_features(S,A):
			self.w[feature] += self.alpha*error

	# Receive a message form RLGlue
	# Args:
	#	Message (str): the message passed
	# returns:
	#	response (str): The agent's response to the message (optional)
	def agent_message(self, message):
		# Output what the agent thinks is the optimal policy
		if message == "action-values":
			return None

	
	def pi_select(self, pi, s, A):
		rand = np.random.random()
		probs = self.pi.get(s,{a:1/len(A) for a in A})
		val = 0
		for a in A:
			val += probs[a]
			if rand < val:
				return a

	def epGreedy(self, S):
		rand = np.random.random()
		if (rand > self.epsilon):
			return np.random.choice(self.A)
		else:
			maxQ = max([self.Q(S,a,self.w) for a in self.A])
			max_actions = [a for a in self.A if self.Q(S,a,self.w) >= maxQ]
			return np.random.choice(max_actions)


	def Q(self, state, action, w):
		features = self.active_features(state, action)
		#return np.sum(w[features])
		return sum([w[i] for i in features])
	



	# returns the action with the highest action-value
	# Args:
	#	Q: Action values
	#	s: state to evaluate
	#	A: Actions to pick from
	def argmaxa(self, Q, s, A):
		
		# Find the highest action value
		maxQ = max([Q.get((s,a),0) for a in A])
		# Find actions with highest value
		best_actions = [a for a in A if Q.get((s,a),0) == maxQ]
		# consistently pick from best actions
		return sorted(best_actions)[0]

			
	# returns the action with the highest action-value
	# Args:
	#	Q: Action values
	#	s: state to evaluate
	#	A: Actions to pick from
	def argmaxa_rand(self, Q, s, A):
		# Find the highest action value
		maxQ = max([Q.get((s,a),0) for a in A])
		# Find actions with highest value
		best_actions = [a for a in A if Q.get((s,a),0) == maxQ]
		# Randomly pick from actions with top action value
		return np.random.choice(best_actions)
