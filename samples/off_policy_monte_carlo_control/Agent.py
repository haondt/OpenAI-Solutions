# Agent
# Based off the agent define in the RLGlue Software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0

# Modified by Noah Burghardt
import numpy as np

# Agent class to represent a particular RL strategy
class Agent:

	Q = None
	C = None
	pi = None

	# Variables to hold the execution of an episode
	ep_states = None
	ep_actions = None
	ep_rewards = None

	# State space for entire model
	# Start empty, add states as we encounter them
	S = None
	# Action space for any state in cartpole is 0 or 1 (left/right)
	A = None
	
	# Declare agent variables
	def __init__(self):
		Q = {}
		S = {}
		A = []
		C = {}
		pi = {}

	# Initialize agent variables
	# Run once, in experiments
	def agent_init(self):
		# Assume there are no states (new states can be valued lazily)
		Q = {}
		C = {}
		pi = {}
		S = {}
		A = [0, 1]

	# Start agent
	# Runs at the beginning of an episode. The first method called when the experiment
	# starts, called after the environment starts
	# Args:
	#	state (state observation): The agent's current state
	# Returns:
	#	The first action the agent takes
	def agent_start(self, state):
		# If state is new, add to action values and state list
		if state not in self.S.values():
			self.states[state] = 0
			for a in self.A:
				self.Q[(state,a)] = 0

		# start episode
		a = self.epGreedy(self.epsilon, self.Q, state, self.A)

		# Reset episode data and store new
		self.ep_states = [state]
		self.ep_actions = [a]

		return a

	# A step taken by the agent
	# Args:
	#	reward (float): the reward received for taking the last action taken
	#	state (state observation): The agen's current state
	# Returns:
	#	The action the agent is taking
	def agent_step(self, reward, state):
		# Choose ep_greedy action
		a = self.epGreedy(self.epsilon, self.Q, state, self.A)

		# Save episode step
		self.ep_states.append(state)
		self.ep_actions.append(a)
		self.ep_rewards.append(reward)

		return a
	
	# Run when the agent termintates
	# Args:
	#	reward (float): the reward the agent received for entering the terminal state
	def agent_end(self, reward):
		# finish off episode
		self.ep_rewards.append(reward)

		# set up variables for processing episode
		G = 0
		W = 1

		# Loop through steps in episode
		T = len(self.ep_states)-1
		for t in range(T-1, -1,-1):
			G = self.gamma

	# Receive a message form RLGlue
	# Args:
	#	Message (str): the message passed
	# Returns:
	#	response (str): The agent's response to the message (optional)
	def agent_message(self, message):
		pass

	# Perform an action selection using an epsilon-greedy selection
	# Args
	#	eps: epsilon value
	#	Q: action values to base selection off of
	#	A: actions to choose from
	#	s: current state
	# Returns:
	#	action (action): action chosen
	def epGreedy(self, eps, Q, s, A):
		# Best action with probability 1-eps
		if np.random.random() > eps:
			return self.argmax(Q, s, A)

		# Choose an action at random with porbability eps
		else
			return np.random.choice(A)


	# Returns the action with the highest action-value
	# Args:
	#	Q: Action values
	#	s: state to evaluate
	#	A: Actions to pick from
	def argmax(self, Q, s, A):
		# Find the highest action value
		maxQ = max([Q[(s,a)] for a in A])
		# Find actions with highest value
		best_actions = [a for a in A if Q[(s,a)] == maxQ]
		# Randomly pick from actions with top action value
		return np.random.choice(best_actions)
			
