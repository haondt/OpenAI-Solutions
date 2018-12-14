# Agent
# Based off the agent define in the RLGlue Software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0
# Modified by Noah Burghardt

# Q-learning
import numpy as np

class Agent:
	"""
	Defines the interface of an RLGlue Agent

	ie. These methods must be defined in your own Agent classes
	"""

	epsilon = 0.1
	Q = None
	alpha = 0.5
	gamma = 0.9

	# actions are defined as
	# 0: left
	# 1: down
	# 2: right
	# 3: up
	actions = list(range(4))

	# states are defined as 64 discrete numbers
	states = np.linspace(0,63,64)
	
	def __init__(self):
		"""Declare agent variables."""
		pass

	def agent_init(self):
		"""
		run once, in experiment
		Initialize agent variables.
		"""
		self.Q = {(state, action): 0 for state in self.states for action in self.actions}

	def agent_start(self, state):
		"""
		run at the beginning of an episode
		The first method called when the experiment starts, called after
		the environment starts.
		Args:
			state (state observation): The agent's current state

		Returns:
			The first action the agent takes.
		"""
		self.lastState = state
		self.lastAction = self.epGreedy(state)
		return self.lastAction

	def agent_step(self, reward, state):
		"""
		A step taken by the agent.
		Args:
			reward (float): the reward received for taking the last action taken
			state (state observation): The agent's current state
		Returns:
			The action the agent is taking.
		"""
		# perform update
		S = self.lastState
		A = self.lastAction
		Sp = state
		R = reward
		
		self.Q[(S,A)] += self.alpha*(R + self.gamma*max([self.Q[(Sp, a)] for a in self.actions]) - self.Q[(S,A)])
		self.lastAction = self.epGreedy(state)
		self.lastState = state
		
		return self.lastAction
	
	def agent_end(self, reward):
		"""
		Run when the agent terminates.
		Args:
			reward (float): the reward the agent received for entering the
				terminal state.
		"""
		#print(self.lastState, self.lastAction, -1)
		#input()
		# perform update
		S = self.lastState
		A = self.lastAction
		R = reward
		self.Q[(S,A)] += self.alpha*(R - self.Q[(S,A)])
		

	def agent_message(self, message):
		"""
		receive a message from rlglue
		args:
			message (str): the message passed
		returns:
			str : the agent's response to the message (optional)
		"""
		pass

	def epGreedy(self, state):
		if np.random.rand() < self.epsilon: # prob ep
			# choose random action
			return np.random.choice(self.actions)
		else: # prob 1-ep
			# choose best action with random tie breaking
			maxQ = max([self.Q[(state,a)] for a in self.actions])
			maxAs = [a for a in self.actions if self.Q[(state,a)] == maxQ]
			return np.random.choice(maxAs)

