# Agent
# Based off the agent define in the RLGlue Software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0
# Modified by Noah Burghardt

# Implementation of the Tabular Dyna-Q agent defined in Chapter 8 
# of Sutton and Barto
import numpy as np

class Agent:
	"""
	Defines the interface of an RLGlue Agent

	ie. These methods must be defined in your own Agent classes
	"""

	planningSteps = 5
	lastState = None
	lastAction = None
	Q = None
	model = None
	epsilon = 0.1
	stepSize = 0.1
	gamma = 0.95
	observedStates = None
	
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
		self.Q = {(state, action): 0 for state in np.append(self.states,[-1,-2]) for action in self.actions}
		self.model = {(state,action):{} for state in self.states for action in self.actions}
		self.observedStates= {}

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
		alpha = self.stepSize
		R = reward
		self.Q[(S,A)] += alpha*(R + self.gamma*max([self.Q[(Sp, a)] for a in self.actions]) - self.Q[(S,A)])
		# update observed states
		if S not in self.observedStates:
			self.observedStates[S] = []
		if A not in self.observedStates[S]:
			self.observedStates[S].append(A)
		
		# update model
		if (R, Sp) not in self.model[(S,A)]:
			self.model[(S,A)][(R,Sp)] = 0
		self.model[(S,A)][(R,Sp)] += 1
		
		# plan
		self.plan(reward)

		#print(self.lastState, self.lastAction, state)
		# set up for next step
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
		alpha = self.stepSize
		R = reward
		self.Q[(S,A)] += alpha*(R - self.Q[(S,A)])
		
		# update observed states
		if S not in self.observedStates:
			self.observedStates[S] = []
		if A not in self.observedStates[S]:
			self.observedStates[S].append(A)
		
		# update model
		# state -1 = goal
		# state -2 = hole
		if (R, R-1) not in self.model[(S,A)]:
			self.model[(S,A)][(R,R-1)] = 0
		self.model[(S,A)][(R,R-1)] += 1
		

		# plan
		self.plan(reward)

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

	def plan(self, reward):
		for _ in range(self.planningSteps):
			S = self.lastState
			A = np.random.choice(self.observedStates[S])
			# find most common state
			print(self.model[(S,A)])
			input()
			R, Sp = max(list(self.model[(S,A)].keys()), key=lambda x: self.model[(S,A)][x])
			
			self.Q[(S,A)] += self.stepSize*(reward + self.gamma*max([self.Q[(Sp, a)] for a in self.actions]) - self.Q[(S,A)])
