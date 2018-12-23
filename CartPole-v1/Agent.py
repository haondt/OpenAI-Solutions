# Agent
# Based off the agent define in the RLGlue Software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0
# Modified by Noah Burghardt

# Monte Carlo
import numpy as np
import random
class Agent:
	"""
	Defines the interface of an RLGlue Agent

	ie. These methods must be defined in your own Agent classes
	"""

	epsilon = 0.1
	Q = None
	Returns = None
	S = []
	R = []
	A = []
	pi = {}
	alpha = 0.5
	gamma = 0.9
	actions = [(x,y,z)
				for x in range(2)
				for y in range(2)
				for z in range(5)
					]
	states = [i for i in range(6)]
	def __init__(self):
		"""Declare agent variables."""
		pass

	def agent_init(self):
		"""
		run once, in experiment
		Initialize agent variables.
		"""
		self.Q = {(state, action): 0 for state in self.states for action in self.actions}
		self.Returns = {(state, action): [] for state in self.states for action in self.actions}
		self.pi = {(action, state): [] for state in self.states for action in self.actions}
	
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
		# start episode
		action = self.epGreedy(state)
		
		self.S = [state]
		self.A = [action]
		self.R = [0]
		return action

	def agent_step(self, reward, state):
		"""
		A step taken by the agent.
		Args:
			reward (float): the reward received for taking the last action taken
			state (state observation): The agent's current state
		Returns:
			The action the agent is taking.
		"""

		# Generate episode	
		action = self.epGreedy(state)

		self.R.append(reward)
		self.S.append(state)
		self.A.append(action)

		return action
	
	def agent_end(self, reward):
		"""
		Run when the agent terminates.
		Args:
			reward (float): the reward the agent received for entering the
				terminal state.
		"""
		# finish off episode
		self.R.append(reward)

		G = 0
		# find index of terminal state
		T = len(self.R)-1

		for t in range(T-1,-1,-1):
			G = self.gamma*G + self.R[t+1]
			if not (self.S[t] in self.S[0:t] and self.A[t] in self.A[0:t]):
				self.Returns[(self.S[t], self.A[t])].append(G)
				self.Q[(self.S[t], self.A[t])] = np.average(self.Returns[(self.S[t], self.A[t])])
				
				maxQ = max([self.Q[(self.S[t],a)] for a in self.actions])
				Astar = random.choice([a for a in self.actions if self.Q[(self.S[t], a)] == maxQ])

				for a in self.actions:
					if a == Astar:
						self.pi[(a,self.S[t])] = 1 - self.epsilon + self.epsilon/len(self.actions)
					else:
						self.pi[(a,self.S[t])] = self.epsilon/len(self.actions)

	def agent_message(self, message):
		"""
		receive a message from rlglue
		args:
			message (str): the message passed
		returns:
			str : the agent's response to the message (optional)
		"""
		if message == 'policy':
			policy = {}
			for state in self.states:
				policy[state] = max(self.actions, key=lambda x: self.pi[(x,state)])
			return policy
	
	def epGreedy(self, state):
		if np.random.random() > self.epsilon:
			maxQ = max([self.pi[(a,state)] for a in self.actions])
			actions = [a for a in self.actions if self.pi[(a,state)] == maxQ]
			achoice = np.random.choice(list(range(len(actions))))
			return actions[achoice]
		else:
			return random.choice(self.actions)
