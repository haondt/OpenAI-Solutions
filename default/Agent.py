# Agent
# Based off the agent define in the RLGlue Software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0

# Modified by Noah Burghardt
import numpy as np

class Agent:
	"""
	Defines the interface of an RLGlue Agent

	ie. These methods must be defined in your own Agent classes
	"""
	
	def __init__(self, width, height, moveset='normal'):
		"""Declare agent variables."""
		pass

	def agent_init(self):
		"""
		run once, in experiment
		Initialize agent variables.
		"""
		pass

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
		return None

	def agent_step(self, reward, state):
		"""
		A step taken by the agent.
		Args:
			reward (float): the reward received for taking the last action taken
			state (state observation): The agent's current state
		Returns:
			The action the agent is taking.
		"""
		return None
	
	def agent_end(self, reward):
		"""
		Run when the agent terminates.
		Args:
			reward (float): the reward the agent received for entering the
				terminal state.
		"""
		pass

	def agent_message(self, message):
		"""
		receive a message from rlglue
		args:
			message (str): the message passed
		returns:
			str : the agent's response to the message (optional)
		"""

