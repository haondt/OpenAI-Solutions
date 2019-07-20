# Agent
# Based off the agent define in the RLGlue Software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0

# Modified by Noah Burghardt
import numpy as np

# Agent class to represent a particular RL strategy
class Agent:
	
	# Declare agent variables
	def __init__(self):
		"""Declare agent variables."""
		pass

	# Initialize agent variables
	# Run once, in experiments
	def agent_init(self):
		pass

	# Start agent
	# Runs at the beginning of an episode. The first method called when the experiment
	# starts, called after the environment starts
	# Args:
	#	state (state observation): The agent's current state
	# Returns:
	#	The first action the agent takes
	def agent_start(self, state):
		return None

	# A step taken by the agent
	# Args:
	#	reward (float): the reward received for thaking the last action taken
	#	state (state observation): The agen's current state
	# Returns:
	#	The action the agent is taking
	def agent_step(self, reward, state):
		return None
	
	# Run when the agent termintates
	# Args:
	#	reward (float): the reward the agent received for entering the terminal state
	def agent_end(self, reward):
		pass

	# Receive a message form RLGlue
	# Args:
	#	Message (str): the message passed
	# Returns:
	#	response (str): The agent's response to the message (optional)
	def agent_message(self, message):
		pass

