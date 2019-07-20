# Environment
# Based of the environment defined in the RLGlue software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0

# Modified by Noah Burghardt
import numpy as np

# Defines the interface of an RLGlue environment
class Environment:

	# Declare environment variables
	# Run once, in experiment
	def __init__(self, width, height):
		pass

	# Initialize environment variables
	# Run once, in experiment
	def env_init(self):
		pass

	# Start environment
	# Run at the beginning of each episode
	# The first method call when the experiment starts, called before the
	# agent starts.
	# Returns:
	#	The first state observation form the environment
	def env_start(self):
		return None

	# A step taken by the environment
	# Args:
	#	action: The action taken by the agent
	# Returns:
	#	Reward, state and whether the action is terminal (float, state, boolean)
	def env_step(self, action):
		return None

	# Receive a message from RLGlue
	# Args:
	#	message (str): the message passed
	# Returns:
	#	message (str): the environment's response to the message (optional)
	def env_message(self, message):
		pass
