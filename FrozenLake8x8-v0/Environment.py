# Environment
# Based of the environment defined in the RLGlue software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0

# Modified by Noah Burghardt
import numpy as np

class Environment:
	"""
	Defines the interface of an RLGlue environment

	ie. These methods must be defined in your own environment classes
	"""

	def __init__(self, width, height):
		"""
		(run on initialization)
		Declare environment variables.
		"""
		pass

	def env_init(self):
		"""
		Initialize environment variables.
		(run once in experiment)
		"""
		pass

	def env_start(self):
		"""
		(run at the beginning of each episode)
		The first method called when the experiment starts, called before the
		agent starts.

		Returns:
			The first state observation from the environment.
		"""
		return None

	def env_step(self, action):
		"""
		A step taken by the environment.

		Args:
			action: The action taken by the agent

		Returns:
			(float, state, Boolean): a tuple of the reward, state observation,
				and boolean indicating if it's terminal.
		"""
		return None

	def env_message(self, message):
		"""
		receive a message from RLGlue
		Args:
		   message (str): the message passed
		Returns:
		   str: the environment's response to the message (optional)
		"""
		pass
