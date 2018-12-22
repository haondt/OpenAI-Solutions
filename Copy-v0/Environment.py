# Environment
# Based of the environment defined in the RLGlue software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0

# Modified by Noah Burghardt
import numpy as np
import gym
class Environment:
	"""
	Defines the interface of an RLGlue environment

	ie. These methods must be defined in your own environment classes
	"""
	env = None
	render = None

	def __init__(self):
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
		self.env = gym.make('Copy-v0')
	def env_start(self):
		"""
		(run at the beginning of each episode)
		The first method called when the experiment starts, called before the
		agent starts.

		Returns:
			The first state observation from the environment.
		"""
		x = self.env.reset()
		if self.render:
			self.env.render()

		return x

	def env_step(self, action):
		"""
		A step taken by the environment.

		Args:
			action: The action taken by the agent

		Returns:
			(float, state, Boolean): a tuple of the reward, state observation,
				and boolean indicating if it's terminal.
		"""	
		obs, reward, terminal, info = self.env.step(action)
		
		if self.render:
			self.env.render()
		
		return reward, obs, terminal

	def env_message(self, message):
		"""
		receive a message from RLGlue
		Args:
		   message (str): the message passed
		Returns:
		   str: the environment's response to the message (optional)
		"""
		if message == 'renderON':
			self.render = True
		elif message == 'renderOFF':
			self.render = False
		elif message == 'close':
			self.env.close()
