# Environment
# Based of the environment defined in the RLGlue software
# https://sites.google.com/a/rl-community.org/rl-glue/Home?authuser=0

# Environment wrapper for CartPole-v1

# Modified by Noah Burghardt
import numpy as np
import gym

# Defines the interface of an RLGlue environment
class Environment:
	
	env = None
	render = None
	precision = 1

	# Declare environment variables
	# Run once, in experiment
	def __init__(self):
		self.render = None
		self.env = None

	# Initialize environment variables
	# Run once, in experiment
	def env_init(self):
		self.render = False
		self.env = gym.make('CartPole-v1')

	# Start environment
	# Run at the beginning of each episode
	# The first method call when the experiment starts, called before the
	# agent starts.
	# Returns:
	#	The first state observation form the environment
	def env_start(self):
		x = self.env.reset()
		if self.render:
			self.env.render()
		return tuple(np.round(x,self.precision))

	# A step taken by the environment
	# Args:
	#	action: The action taken by the agent
	# Returns:
	#	Reward, state and whether the action is terminal (float, state, boolean)
	def env_step(self, action):
		observation, reward, terminal, info = self.env.step(action)
		
		if self.render:
			self.env.render()

		return reward, tuple(np.round(observation,self.precision)), terminal

	# Receive a message from RLGlue
	# Args:
	#	message (str): the message passed
	# Returns:
	#	message (str): the environment's response to the message (optional)
	def env_message(self, message):
		if message == 'renderON':
			self.render = True
		elif message == 'renderOFF':
			self.render = False
		elif message == 'close':
			self.env.close()
	
	def get_actions(self):
		return [0,1]
