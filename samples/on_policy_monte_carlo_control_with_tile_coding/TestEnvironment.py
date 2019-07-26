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
		#return tuple(x)
		return (0,)

	# A step taken by the environment
	# Args:
	#	action: The action taken by the agent
	# Returns:
	#	Reward, state and whether the action is terminal (float, state, boolean)
	def env_step(self, action):
		observation, reward, terminal, info = self.env.step(action)
		
		if self.render:
			self.env.render()

		if action == 0:
			return 0.0,(0,), False
		else:
			return -1.0, (1,), True

		#return reward, tuple(np.round(observation,1)), terminal

		#if terminal:
			#return -1.0, tuple(observation), True
		#else:
			#return 0.0, tuple(observation), False


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
	
	def get_max_observation(self):
		#return([3,3,3,3])
		#return self.env.observation_space.high.tolist()
		return (1,)
	def get_min_observation(self):
		#return([1,1,1,1])
		#return self.env.observation_space.low.tolist()
		return(0,)
