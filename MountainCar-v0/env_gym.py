# Modified to use openai gym
import numpy as np
import gym
import time
from rl_glue import BaseEnvironment


class Environment(BaseEnvironment):

	env = None
	render = False

	def __init__(self):
		pass
	def env_init(self):
		self.env = gym.make('MountainCar-v0')
		self.env.seed(0)
	def env_start(self):
		obs = self.env.reset()
		return obs

	def env_step(self, action):
		if self.render:
			self.env.render()
			time.sleep(0.01)
		state, reward, done, info = self.env.step(action)
		return reward, state, done

	def env_message(self, message):
		if message == "rON":
			self.render = True
		elif message == "rOFF":
			self.render = False
	
	def close(self):
		self.env.close()

